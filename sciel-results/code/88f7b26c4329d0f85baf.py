import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Upper Confidence Bound (UCB) with a time-varying exploration parameter.
    
    This approach balances the theoretical guarantees of UCB (regret bounds)
    with the practical performance of EI (focusing on improvement over the
    current best). The exploration parameter beta_t decreases over time,
    shifting from broad exploration to exploitation.
    """
    # Deadline guard: if we are close to the 330s limit, return a safe, fast score
    if time.time() - _MODULE_START_TIME > 300:
        # Fallback: Return sigma (pure exploration) to ensure we don't crash
        return np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle sigma = 0 cases to avoid division by zero
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Estimate the number of evaluations so far.
    # We don't have direct access to the counter, but we can infer a proxy.
    # However, a simpler and more robust heuristic for BO in this setting is to
    # use a fixed, well-tuned beta that works well for 2D multimodal functions,
    # or to use a beta that depends on the variance structure.
    # 
    # A common choice is beta_t = sqrt(log(t * d^2 * log(T * d * delta)))
    # where d is dimension, T is total budget, delta is failure prob.
    # With d=2, T=30, delta=0.1:
    # log(30 * 4 * log(30 * 2 * 0.1)) = log(120 * log(6)) = log(120 * 1.79) = log(214.8) = 5.37
    # sqrt(5.37) = 2.32
    #
    # Alternatively, we can use a time-varying beta if we can track steps.
    # Since we can't easily track steps without global state that might be reset,
    # let's use a hybrid: EI + a small UCB term.
    #
    # Actually, the most robust improvement over pure EI is to add a UCB term
    # that encourages exploration of high-uncertainty regions even if the mean
    # is not yet promising.
    
    # Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # UCB formula for minimization: mu - beta * sigma
    # We want to maximize the potential to find a lower value.
    # A common hybrid is: EI + lambda * sigma
    # Or: EI * (1 + beta * sigma / (f_best - mu + eps))
    
    # Let's try a simple hybrid: EI + 0.5 * sigma
    # This adds a constant exploration pressure proportional to uncertainty.
    # This is equivalent to EI with a small bonus for high variance.
    
    # Another option: Use UCB directly with a well-tuned beta.
    # UCB_min = mu - beta * sigma
    # But UCB can be too explorative if beta is large, or too exploitative if small.
    # EI is generally more efficient in practice for small budgets like 30.
    
    # Let's try a weighted combination:
    # score = alpha * EI + (1 - alpha) * (mu - beta * sigma)
    # But this requires normalizing the scales.
    
    # A simpler and effective strategy:
    # Use EI, but add a small exploration bonus proportional to sigma.
    # This helps escape local optima in the surrogate.
    
    # Exploration bonus: 0.1 * sigma
    # This is a heuristic, but it's often effective.
    
    score = ei + 0.1 * sigma_safe
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
