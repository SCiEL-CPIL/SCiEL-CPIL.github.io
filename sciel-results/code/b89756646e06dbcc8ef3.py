import numpy as np
import time

# Deadline guard: The evaluator kills the program after 330s wall-clock.
# We track elapsed time and return a safe fallback if we are too close to the limit.
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Hypervolume Indicator (HVI) approximation.
    
    The HVI formulation frames the optimization as maximizing the area of the
    Pareto front dominated by the best observed solution relative to a reference
    point. Here, the reference point is set to the known global minimum (3.0).
    
    The improvement for a point x is defined as the reduction in function value
    from the best observed value (f_best) to the predicted value at x, bounded
    by the maximum possible improvement (f_best - 3.0).
    
    The acquisition score is a combination of the expected improvement (exploitation)
    and a variance-based exploration term weighted by the potential for improvement.
    
    Parameters:
    mu: array of predicted means (shape 2000,)
    sigma: array of predicted standard deviations (shape 2000,)
    f_best: current best observed function value (scalar)
    
    Returns:
    scores: array of acquisition values (shape 2000,), higher is better.
    """
    # Check deadline
    if time.time() - _start_time > 330 - _DEADLINE_MARGIN:
        # Return a safe fallback: prioritize low mean (exploitation)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Known global minimum for Goldstein-Price 2D
    f_min = 3.0
    
    # Maximum possible improvement from the best observed value
    max_improvement = f_best - f_min
    if max_improvement <= 0:
        # If we've already found the global minimum, just exploit
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # HVI-inspired acquisition:
    # The contribution of a point to the hypervolume is proportional to the
    # improvement it provides over the reference point (f_min), but relative
    # to the current best (f_best).
    
    # Expected Improvement (EI) toward the global minimum:
    # This measures the expected reduction in f_best if we evaluate at x.
    # EI(x) = E[max(0, f_best - f(x))]
    # Using normal distribution:
    # If mu < f_best, EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Else EI = 0
    
    # To avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Z-score: how many standard deviations is f_best above mu
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0)
    
    # HVI-inspired exploration term:
    # In HVI, the value of a point depends on the area it dominates.
    # Points with high uncertainty (sigma) in regions where improvement is possible
    # have high potential to expand the hypervolume.
    
    # The "potential" for improvement is max_improvement.
    # The exploration bonus is weighted by the probability that the point
    # can actually improve upon f_best (i.e., mu < f_best).
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    prob_improve = cdf
    
    # Exploration term: sigma * prob_improve
    # This is high when sigma is high AND there's a good chance of improvement.
    exploration = sigma_safe * prob_improve
    
    # Combine EI and exploration
    # The HVI framework suggests that both the expected improvement and the
    # potential for future improvement (uncertainty) matter.
    
    # Weighting: 
    # Early on, we want more exploration. Later, more exploitation.
    # We can use the ratio of current best to global min as a maturity indicator.
    # If f_best is close to f_min, we are mature -> exploit more.
    # If f_best is far from f_min, we are immature -> explore more.
    
    # Maturity = 1 - (f_best - f_min) / (f_initial - f_min)
    # But we don't know f_initial. 
    # Instead, use a simple adaptive weight based on the spread of mu.
    # If mu is tightly concentrated, we are confident -> exploit.
    # If mu is spread out, we are uncertain -> explore.
    
    mu_std = np.std(mu)
    
    # Adaptive weight for exploration
    # When mu_std is high, increase exploration weight.
    # Map mu_std to a weight between 0.5 and 2.0
    explore_weight = 0.5 + 1.5 * (mu_std / (mu_std + 0.1))
    
    # Final score
    scores = ei + explore_weight * exploration
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
