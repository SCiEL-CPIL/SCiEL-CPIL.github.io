"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB), with aggressive cost normalization.
    
    The strategy:
    1. Compute EI for minimization: E[max(f_best - f, 0)]
    2. Compute UCB for minimization: mu - kappa * sigma (we want low values)
    3. Combine them with a balance parameter
    4. Divide by cost^gamma to strongly favor cheap evaluations
    
    Args:
        mu: Array of shape (N,) with mean predictions.
        sigma: Array of shape (N,) with standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with scores (higher is better).
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE_SECONDS:
        # Return a simple, fast score to avoid timeout
        return np.zeros_like(mu)
    
    N = len(mu)
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # --- Expected Improvement for minimization ---
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    delta = f_best - mu  # positive when mu < f_best (improvement possible)
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    EI = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # --- UCB for minimization ---
    # Lower is better, so UCB_exploit = mu - kappa * sigma
    # We want to maximize the score, so we use -mu + kappa*sigma
    # But let's think in terms of "regret reduction"
    # Actually, for minimization, the optimistic value is mu - kappa*sigma
    # We want to maximize (f_best - (mu - kappa*sigma)) = (f_best - mu) + kappa*sigma
    # But that's not quite right either. Let's use:
    # UCB_score = -mu + kappa * sigma (higher is better, encourages exploration)
    kappa = 2.5
    UCB_score = -mu + kappa * sigma_safe
    
    # --- Hybrid: combine EI and UCB ---
    # Normalize both to comparable scales
    # EI is in objective units, UCB_score is also in objective units (roughly)
    # Use a weighted combination
    alpha = 0.5  # weight for EI
    hybrid = alpha * EI + (1 - alpha) * UCB_score
    
    # --- Cost normalization ---
    # cost is in [1, 2]
    # Use cost^gamma in the denominator to strongly favor cheap points
    gamma = 3.0
    cost_penalty = np.power(cost, gamma)
    
    # Final score: higher is better
    score = hybrid / cost_penalty
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
