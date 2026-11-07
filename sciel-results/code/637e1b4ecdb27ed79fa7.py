# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = None
_deadline = None

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Monte Carlo Thompson Sampling.
    
    Strategy:
    1. Draw M samples from the posterior distribution N(mu, sigma^2) for each candidate.
    2. For each sample, calculate the improvement over f_best: max(0, f_best - y).
    3. Divide the improvement by the cost of the candidate to get a cost-adjusted improvement score.
    4. Average these scores over the M samples to get the final acquisition score.
    
    This approach naturally balances exploration (via the variance in the samples) 
    and exploitation (via the mean), and explicitly accounts for the evaluation cost.
    """
    global _start_time, _deadline
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # Safety margin before 330s limit
    
    # Emergency check
    if time.time() > _deadline:
        # Fallback to simple deterministic EI / cost if time is up
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        cost = np.asarray(cost, dtype=float)
        eps = 1e-12
        s = np.maximum(sigma, eps)
        z = (f_best - mu) / s
        ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1.0)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(mu)
    
    if n == 0:
        return np.array([])
    
    # Ensure cost is at least 1 to avoid division by zero or very small numbers
    # Problem states cost = 1 + mean(x), so cost >= 1.
    cost_safe = np.maximum(cost, 1.0)
    
    # Handle sigma
    eps = 1e-12
    s = np.maximum(sigma, eps)
    
    # Number of Monte Carlo samples
    # 50 samples provide a good balance between accuracy and speed for 2000 candidates
    M = 50
    
    # Generate samples: shape (M, n)
    # Y[m, i] ~ N(mu[i], s[i]^2)
    # We use standard normal noise scaled by s
    noise = np.random.standard_normal((M, n))
    Y = mu[None, :] + s[None, :] * noise
    
    # Calculate improvement for each sample
    # Imp[m, i] = max(0, f_best - Y[m, i])
    Imp = np.maximum(0.0, f_best - Y)
    
    # Calculate cost-adjusted score for each sample
    # Score[m, i] = Imp[m, i] / cost[i]
    Scores = Imp / cost_safe[None, :]
    
    # Average over the M samples to get the expected cost-adjusted improvement
    final_score = np.mean(Scores, axis=0)
    
    # Ensure finite values
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_score
# EVOLVE-BLOCK-END
