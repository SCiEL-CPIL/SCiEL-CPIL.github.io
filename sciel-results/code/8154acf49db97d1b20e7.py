"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time and ensure we finish within budget
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement and
    Thompson-sampling-inspired diversity, weighted by cost efficiency.
    
    The key structural change: instead of a deterministic EI+UCB blend,
    we use a probabilistic weighting that emphasizes candidates where
    the posterior has high variance relative to the current best,
    creating a more exploratory search in high-dimensional space.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Check deadline
    if _time_left() < 1.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline again
    if _time_left() < 0.5:
        return np.zeros(n, dtype=np.float64)
    
    # Core computation: Use a cost-aware variant of the Probability of Improvement
    # combined with an entropy-based exploration bonus
    
    # Probability of Improvement (PI): P(f(x) < f_best)
    # This is more robust than EI in high dimensions
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Expected Improvement for exploitation
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Exploration bonus: proportional to uncertainty (sigma) but damped
    # by how far we are from the best (to avoid exploring hopeless regions)
    # Use a softplus-like damping
    exploration = sigma * np.exp(-np.maximum(f_best - mu, 0) / (10 * np.std(sigma) + 1e-9))
    
    # Combine: weight PI heavily for exploitation, exploration for discovery
    # The weights are adaptive based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Adaptive blending: more exploration when uncertainty is high
    w_explore = 0.4 * (sigma_std / (sigma_mean + 1e-9))
    w_explore = np.clip(w_explore, 0.1, 0.7)
    
    combined = (1 - w_explore) * ei + w_explore * exploration
    
    # Cost-aware normalization: divide by cost to get improvement per unit cost
    scores = combined / cost
    
    # Rank-based stabilization: add a small rank-based bonus to break ties
    # and encourage diversity in selection
    if _time_left() > 0.3:
        ranks = np.argsort(np.argsort(-scores)).astype(np.float64)
        n_float = float(n)
        # Small diversity bonus: prefer mid-range ranks to avoid extreme clustering
        diversity_bonus = 0.01 * (1.0 - np.abs(ranks / n_float - 0.5) * 2.0)
        scores = scores + diversity_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
