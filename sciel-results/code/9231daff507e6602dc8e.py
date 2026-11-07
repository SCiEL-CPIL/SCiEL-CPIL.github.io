import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return -np.asarray(mu, dtype=np.float64) / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # f_best is a scalar
    if isinstance(f_best, np.ndarray):
        f_best_val = float(np.min(f_best))
    else:
        f_best_val = float(f_best)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best_val - mu) / sigma
    pi = norm.cdf(z)
    
    # Cost-aware PI
    scores = pi / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
