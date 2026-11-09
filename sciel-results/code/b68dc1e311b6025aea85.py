import numpy as np
import time
import scipy.special as sp

_start_time = time.time()
TIME_LIMIT = 300.0

_acq_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) with adaptive exploration.
    
    The score is calculated as -(mu - kappa * sigma), encouraging the selection of
    points with low predicted mean and high uncertainty. The exploration parameter
    kappa decays over time, shifting the strategy from exploration to exploitation.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _acq_call_count
    _acq_call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to exploitation: return negative mean (lower mean = higher score)
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Total evaluations is 30. We start with 5 initial points.
    # So we have 25 acquisition steps.
    # _acq_call_count starts at 1 for the first acquisition call (after 5 initial points).
    # So step index t = _acq_call_count - 1.
    # t ranges from 0 to 24.
    total_steps = 25
    t = _acq_call_count - 1
    
    # Decay kappa from 2.5 to 0.1
    # Linear decay
    kappa_max = 2.5
    kappa_min = 0.1
    if total_steps > 1:
        kappa = kappa_max + (kappa_min - kappa_max) * (t / (total_steps - 1))
    else:
        kappa = kappa_min
        
    # LCB score: we want to minimize (mu - kappa * sigma)
    # So we return the negative of that to maximize
    lcb_value = mu - kappa * sigma
    score = -lcb_value
    
    # Ensure finite scores
    score[~np.isfinite(score)] = 0.0
    
    return score
