import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    # If we are close to the deadline, return a safe default (e.g., UCB with kappa=0, i.e., pure exploitation)
    # Or just return the current best guess quickly.
    # However, the function is called in a loop. The "330s wall-clock" is for the whole test.
    # The "300s" limit per evaluation in the text might be a red herring or for the whole loop.
    # "Each evaluation must finish within 300 s." -> This likely means the *entire* acquisition function call (including GP inference? No, GP is done outside).
    # The function `acquisition` is just a math op on arrays. It should take milliseconds.
    # So the deadline guard is mostly for safety if something hangs.
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety margin: return exploitation only (low mu)
        # score = -mu / cost
        cost_safe = cost + 1e-9
        return -mu / cost_safe

    # Normal path: Cost-Aware UCB
    # kappa controls exploration vs exploitation
    # Higher kappa -> more exploration
    kappa = 1.0
    
    cost_safe = cost + 1e-9
    score = (-mu + kappa * sigma) / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
