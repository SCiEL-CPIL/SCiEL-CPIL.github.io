import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time per call.
# The acquisition function is called many times. We need to ensure we don't
# exceed the total budget. However, the task says "Each evaluation must finish within 300 s".
# The acquisition function itself is called 2000 times per loop iteration? No, "the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one".
# So acquisition is called once with arrays of size 2000.
# The total time limit is 330s.
# We should check if we are running out of time. If we are, we should return a valid score quickly.
# Since we don't have access to the global start time easily without a global variable,
# and the function is called repeatedly, we can use a global start time.

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: if we are close to the 300s limit (per evaluation constraint) or 330s total,
    # we should return a simple, fast result.
    # However, the constraint is "Each evaluation must finish within 300 s".
    # This likely refers to the whole run or each step? "Each evaluation" usually means the function call.
    # But the loop runs while total cost < 30.
    # Let's assume the 300s is a per-call timeout or a total run timeout.
    # Given "evaluator KILLS the program after 330s wall-clock", it's a total run time.
    # If we are close to 330s, we should stop doing heavy work.
    # But acquisition is just numpy ops on 2000 points. It's very fast.
    # The heavy part is the GP fitting, which is done by the evaluator, not us.
    # So our function should be fast.
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    # Numerically stable EI calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = delta * cdf_z + sigma * pdf_z
    
    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Add a small exploration bonus to encourage visiting high-uncertainty regions
    # especially when EI is low, to help escape local minima or cover the space.
    # The term (sigma / cost) rewards high uncertainty per unit cost.
    # We scale it down to not dominate the EI term.
    score += 1e-6 * (sigma / cost)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
