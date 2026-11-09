import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Explicit deadline guard
    start_time = time.time()
    # 300s limit, keep 10s margin
    deadline = start_time + 290.0
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential non-finite values in inputs
    # If mu or sigma is nan/inf, the result should be invalid (low score)
    # We'll handle this in the score sanitization step.
    
    # Thompson Sampling: Sample a single function from the posterior
    # The prompt specifies: f_sample(x) = mu(x) - sigma(x) * z, where z ~ N(0, 1)
    # Using a scalar z creates a single consistent function sample across all points
    z = np.random.normal(0.0, 1.0)
    
    # Compute sampled function values
    # Note: We want to MINIMIZE the function value.
    # The prompt says: "select the candidate that minimizes this sampled value divided by its cost"
    # So we want to find x that minimizes J(x) = (mu(x) - sigma(x)*z) / cost(x)
    
    # Calculate the ratio to be minimized
    # Cost is >= 1, so division is safe from zero
    # We use np.errstate to ignore warnings for inf/nan in intermediate steps
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (mu - sigma * z) / cost
        
    # The function must return scores where HIGHER = evaluate next
    # So we return the negative of the ratio to be minimized
    scores = -ratio
    
    # Sanitize scores
    # Replace non-finite values (nan, inf) with a very low score so they are not selected
    # Since we maximize the score, -1e10 is effectively "never pick"
    scores[~np.isfinite(scores)] = -1e10
    
    # Check deadline (computation is O(N), so this is mostly for compliance)
    if time.time() > deadline:
        # If we somehow exceeded (unlikely), we just return what we have
        pass
        
    return scores
