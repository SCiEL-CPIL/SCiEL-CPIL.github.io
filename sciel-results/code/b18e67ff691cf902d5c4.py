"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right
    after item i in the same bin. Larger is more attractive.
    
    This heuristic implements a 'Best Fit' strategy:
    It favors placing item j into a bin (represented by the last item i) 
    if it results in a tight fit (minimal remaining space).
    H[i, j] is proportional to 1 / (remaining_space_after_placement).
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Calculate the sum of sizes for all pairs (i, j)
    # pair_sum[i, j] = demands[i] + demands[j]
    pair_sum = demands[:, None] + demands[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sum <= capacity
    
    # Calculate remaining space after placing j after i
    # remaining = capacity - (d_i + d_j)
    # We only care about this where fits is True
    remaining = capacity - pair_sum
    
    # To avoid division by zero and to penalize very loose fits less aggressively 
    # than tight ones, we add a small epsilon.
    # A smaller remaining space -> larger 1/remaining -> higher attractiveness.
    epsilon = 1e-6
    safe_remaining = remaining + epsilon
    
    # Compute the heuristic: inverse of remaining space
    # For non-fitting pairs, we will set to a very small value later.
    # For fitting pairs, H = 1 / (remaining + eps)
    # Note: We want larger values for tighter fits.
    # If remaining is 0, H is 1/eps (very large).
    # If remaining is 50, H is 1/50 (smaller).
    
    heuristic = np.zeros((n, n), dtype=np.float64)
    
    # Apply the formula only to fitting pairs
    # Using np.where is efficient
    # We use 1e-9 for non-fitting pairs as required by the task description 
    # ("Heuristic entries that are not positive become 1e-9")
    
    # Calculate values for fitting pairs
    vals = 1.0 / safe_remaining
    
    # Set non-fitting pairs to 1e-9
    # Also handle diagonal (i==j) if an item doesn't fit itself? 
    # d_i + d_i <= capacity? If not, it's non-fitting.
    # The logic holds: if 2*d_i > capacity, it won't fit, so H=1e-9.
    
    heuristic = np.where(fits, vals, 1e-9)
    
    # Ensure all values are positive and finite
    # The calculation 1/(x+eps) for x>=0 is always finite and positive.
    # The 1e-9 is positive and finite.
    
    return heuristic
# EVOLVE-BLOCK-END
