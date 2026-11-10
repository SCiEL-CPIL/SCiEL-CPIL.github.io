"""Rewritten program: DIRECT approach for bin-packing ACO heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    DIRECT approach: Build the heuristic explicitly from task structure.
    For bin packing, the key insight is that we want to pair items that
    together fill a bin as completely as possible without exceeding capacity.
    
    The heuristic H[i,j] represents attractiveness of placing item j right
    after item i in the same bin.
    
    Strategy:
    1. If items don't fit together (d_i + d_j > capacity), set to 1e-9.
    2. If they fit, the attractiveness should reflect how well they fill the bin.
       A pair that fills the bin completely is ideal (waste = 0).
       A pair that leaves lots of space is less ideal.
    
    The most direct formula: for fitting pairs, use a function that peaks
    when d_i + d_j is close to capacity. We use:
      H[i,j] = (d_i + d_j) / (capacity - d_i - d_j + 1)
    This increases with fill and has a gentle penalty for waste.
    
    Alternatively, a simpler and very effective direct approach:
      H[i,j] = 1 / (1 + (capacity - d_i - d_j))
    which is 1 when perfectly filled and decreases with waste.
    
    Let's combine: use the sum/capacity ratio as a base, amplified for tight fits.
    """
    start_time = time.time()
    
    # Deadline guard: must return well before 330s wall-clock limit
    DEADLINE = 320.0  # 10s safety margin
    
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
    
    # Vectorized computation
    # sums[i,j] = d_i + d_j
    sums = d[:, None] + d[None, :]
    
    # Check if pair fits in a bin
    fits = sums <= capacity
    
    # For fitting pairs, compute waste = capacity - sum
    # waste is in [0, capacity] for fitting pairs
    waste = capacity - sums  # >= 0 for fitting pairs
    
    # New Heuristic Strategy: Exponential Decay of Waste
    # H[i,j] = exp(-waste / capacity)
    # This bounds the heuristic values between 0 and 1 for fitting pairs.
    # Perfect fit (waste=0) -> H=1.
    # Max waste (waste=capacity, i.e., sum=0 which is impossible for d>=20, but theoretically) -> H=e^-1.
    # For d_i+d_j=20+20=40, waste=110. H=exp(-110/150) ~ 0.48.
    # This keeps the heuristic scale consistent and lets pheromones play a larger role,
    # potentially allowing the ACO to learn better global patterns rather than being
    # dominated by the immediate pair fill ratio.
    
    raw_scores = np.exp(-waste / capacity)
    
    # For non-fitting pairs, set to 1e-9
    H = np.where(fits, raw_scores, 1e-9)
    
    # Ensure no non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass  # Return immediately; computation is fast anyway
    
    return H
# EVOLVE-BLOCK-END
