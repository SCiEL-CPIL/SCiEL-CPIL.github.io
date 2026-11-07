import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]

    # Calculate remaining space if j is placed after i
    # sum_ij = demands[i] + demands[j]
    # gap = capacity - sum_ij
    # We want to minimize gap (maximize sum_ij)
    
    sums = demands[:, None] + demands[None, :]
    fits = sums <= capacity
    
    # Gap is capacity - sums
    # For fitting pairs, gap >= 0
    # We want high attractiveness for small gap.
    # H = 1 / (gap + 1)
    
    gap = capacity - sums
    # For non-fitting pairs, gap < 0. We will mask them.
    
    attractiveness = np.zeros((n, n))
    # Only compute for fitting pairs
    # gap + 1 avoids division by zero if gap is 0 (perfect fit)
    # If gap is 0, H = 1. If gap is 100, H = 1/101.
    
    # To avoid warnings/errors, compute safely
    # We can use np.where
    
    # Initialize with 1e-9
    attractiveness = np.full((n, n), 1e-9)
    
    # Calculate for valid pairs
    # gap is valid where fits is True
    # H = 1 / (gap + 1)
    
    # Note: gap can be 0. 1/(0+1) = 1.
    # Max gap is capacity - min(demands) - min(demands) approx 110.
    # Min H approx 1/111.
    
    # Let's scale it? 
    # The magnitude of H matters relative to Pheromone.
    # Pheromone starts at 1.
    # If H is around 1, probability is proportional to P.
    # If H is around 0.01, probability is dominated by P.
    
    # Let's try H = 1 / (gap + 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        h_val = 1.0 / (gap + 1.0)
        
    # Mask non-fitting
    attractiveness = np.where(fits, h_val, 1e-9)
    
    # Ensure positive and finite
    attractiveness = np.where(np.isfinite(attractiveness) & (attractiveness > 0), attractiveness, 1e-9)
    
    return attractiveness
