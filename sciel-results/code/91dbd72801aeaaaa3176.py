import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO using a Gaussian kernel approach.
    This approach uses a soft exponential decay based on the incremental cost
    of the move (distance from current node to next + distance from next to start).
    This bounded heuristic can provide better exploration balance compared to
    unbounded inverse-distance metrics.
    """
    # prizes shape: (N,)
    # distance_matrix shape: (N, N)
    
    # Extract distances from every node j to node 0
    d_to_start = distance_matrix[:, 0] # Shape (N,)
    
    # Effective cost for moving i -> j
    # This is the extra length added to the path if we visit j next and then return to 0
    # cost[i, j] = d[i, j] + d[j, 0]
    effective_cost = distance_matrix + d_to_start[None, :]
    
    # Gaussian kernel: exp(-cost^2 / (2 * sigma^2))
    # sigma is chosen based on the scale of distances in the unit square.
    # Average distance is ~0.5. MaxLen is 3.
    # A sigma of 0.5 allows costs up to ~1.0 to have significant weight.
    sigma = 0.5
    
    # Calculate the kernel
    # We want to avoid overflow/underflow issues, though cost is small.
    exponent = -(effective_cost ** 2) / (2 * sigma ** 2)
    
    # Clip exponent to prevent underflow to 0 too aggressively if needed, 
    # but np.exp handles small numbers fine.
    kernel = np.exp(exponent)
    
    # Heuristic: Prize * Kernel
    # If prize is 0, heuristic is 0.
    heuristic = prizes[None, :] * kernel
    
    # Ensure non-negative and finite.
    # If heuristic is 0 (due to prize 0 or very high cost), replace with small epsilon.
    # The task says "Heuristic entries that are not positive become 1e-9".
    # So 0 becomes 1e-9.
    
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
