import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    # Calculate pair sums and residuals
    pair_sum = demands[:, None] + demands[None, :]
    residual = capacity - pair_sum

    # Determine which pairs fit in the bin (residual >= 0)
    fits_pair = residual >= 0

    # Count how many items k fit into the residual space for each pair (i, j)
    # residual[i, j] is the space left after placing i then j.
    # We want to know how many other items could potentially fit in that space.
    # This count serves as a proxy for how "open" the bin is.
    # Fewer items fitting => "dead gap" => more attractive.
    
    # Optimization: 
    # We can compute the count by sorting demands?
    # Or just use the broadcasting method as it's fast enough for N=120.
    
    # residual shape: (n, n)
    # demands shape: (n,)
    # We want count[i, j] = sum_k (demands[k] <= residual[i, j])
    
    # Broadcasting:
    # residual_expanded: (n, n, 1)
    # demands_expanded: (1, 1, n)
    # Comparison: (n, n, n)
    
    fits_items = (residual[:, :, None] >= demands[None, None, :])
    
    # If residual is negative, no items fit (since demands > 0).
    # The comparison handles this naturally: negative >= positive is False.
    # So count will be 0 for non-fitting pairs.
    
    count = fits_items.sum(axis=2)
    
    # Heuristic design:
    # We want high values for pairs that leave a "dead gap" (count == 0).
    # We want low values for pairs that leave a large gap (high count).
    # Formula: 1 / (1 + count^2)
    # count=0 -> 1.0
    # count=1 -> 0.5
    # count=2 -> 0.2
    # count=10 -> ~0.009
    
    heuristic = 1.0 / (1.0 + count * count)
    
    # Pairs that don't fit (residual < 0) must have low attractiveness.
    # Although count is 0 for them, which would give 1.0, we must override this.
    heuristic = np.where(fits_pair, heuristic, 1e-9)
    
    # Safety: Ensure diagonal is low to prevent self-placement if not masked by evaluator
    diag_idx = np.diag_indices(n)
    heuristic[diag_idx] = 1e-9
    
    # Final safety checks
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
