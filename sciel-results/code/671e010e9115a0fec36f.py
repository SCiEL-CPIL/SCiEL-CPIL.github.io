# EVOLVE-BLOCK-START
import numpy as np


def _entropy_from_counts(counts, total):
    """
    Compute Shannon entropy in bits from counts and total probability mass.
    counts: array of integers (counts of items in each bin)
    total: total number of items (sum of counts)
    """
    if total <= 0:
        return 0.0
    probs = counts / total
    # Ignore zero probabilities
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Uses greedy submodular maximization to find a sparse support set S
    that maximizes D/M, where D is entropy of x-y and M is max entropy of x, x+y, x+2y.
    """
    n = 10
    total_cells = n * n
    
    # Precompute bin indices for each cell (y, x)
    # D: x - y. Range: -9 to 9. Offset by 9 -> 0 to 18. Size 19.
    # H0: x. Range: 0 to 9. Size 10.
    # H1: x + y. Range: 0 to 18. Size 19.
    # H2: x + 2y. Range: 0 to 27. Size 28.
    
    idx_D = np.zeros((n, n), dtype=np.int32)
    idx_H0 = np.zeros((n, n), dtype=np.int32)
    idx_H1 = np.zeros((n, n), dtype=np.int32)
    idx_H2 = np.zeros((n, n), dtype=np.int32)
    
    for y in range(n):
        for x in range(n):
            idx_D[y, x] = (x - y) + (n - 1)
            idx_H0[y, x] = x
            idx_H1[y, x] = x + y
            idx_H2[y, x] = x + 2 * y
            
    size_D = 2 * n - 1
    size_H0 = n
    size_H1 = 2 * n - 1
    size_H2 = 3 * n - 1
    
    # Greedy selection
    # We will build the support set S one cell at a time.
    # We start with an empty set.
    selected = np.zeros((n, n), dtype=bool)
    
    # Current counts in each bin
    counts_D = np.zeros(size_D, dtype=np.int32)
    counts_H0 = np.zeros(size_H0, dtype=np.int32)
    counts_H1 = np.zeros(size_H1, dtype=np.int32)
    counts_H2 = np.zeros(size_H2, dtype=np.int32)
    
    current_k = 0
    best_ratio = 0.0
    best_support = np.zeros((n, n), dtype=np.float64)
    
    # We iterate until we have added all cells or no improvement is found.
    # In practice, adding more cells usually decreases the ratio if the optimal is sparse,
    # but we don't know the optimal size. We will track the best ratio seen so far.
    
    for step in range(total_cells):
        best_gain = -np.inf
        best_cell = None
        best_new_ratio = -np.inf
        
        # Try adding each unselected cell
        for y in range(n):
            for x in range(n):
                if selected[y, x]:
                    continue
                
                # Get bin indices for this cell
                iD = idx_D[y, x]
                iH0 = idx_H0[y, x]
                iH1 = idx_H1[y, x]
                iH2 = idx_H2[y, x]
                
                # Tentatively add
                counts_D[iD] += 1
                counts_H0[iH0] += 1
                counts_H1[iH1] += 1
                counts_H2[iH2] += 1
                new_k = current_k + 1
                
                # Compute new entropies
                D_new = _entropy_from_counts(counts_D, new_k)
                H0_new = _entropy_from_counts(counts_H0, new_k)
                H1_new = _entropy_from_counts(counts_H1, new_k)
                H2_new = _entropy_from_counts(counts_H2, new_k)
                
                M_new = max(H0_new, H1_new, H2_new)
                
                if M_new < 1e-9:
                    # Invalid or degenerate, avoid division by zero
                    # Though with k>=1, M should be > 0 unless all mass is in one bin
                    # If M is 0, ratio is undefined or infinite? 
                    # Task says M <= 1e-9 is invalid. 
                    # If we hit this, we should probably not select this cell if it leads to invalid state.
                    # However, if k=1, H is 0. D is 0. 0/0.
                    # Let's skip if M is too small.
                    ratio_new = 0.0
                else:
                    ratio_new = D_new / M_new
                
                # Revert tentative addition
                counts_D[iD] -= 1
                counts_H0[iH0] -= 1
                counts_H1[iH1] -= 1
                counts_H2[iH2] -= 1
                
                if ratio_new > best_gain:
                    best_gain = ratio_new
                    best_cell = (y, x)
                    best_new_ratio = ratio_new
        
        if best_cell is None:
            break
        
        # If the best gain is not better than our current best ratio, and we have a valid best_support,
        # we might want to stop? 
        # The greedy method is monotonic for submodular functions, but D/M is not necessarily submodular.
        # However, usually, the ratio peaks at some intermediate support size.
        # We should check if best_new_ratio > best_ratio.
        # If yes, update best_support.
        # If no, we might still continue if we haven't reached a certain size, 
        # but typically if the ratio drops, it stays low.
        # Let's just track the best ratio found so far.
        
        if best_new_ratio > best_ratio:
            best_ratio = best_new_ratio
            # Record the support set up to this point
            # We need to update the selected mask first
            selected[best_cell[0], best_cell[1]] = True
            # Update counts
            iD = idx_D[best_cell[0], best_cell[1]]
            iH0 = idx_H0[best_cell[0], best_cell[1]]
            iH1 = idx_H1[best_cell[0], best_cell[1]]
            iH2 = idx_H2[best_cell[0], best_cell[1]]
            counts_D[iD] += 1
            counts_H0[iH0] += 1
            counts_H1[iH1] += 1
            counts_H2[iH2] += 1
            current_k += 1
            
            # Create the distribution P
            # P[y, x] = 1/k for selected cells, 0 otherwise
            P_temp = np.zeros((n, n), dtype=np.float64)
            P_temp[selected] = 1.0 / current_k
            best_support = P_temp.copy()
            
        else:
            # Even if it's not the best, we must add it to continue the greedy process?
            # If we don't add it, we stop.
            # If we add it, we might find a better ratio later? Unlikely for ratio.
            # But let's add it to keep the state consistent, just in case.
            # Actually, if the ratio decreases, adding more cells usually keeps it low or decreases it further.
            # To save time, we could break if the ratio drops significantly.
            # But let's just add it and continue.
            selected[best_cell[0], best_cell[1]] = True
            iD = idx_D[best_cell[0], best_cell[1]]
            iH0 = idx_H0[best_cell[0], best_cell[1]]
            iH1 = idx_H1[best_cell[0], best_cell[1]]
            iH2 = idx_H2[best_cell[0], best_cell[1]]
            counts_D[iD] += 1
            counts_H0[iH0] += 1
            counts_H1[iH1] += 1
            counts_H2[iH2] += 1
            current_k += 1
            
            # Optional: Early stopping if ratio is very low
            if best_ratio > 0 and best_new_ratio < best_ratio * 0.5 and current_k > 10:
                 # Heuristic: if it drops by half, stop
                 pass # We continue for now to be safe
        
    # If no valid support was found (e.g. all ratios were 0), fallback to uniform
    if best_ratio < 1e-9:
        best_support = np.ones((n, n), dtype=np.float64) / total_cells
    else:
        # Normalize just in case
        total_sum = np.sum(best_support)
        if total_sum > 1e-9:
            best_support = best_support / total_sum
        else:
            best_support = np.ones((n, n), dtype=np.float64) / total_cells

    # Clip negatives (shouldn't happen) and ensure shape
    best_support = np.clip(best_support, 0, None)
    
    return best_support


# EVOLVE-BLOCK-END
