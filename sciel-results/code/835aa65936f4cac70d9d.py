import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Ensure int64 for accurate cost calculation if needed, though float64 is usually fine for intermediate, 
    # but final cost should be exact integer. The problem asks for exact integer cost.
    # Using int64 for a and b is specified.
    
    # 1. Initialization: Rank Matching
    # Heavy flows to short distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute B_perm[i, j] = b[p[i], p[j]]
    b_perm = b[p[:, None], p[None, :]]
    
    # Calculate initial cost
    # cost = sum(a * b_perm)
    cost = np.sum(a * b_perm)
    best_cost = cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # Helper to update b_perm after swapping p[i] and p[j]
    def swap_update(i, j):
        nonlocal b_perm, p
        # Swap in p
        p[i], p[j] = p[j], p[i]
        # Update b_perm
        # b_perm[i, :] needs to be b[p[i], p[k]] for all k
        # b_perm[j, :] needs to be b[p[j], p[k]] for all k
        # b_perm[:, i] needs to be b[p[k], p[i]] for all k
        # b_perm[:, j] needs to be b[p[k], p[j]] for all k
        
        # Since b is symmetric, b_perm is symmetric.
        # We only need to update row i, row j, col i, col j.
        
        # Get new locations
        loc_i = p[i]
        loc_j = p[j]
        
        # Update row i
        b_perm[i, :] = b[loc_i, p]
        # Update row j
        b_perm[j, :] = b[loc_j, p]
        
        # Since symmetric, update col i and j (which are the same as rows if we assume symmetry of b)
        # But b_perm might not be symmetric if b is not? The problem says "distance matrix", usually symmetric.
        # QAPLIB instances are typically symmetric.
        # To be safe, update cols explicitly if b is not symmetric.
        # Assuming symmetry for speed:
        b_perm[:, i] = b[p, loc_i]
        b_perm[:, j] = b[p, loc_j]
        
        # The diagonal elements b_perm[i, i] and b_perm[j, j] are b[loc_i, loc_i] and b[loc_j, loc_j].
        # The cross terms b_perm[i, j] and b_perm[j, i] are handled by the row/col updates.
        
    # Steepest Descent Local Search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            # Compute delta for swapping i with all j
            # Delta(i, j) = sum_k (a[k, i] - a[k, j]) * (b_perm[k, j] - b_perm[k, i]) - corrections
            
            # Vectorized calculation for all j
            # a_col = a[:, i]
            # a_diffs = a[:, i][None, :] - a  # Shape (1, n) - (n, n) -> (n, n). a_diffs[k, j] = a[k, i] - a[k, j]
            # b_diffs = b_perm - b_perm[:, i][None, :] # Shape (n, n) - (n, 1) -> (n, n). b_diffs[k, j] = b_perm[k, j] - b_perm[k, i]
            
            # This creates large temporary arrays. 
            # Alternative: 
            # D = (a[:, i] - a.T) * (b_perm.T - b_perm[:, i])
            # Wait, a[:, i] is (n,). a.T is (n, n). a[:, i][None, :] - a.T gives (n, n) where element [j, k] is a[k, i] - a[k, j]? No.
            # Let's stick to the formula: Delta(i, j) = sum_k (a[k, i] - a[k, j]) * (b_perm[k, j] - b_perm[k, i])
            
            # Let's compute the full dot product matrix for this i
            # term1 = a[:, i] - a.T  # Shape (n, n). term1[j, k] = a[k, i] - a[k, j]? 
            # a[:, i] is column i of a.
            # a.T is a.
            # a[:, i][None, :] - a.T -> Row j, Col k is a[k, i] - a[j, k]? No.
            # a[:, i] is (n, 1) effectively.
            # Let's use:
            # a_diff = a[:, i][None, :] - a  # (1, n) - (n, n) -> (n, n). a_diff[k, j] = a[k, i] - a[k, j]. Correct.
            # b_diff = b_perm - b_perm[:, i][:, None] # (n, n) - (n, 1) -> (n, n). b_diff[k, j] = b_perm[k, j] - b_perm[k, i]. Correct.
            
            a_diff = a[:, i][None, :] - a
            b_diff = b_perm - b_perm[:, i][:, None]
            
            deltas = np.sum(a_diff * b_diff, axis=0) # Shape (n,). deltas[j] is the raw sum including k=i, k=j.
            
            # Subtract corrections for k=i and k=j
            # Correction for k=i: (a[i, i] - a[i, j]) * (b_perm[i, j] - b_perm[i, i])
            # Correction for k=j: (a[j, i] - a[j, j]) * (b_perm[j, j] - b_perm[j, i])
            
            # We need to do this for all j.
            # corr_i[j] = (a[i, i] - a[i, j]) * (b_perm[i, j] - b_perm[i, i])
            corr_i = (a[i, i] - a[i, :]) * (b_perm[i, :] - b_perm[i, i])
            
            # corr_j[j] = (a[j, i] - a[j, j]) * (b_perm[j, j] - b_perm[j, i])
            # a[j, i] is a.T[i, j] or a[:, i][j]
            # a[j, j] is a[j, j]
            # b_perm[j, j] is b_perm[j, j]
            # b_perm[j, i] is b_perm[:, i][j]
            
            a_ji = a[:, i]
            a_jj = np.diag(a)
            b_perm_jj = np.diag(b_perm)
            b_perm_ji = b_perm[:, i]
            
            corr_j = (a_ji - a_jj) * (b_perm_jj - b_perm_ji)
            
            deltas -= (corr_i + corr_j)
            
            # Mask i itself
            deltas[i] = np.inf
            
            # Find best j
            j_best = np.argmin(deltas)
            delta_best = deltas[j_best]
            
            if delta_best < 0:
                # Perform swap
                swap_update(i, j_best)
                cost += delta_best
                improved = True
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
                    
                # If time is tight, break early? No, just let the while loop check.
                if time.time() - start_time > time_limit:
                    break
        
        if time.time() - start_time > time_limit:
            break

    # If time remains, do Simulated Annealing
    if time.time() - start_time < time_limit - 10:
        T = 1000.0 # Initial temperature
        T_min = 1.0
        cooling = 0.999
        current_cost = cost
        current_p = p.copy()
        
        # Reinitialize b_perm for current_p if it changed? 
        # swap_update keeps b_perm in sync with p. So b_perm is current.
        
        while time.time() - start_time < time_limit - 5:
            # Random move
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            # Calculate delta
            # Same logic as above, but for single pair
            # Delta(i, j) = sum_k (a[k, i] - a[k, j]) * (b_perm[k, j] - b_perm[k, i]) - corr_i - corr_j
            
            # Vectorized for single pair is just dot product
            a_diff = a[:, i] - a[:, j]
            b_diff = b_perm[:, j] - b_perm[:, i]
            raw_delta = np.dot(a_diff, b_diff)
            
            corr_i = (a[i, i] - a[i, j]) * (b_perm[i, j] - b_perm[i, i])
            corr_j = (a[j, i] - a[j, j]) * (b_perm[j, j] - b_perm[j, i])
            delta = raw_delta - corr_i - corr_j
            
            # Acceptance
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                swap_update(i, j)
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
                
    return best_p.tolist()
