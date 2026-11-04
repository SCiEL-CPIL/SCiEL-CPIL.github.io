import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Helper to calculate cost
    def calc_cost(p):
        # Sum a[i,j] * b[p[i], p[j]]
        # Vectorized: b[p[:, None], p[None, :]]
        b_perm = b[p[:, None], p[None, :]]
        return np.sum(a * b_perm)
    
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute diagonal of b (usually 0)
    b_diag = np.diag(b)
    
    # Local Search
    while time.time() < deadline:
        improved = False
        # Update C_curr (current contribution of each facility)
        # C_curr[j] = sum_{k != j} a[j,k] b[p[j], p[k]] + a[k,j] b[p[k], p[j]]
        b_perm = b[p[:, None], p[None, :]]
        prod = a * b_perm
        row_sums = np.sum(prod, axis=1)
        col_sums = np.sum(prod, axis=0)
        c_curr = row_sums + col_sums - np.diag(prod)
        
        for i in range(n):
            if time.time() >= deadline:
                break
            
            L_i = p[i]
            
            # 1. Compute C_i_L for all L
            # C_i_L = sum_{k != i} a[i,k] b[L, p[k]] + a[k,i] b[p[k], L]
            a_i_row = a[i, :].copy()
            a_i_row[i] = 0
            a_i_col = a[:, i].copy()
            a_i_col[i] = 0
            
            # V1[L] = sum_k a_i_row[k] b[L, p[k]]
            # b[:, p] has shape (n, n). Row L is b[L, p[0]], ...
            V1 = b[:, p] @ a_i_row
            
            # V2[L] = sum_k a_i_col[k] b[p[k], L]
            # b.T[:, p] has shape (n, n). Row L is b[p[0], L], ...
            V2 = b.T[:, p] @ a_i_col
            
            C_i_L = V1 + V2
            C_i_Li = C_i_L[L_i]
            
            # 2. Compute C_j_Li for all j
            # L_new = L_i
            # C_j_Li = sum_{k != j} a[j,k] b[L_i, p[k]] + a[k,j] b[p[k], L_i]
            
            # M1 = b[L_i, p] (vector of size n)
            M1 = b[L_i, p]
            # T1[j] = sum_k a[j,k] M1[k]
            T1 = a @ M1
            # Subtract term where k=j: a[j,j] M1[j]
            T1 -= a[np.arange(n), np.arange(n)] * M1
            
            # M2 = b[p, L_i] (vector of size n)
            M2 = b[p, L_i]
            # T2[j] = sum_k a[k,j] M2[k]
            T2 = a.T @ M2
            # Subtract term where k=j: a[j,j] M2[j]
            T2 -= a[np.arange(n), np.arange(n)] * M2
            
            C_j_Li = T1 + T2
            
            # 3. Compute Delta for all j != i
            # Delta[j] = (C_i_Li_j - C_i_Li) + (C_j_Li_j - C_curr[j]) - Correction + AddBack
            
            C_i_pj = C_i_L[p]  # Vector of C_i_{p[j]} for all j
            C_j_pj = c_curr     # Vector of C_j_{p[j]} for all j
            
            # Correction terms
            # Term 1: (a[i,j] + a[j,i]) * b[p[j], p[j]]
            a_ij = a[i, :]
            a_ji = a[:, i]
            b_pj_pj = b_diag[p]  # b[p[j], p[j]]
            corr1 = (a_ij + a_ji) * b_pj_pj
            
            # Term 2: (a[i,j] + a[j,i]) * b[p[i], p[i]]
            b_Li_Li = b[L_i, L_i]
            corr2 = (a_ij + a_ji) * b_Li_Li
            
            # Add Back: a[i,j] b[p[j], p[i]] + a[j,i] b[p[i], p[j]]
            # b[p[j], p[i]] is column L_i of b indexed by p
            b_pj_Li = b[p, L_i]
            b_Li_pj = b[L_i, p]
            add_back = a_ij * b_pj_Li + a_ji * b_Li_pj
            
            delta = (C_i_pj - C_i_Li) + (C_j_Li - C_j_pj) - corr1 - corr2 + add_back
            
            # Exclude i itself
            delta[i] = np.inf
            
            # Find best j
            j_best = np.argmin(delta)
            
            if delta[j_best] < -1e-9:
                # Apply swap
                p[i], p[j_best] = p[j_best], p[i]
                current_cost += delta[j_best]
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                
                improved = True
                # We could break the i-loop and restart, but continuing is often fine
                # Updating c_curr is skipped here for simplicity, will be recomputed next outer loop
                # Or we can update c_curr incrementally? 
                # For n=256, recomputing c_curr at start of next i-iteration is not done, 
                # it's done at start of while-loop. 
                # Wait, c_curr is computed once per while-pass. 
                # If we swap, c_curr is stale for subsequent i's in this pass.
                # This might lead to suboptimal swaps or incorrect delta calculations.
                # To be safe and simple, we should break the inner loop after a swap 
                # and restart the while loop (or just the inner loop) to recompute c_curr.
                # However, recomputing c_curr is O(n^2). 
                # If we break, we do O(n^2) + O(n^2) per swap.
                # If we don't break, we use stale c_curr. 
                # Given the high accuracy of NumPy, using stale c_curr might be risky.
                # Let's break the inner loop and restart the outer while loop logic?
                # No, just break the for-i loop and let the while loop restart?
                # That would be very slow if many swaps.
                # Better: Update c_curr incrementally?
                # c_curr[i] and c_curr[j] change.
                # c_curr[k] for k != i, j changes due to interaction with i and j.
                # c_curr[k] += (a[k,i] b[p[k], p[j]] - a[k,i] b[p[k], p[i]]) + ...
                # This is O(n) per swap.
                # Let's implement incremental update of c_curr.
                
                # Update c_curr
                # Swap i and j_best
                # Let old_i_loc = p[j_best] (before swap, but we swapped p)
                # We swapped p[i] and p[j_best].
                # Let L_i_new = p[i] (which was p[j_best])
                # Let L_j_new = p[j_best] (which was p[i])
                
                # Actually, we already swapped p.
                # L_i = p[i] (new)
                # L_j = p[j_best] (new)
                
                # Update c_curr[i]:
                # It should be C_i_{L_i}. We have C_i_L vector.
                # c_curr[i] = C_i_L[L_i]
                
                # Update c_curr[j_best]:
                # It should be C_j_{L_j}. We have C_j_Li vector? No, that was for L_i.
                # We need C_j_{L_j}.
                # This is not directly available.
                # However, we can recompute c_curr for i and j_best?
                # Or just recompute all c_curr?
                # Given n=256, recomputing all c_curr is O(n^2) = 65k ops.
                # This is cheap.
                # Let's just recompute c_curr immediately after swap.
                
                b_perm = b[p[:, None], p[None, :]]
                prod = a * b_perm
                row_sums = np.sum(prod, axis=1)
                col_sums = np.sum(prod, axis=0)
                c_curr = row_sums + col_sums - np.diag(prod)
                
                # Continue to next i? Or restart?
                # Continuing is fine as c_curr is updated.
        
        if not improved:
            break
            
    return best_p.tolist()
