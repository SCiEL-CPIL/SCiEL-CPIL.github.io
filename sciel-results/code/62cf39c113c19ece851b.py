"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(p, a, b):
    """Compute the QAP cost for permutation p: sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.dot(a.ravel(), bp.ravel()))


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit

    # --- 1. Hungarian initialization (linear surrogate) ---
    # The linear surrogate: assign facility i to location p[i] to minimize
    # sum_i (sum_j a[i,j]) * b[i, p[i]] + (sum_j a[j,i]) * b[p[i], i]
    # Actually the standard linearization: cost(i, j) = (row_sum_a[i] + col_sum_a[i]) * b[i, j]
    # Wait, the standard assignment relaxation of QAP:
    # minimize sum_{i,j} c[i,j] x[i,j] where c[i,j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    # Since b is symmetric (typically), c[i,j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[j,k]
    # Let's compute the cost matrix properly.
    # c[i, j] = sum_k a[i,k] * b[j,k] + sum_k a[k,i] * b[k,j]
    # For symmetric b: b[j,k] = b[k,j], so c[i,j] = sum_k (a[i,k] + a[k,i]) * b[k,j]
    
    a_sym = a + a.T  # a[i,k] + a[k,i]
    c = np.zeros((n, n), dtype=np.int64)
    # c[i, j] = sum_k a_sym[i, k] * b[k, j]
    # This is a matrix multiplication: c = a_sym @ b
    c = a_sym.astype(np.float64) @ b.astype(np.float64)
    
    # Hungarian algorithm to find optimal assignment for the linear surrogate
    row_ind, col_ind = linear_sum_assignment(c)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    best_p = p.copy()
    best_cost = _qap_cost(p, a, b)
    
    # --- 2. Vectorized steepest descent local search with restarts ---
    def compute_deltas(p, a, b):
        """Compute the delta cost for all pairs (i, j) of swapping p[i] and p[j]."""
        # Delta for swapping i and j:
        # C' - C = (a[i,i] - a[j,j]) * (b[p[i],p[i]] - b[p[j],p[j]])
        #         + 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
        #         + 2 * sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k],p[i]] - b[p[k],p[j]])
        
        # Since b is symmetric: b[p[k],p[i]] = b[p[i],p[k]]
        # So: 2 * sum_{k != i,j} (a[i,k] - a[j,k] + a[k,i] - a[k,j]) * (b[p[i],p[k]] - b[p[j],p[k]])
        #     = 2 * sum_{k != i,j} a_sym[i,k] - a_sym[j,k] ... wait
        # a[i,k] - a[j,k] + a[k,i] - a[k,j] = (a[i,k] + a[k,i]) - (a[j,k] + a[k,j]) = a_sym[i,k] - a_sym[j,k]
        
        # Let M[i, k] = sum over ... no.
        # Let's define: D[i, k] = a_sym[i, k]  (this is a[i,k] + a[k,i])
        # Then the term is: 2 * sum_{k != i,j} (D[i,k] - D[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
        
        # Let's expand: 2 * sum_{k != i,j} D[i,k] b[p[i],p[k]] - 2 * sum_{k != i,j} D[i,k] b[p[j],p[k]]
        #               - 2 * sum_{k != i,j} D[j,k] b[p[i],p[k]] + 2 * sum_{k != i,j} D[j,k] b[p[j],p[k]]
        
        # Let S1 = sum_k D[i,k] b[p[i],p[k]]  (includes k=i and k=j)
        # Let S2 = sum_k D[i,k] b[p[j],p[k]]
        # Let S3 = sum_k D[j,k] b[p[i],p[k]]
        # Let S4 = sum_k D[j,k] b[p[j],p[k]]
        
        # sum_{k != i,j} D[i,k] b[p[i],p[k]] = S1 - D[i,i] b[p[i],p[i]] - D[i,j] b[p[i],p[j]]
        # sum_{k != i,j} D[i,k] b[p[j],p[k]] = S2 - D[i,i] b[p[j],p[i]] - D[i,j] b[p[j],p[j]]
        # sum_{k != i,j} D[j,k] b[p[i],p[k]] = S3 - D[j,i] b[p[i],p[i]] - D[j,j] b[p[i],p[j]]
        # sum_{k != i,j} D[j,k] b[p[j],p[k]] = S4 - D[j,i] b[p[j],p[i]] - D[j,j] b[p[j],p[j]]
        
        # Note b[p[i],p[j]] = b[p[j],p[i]] (symmetry)
        
        # This is getting complex. Let's use the matrix multiplication approach.
        # Let Bp = b[np.ix_(p, p)]
        # Let M = a_sym @ Bp  (M[i, k] = sum_j a_sym[i, j] * Bp[j, k] = sum_j (a[i,j]+a[j,i]) * b[p[j], p[k]])
        # Wait, that's not quite right for the delta formula.
        
        # Let's stick to the O(n^2) per swap evaluation but vectorize over all pairs.
        # For each pair (i, j), delta = (a[i,i] - a[j,j]) * (Bp[i,i] - Bp[j,j])
        #              + 2 * (sum_{k} (a_sym[i,k] - a_sym[j,k]) * Bp[i,k]) 
        #              - 2 * (sum_{k} (a_sym[i,k] - a_sym[j,k]) * Bp[j,k])
        #              - correction for k=i and k=j.
        
        # Let's use the formula:
        # delta(i,j) = 2 * (M[i,i] - M[i,j] - M[j,i] + M[j,j]) 
        #              where M = a_sym @ Bp ... no.
        
        # Let's re-derive carefully.
        # C = sum_{i,j} a[i,j] Bp[i,j]
        # Swap p[i] <-> p[j].
        # Change in cost:
        # Terms involving i or j change.
        # Old: a[i,i]Bp[i,i] + a[i,j]Bp[i,j] + a[j,i]Bp[j,i] + a[j,j]Bp[j,j] + sum_{k!=i,j}(a[i,k]Bp[i,k] + a[k,i]Bp[k,i] + a[j,k]Bp[j,k] + a[k,j]Bp[k,j])
        # New: a[i,i]Bp[j,j] + a[i,j]Bp[i,j] (unchanged? No, Bp[i,j] becomes b[p[i],p[j]] which is same as before but swapped indices in Bp? 
        #       Wait, Bp is defined by p. If we swap p[i] and p[j], then new Bp' has:
        #       Bp'[i,i] = Bp[j,j]
        #       Bp'[i,j] = Bp[j,i] = Bp[i,j] (symmetry)
        #       Bp'[j,i] = Bp[i,j]
        #       Bp'[j,j] = Bp[i,i]
        #       Bp'[i,k] = Bp[j,k] for k!=i,j
        #       Bp'[k,i] = Bp[k,j] for k!=i,j
        #       Bp'[j,k] = Bp[i,k] for k!=i,j
        #       Bp'[k,j] = Bp[k,i] for k!=i,j
        
        # New C' = a[i,i]Bp[j,j] + a[i,j]Bp[i,j] + a[j,i]Bp[i,j] + a[j,j]Bp[i,i]
        #          + sum_{k!=i,j} (a[i,k]Bp[j,k] + a[k,i]Bp[k,j] + a[j,k]Bp[i,k] + a[k,j]Bp[k,i])
        
        # Delta = C' - C
        # = a[i,i](Bp[j,j]-Bp[i,i]) + a[j,j](Bp[i,i]-Bp[j,j])
        #   + a[i,j](Bp[i,j]-Bp[i,j]) + a[j,i](Bp[i,j]-Bp[j,i])  [these cancel if symmetric]
        #   + sum_{k!=i,j} [a[i,k](Bp[j,k]-Bp[i,k]) + a[k,i](Bp[k,j]-Bp[k,i]) + a[j,k](Bp[i,k]-Bp[j,k]) + a[k,j](Bp[k,i]-Bp[k,j])]
        
        # Group terms:
        # = (a[i,i]-a[j,j])(Bp[j,j]-Bp[i,i])
        #   + sum_{k!=i,j} (a[i,k]-a[j,k])(Bp[j,k]-Bp[i,k])
        #   + sum_{k!=i,j} (a[k,i]-a[k,j])(Bp[k,j]-Bp[k,i])
        
        # Using symmetry Bp[k,j] = Bp[j,k] and Bp[k,i] = Bp[i,k]:
        # = (a[i,i]-a[j,j])(Bp[j,j]-Bp[i,i])
        #   + sum_{k!=i,j} (a[i,k]-a[j,k])(Bp[j,k]-Bp[i,k])
        #   + sum_{k!=i,j} (a[k,i]-a[k,j])(Bp[j,k]-Bp[i,k])
        # = (a[i,i]-a[j,j])(Bp[j,j]-Bp[i,i])
        #   + sum_{k!=i,j} (a[i,k]+a[k,i]-a[j,k]-a[k,j])(Bp[j,k]-Bp[i,k])
        # = (a[i,i]-a[j,j])(Bp[j,j]-Bp[i,i])
        #   + sum_{k!=i,j} (a_sym[i,k]-a_sym[j,k])(Bp[j,k]-Bp[i,k])
        
        # Let D = a_sym. Let V = Bp[j,:] - Bp[i,:] (vector).
        # Sum_{k!=i,j} (D[i,k]-D[j,k]) V[k]
        # = sum_k (D[i,k]-D[j,k]) V[k] - (D[i,i]-D[j,i])V[i] - (D[i,j]-D[j,j])V[j]
        
        # Let's compute sum_k (D[i,k]-D[j,k]) V[k] = D[i,:] @ V - D[j,:] @ V
        # V[k] = Bp[j,k] - Bp[i,k]
        # D[i,:] @ V = sum_k D[i,k] Bp[j,k] - sum_k D[i,k] Bp[i,k]
        # D[j,:] @ V = sum_k D[j,k] Bp[j,k] - sum_k D[j,k] Bp[i,k]
        
        # This is still O(n^2) per pair if done naively. 
        # But we can vectorize: for all i, j, compute the delta.
        # Let's use the matrix M = a_sym @ Bp (n x n).
        # M[i, k] = sum_j a_sym[i, j] Bp[j, k]
        # This doesn't directly give us what we need.
        
        # Alternative: compute the delta matrix in O(n^3) using BLAS.
        # For a fixed j, the delta for all i can be computed.
        # But we want all pairs.
        
        # Let's just do O(n^2) per iteration with NumPy vectorization over i for each j? No.
        # Let's compute the full delta matrix in O(n^3) once, then find the best swap.
        # Delta(i, j) = (a[i,i]-a[j,j])*(Bp[j,j]-Bp[i,i]) + sum_{k!=i,j} (a_sym[i,k]-a_sym[j,k])*(Bp[j,k]-Bp[i,k])
        
        # Let S1[i, j] = sum_k a_sym[i,k] Bp[j,k]  (this is a_sym @ Bp^T? No, Bp is symmetric)
        # S1 = a_sym @ Bp  (since Bp is symmetric, Bp^T = Bp)
        # S1[i, j] = sum_k a_sym[i,k] Bp[k, j] = sum_k a_sym[i,k] Bp[j, k]
        # So S1 = a_sym @ Bp.
        
        # sum_k a_sym[i,k] Bp[j,k] = S1[i, j]
        # sum_k a_sym[j,k] Bp[j,k] = S1[j, j]
        # sum_k a_sym[i,k] Bp[i,k] = S1[i, i]
        # sum_k a_sym[j,k] Bp[i,k] = S1[j, i]
        
        # sum_{k!=i,j} (a_sym[i,k]-a_sym[j,k]) Bp[j,k] 
        # = (sum_k a_sym[i,k] Bp[j,k] - sum_k a_sym[j,k] Bp[j,k]) 
        #   - (a_sym[i,i]-a_sym[j,i])Bp[j,i] - (a_sym[i,j]-a_sym[j,j])Bp[j,j]
        # = (S1[i,j] - S1[j,j]) - (a_sym[i,i]-a_sym[j,i])Bp[i,j] - (a_sym[i,j]-a_sym[j,j])Bp[j,j]
        
        # Similarly for Bp[i,k]:
        # sum_{k!=i,j} (a_sym[i,k]-a_sym[j,k]) Bp[i,k]
        # = (S1[i,i] - S1[j,i]) - (a_sym[i,i]-a_sym[j,i])Bp[i,i] - (a_sym[i,j]-a_sym[j,j])Bp[i,j]
        
        # So the sum term is:
        # [ (S1[i,j] - S1[j,j]) - (a_sym[i,i]-a_sym[j,i])Bp[i,j] - (a_sym[i,j]-a_sym[j,j])Bp[j,j] ]
        # - [ (S1[i,i] - S1[j,i]) - (a_sym[i,i]-a_sym[j,i])Bp[i,i] - (a_sym[i,j]-a_sym[j,j])Bp[i,j] ]
        # = S1[i,j] - S1[j,j] - S1[i,i] + S1[j,i]
        #   - (a_sym[i,i]-a_sym[j,i])(Bp[i,j] - Bp[i,i])
        #   - (a_sym[i,j]-a_sym[j,j])(Bp[j,j] - Bp[i,j])
        
        # Note a_sym[i,i] = 2*a[i,i], a_sym[i,j] = a[i,j]+a[j,i]
        
        # This is O(n^2) to compute the full delta matrix after O(n^3) for S1.
        # Let's implement this.
        
        Bp = b[np.ix_(p, p)]
        S1 = a_sym.astype(np.float64) @ Bp.astype(np.float64)
        
        # Initialize delta matrix
        delta = np.zeros((n, n), dtype=np.float64)
        
        # Term 1: (a[i,i]-a[j,j])*(Bp[j,j]-Bp[i,i])
        diag_a = np.diag(a)
        diag_Bp = np.diag(Bp)
        T1 = (diag_a[:, None] - diag_a[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
        
        # Term 2: S1[i,j] - S1[j,j] - S1[i,i] + S1[j,i]
        T2 = S1 - S1[None, :].T - S1[None, :].T + S1[:, None]  # Wait, S1[i,j] is S1[i,j]. S1[j,j] is row j col j.
        # Let's be careful.
        # S1[i,j] is just S1[i,j]
        # S1[j,j] is a matrix where element (i,j) is S1[j,j]
        # S1[i,i] is a matrix where element (i,j) is S1[i,i]
        # S1[j,i] is a matrix where element (i,j) is S1[j,i]
        
        T2 = S1.copy()
        T2 -= np.tile(S1.diagonal(), (n, 1))  # Subtract S1[j,j]
        T2 -= np.tile(S1.diagonal(), (1, n))  # Subtract S1[i,i]
        T2 += S1.T  # Add S1[j,i]
        
        # Term 3: - (a_sym[i,i]-a_sym[j,i])*(Bp[i,j] - Bp[i,i])
        # a_sym[i,i] = 2*a[i,i]. a_sym[j,i] = a[j,i]+a[i,j]
        # Let's compute A_sym_diag_diff[i, j] = a_sym[i,i] - a_sym[j,i]
        A_sym_diag = np.diag(a_sym)
        A_sym_diag_diff = A_sym_diag[:, None] - a_sym.T  # a_sym[j,i] is a_sym.T[i,j]
        
        Bp_diff = Bp - np.tile(np.diag(Bp), (1, n))  # Bp[i,j] - Bp[i,i]
        
        T3 = -A_sym_diag_diff * Bp_diff
        
        # Term 4: - (a_sym[i,j]-a_sym[j,j])*(Bp[j,j] - Bp[i,j])
        A_sym_jj_diff = a_sym - np.tile(A_sym_diag, (1, n))  # a_sym[i,j] - a_sym[j,j]
        Bp_jj_diff = np.tile(np.diag(Bp), (1, n)) - Bp  # Bp[j,j] - Bp[i,j]
        
        T4 = -A_sym_jj_diff * Bp_jj_diff
        
        delta = T1 + T2 + T3 + T4
        
        # Set diagonal to inf (no self-swap)
        np.fill_diagonal(delta, np.inf)
        
        return delta
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        delta = compute_deltas(p, a, b)
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= 0:
            improved = False
            # Try a perturbation if we have time
            if time.time() < deadline - 5:
                # Random perturbation: swap 2-5 random pairs
                num_swaps = np.random.randint(2, 6)
                for _ in range(num_swaps):
                    i1, j1 = np.random.randint(0, n, 2)
                    if i1 != j1:
                        p[i1], p[j1] = p[j1], p[i1]
                # Update best if perturbed is better? No, just restart local search.
                # But we need to update best_p if the perturbed state is better.
                # For now, just restart.
                improved = True
            else:
                improved = False
        else:
            p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(p, a, b)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
