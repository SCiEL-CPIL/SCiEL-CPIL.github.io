"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching
    # Sort facilities by sum of flows (heavy first)
    # Sort locations by sum of distances (central first)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute symmetric A
    a_sym = a + a.T
    
    # Helper to compute cost
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Precompute C3: C3[j] = sum_k a_sym[j, k] * M[j, k]
    # This depends on M, so it must be updated or recomputed.
    # Since M changes, C3 changes.
    # However, C3[j] only depends on row j of M.
    # If we swap i and j, rows i and j of M change. So C3[i] and C3[j] change.
    # We can update C3 incrementally.
    
    def get_M(p):
        return b[np.ix_(p, p)]
        
    M = get_M(p)
    # C3[j] = sum_k a_sym[j, k] * M[j, k]
    C3 = np.einsum('jk,jk->j', a_sym, M)
    
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # We will do a full pass of best-improvement swaps
        # To avoid O(n^2) Python loops, we loop i and vectorize j
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Compute S_i = a[i, :] + a[:, i]
            S_i = a[i, :] + a[:, i]
            
            # C2 = S_i dot M[i, :]
            C2 = np.dot(S_i, M[i, :])
            
            # C1 = M dot S_i (vector)
            # C1[j] = sum_k M[j, k] * S_i[k]
            C1 = M @ S_i
            
            # C4 = a_sym dot M[i, :] (vector)
            # C4[j] = sum_k a_sym[j, k] * M[i, k]
            C4 = a_sym @ M[i, :]
            
            # Main delta part: Delta_main[j] = C1[j] - C2 - C3[j] + C4[j]
            Delta_main = C1 - C2 - C3 + C4
            
            # Interaction term: a_sym[i, j] * (M[i, j] - M[j, i])
            # M[i, :] is row i. M[:, i] is col i.
            # M[i, j] is element (i,j) of M.
            # M[j, i] is element (j,i) of M.
            # We need vector of M[i, j] for all j: M[i, :]
            # We need vector of M[j, i] for all j: M[:, i]
            Interact = a_sym[i, :] * (M[i, :] - M[:, i])
            
            # Self term: (a[i, i] - a[j, j]) * (M[j, j] - M[i, i])
            # M[j, j] is diagonal of M.
            # M[i, i] is scalar.
            Diag_M = np.diag(M)
            Self = (a[i, i] - a[np.arange(n), np.arange(n)]) * (Diag_M - M[i, i])
            
            # Total Delta
            Delta = Delta_main + Interact + Self
            
            # Mask i
            Delta[i] = np.inf
            
            # Find best j
            best_j = np.argmin(Delta)
            best_delta = Delta[best_j]
            
            if best_delta < 0:
                # Perform swap
                u = p[i]
                v = p[best_j]
                
                # Update p
                p[i], p[best_j] = v, u
                
                # Update M
                # Rows and cols i and best_j change
                # New M[i, :] = Old M[best_j, :]
                # New M[best_j, :] = Old M[i, :]
                # But careful: M[i, best_j] and M[best_j, i] swap values too?
                # Let's check:
                # M_old[i, k] = B[p[i], p[k]]
                # M_new[i, k] = B[p[best_j], p[k]] = M_old[best_j, k] (for k != best_j)
                # M_new[i, best_j] = B[p[best_j], p[i]] = M_old[best_j, i]
                
                # So:
                # M[i, :] = M[best_j, :].copy()
                # M[best_j, :] = M[i, :].copy()  <-- Wait, M[i, :] was just overwritten.
                
                # Correct update:
                temp_row_i = M[i, :].copy()
                temp_row_j = M[best_j, :].copy()
                
                M[i, :] = temp_row_j
                M[best_j, :] = temp_row_i
                
                # Update C3
                # C3[i] depends on M[i, :]
                # C3[best_j] depends on M[best_j, :]
                C3[i] = np.dot(a_sym[i, :], M[i, :])
                C3[best_j] = np.dot(a_sym[best_j, :], M[best_j, :])
                
                current_cost += best_delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                improved = True
        
        if not improved:
            # No improvement in a full pass.
            # Try a random perturbation or restart.
            # Given the time, a simple random swap of 2-3 elements might help escape.
            # Or just break and return best.
            # Let's try a few random perturbations if time permits.
            if time.time() < deadline - 5.0:
                # Perturb best_p
                p = best_p.copy()
                M = get_M(p)
                C3 = np.einsum('jk,jk->j', a_sym, M)
                current_cost = best_cost
                
                # Do 5 random swaps
                for _ in range(5):
                    idx = np.random.choice(n, 2, replace=False)
                    i, j = idx[0], idx[1]
                    p[i], p[j] = p[j], p[i]
                    # Update M and C3 as above
                    temp_row_i = M[i, :].copy()
                    temp_row_j = M[j, :].copy()
                    M[i, :] = temp_row_j
                    M[j, :] = temp_row_i
                    C3[i] = np.dot(a_sym[i, :], M[i, :])
                    C3[j] = np.dot(a_sym[j, :], M[j, :])
                    # Recalculate cost for this perturbed state to continue search
                    current_cost = compute_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
            else:
                break
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
