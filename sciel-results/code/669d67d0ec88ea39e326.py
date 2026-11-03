import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Initialization: Greedy / Rank matching
    # Simple heuristic: sort facilities by sum of flows, locations by sum of distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Helper to compute cost
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    best_cost = calc_cost(p)
    best_p = p.copy()
    
    # Precompute S = a + a.T
    S = a + a.T
    
    # M[v, k] = b[v, p[k]]
    M = b[:, p]
    # P[i, l] = S[i, :] @ M[l, :] = sum_k S[i, k] b[l, p[k]]
    P = S @ M.T
    # V[j] = P[j, p[j]]
    V = P[np.arange(n), p]
    
    # Tabu parameters
    tabu_tenure = 30
    tabu_list = np.zeros((n, n), dtype=np.int32)
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        iteration += 1
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Iterate over i
        for i in range(n):
            u = p[i]
            
            # W = S @ M[u, :]
            W = S @ M[u, :]
            
            # C1[j] = P[i, p[j]] - W[i]
            # P[i, p] is a vector where element j is P[i, p[j]]
            C1 = P[i, p] - W[i]
            
            # C2[j] = V[j] - W[j]
            C2 = V - W
            
            SumPart = C1 - C2
            
            # D_i[j] = M[p[j], i] - M[u, i]
            # M[:, i] is column i of M. M[p, i] is element at row p[j], col i for each j?
            # M is (n, n). M[row, col].
            # We want vector D_i where D_i[j] = M[p[j], i].
            # This is M[p, i]? No. M[p, i] is not valid indexing for a vector of rows.
            # We need M[p, :][:, i].
            # Let's extract column i of M: col_i = M[:, i].
            # Then D_i[j] = col_i[p[j]].
            col_i = M[:, i]
            D_i = col_i[p] - M[u, i]
            
            # D_j[j] = M[p[j], j] - M[u, j]
            # M[p[j], j] is element at row p[j], col j.
            # This is np.diag(M[p, :])?
            # M[p, :] is a matrix where row j is M[p[j], :].
            # So np.diag(M[p, :]) gives M[p[j], j].
            diag_Mp = np.diag(M[p, :])
            D_j = diag_Mp - M[u, p]
            
            # A_ii[j] = 2*a[i, i] - a[j, i] - a[i, j]
            A_ii = 2 * a[i, i] - a[:, i] - a[i, :]
            
            # A_ij[j] = a[i, j] + a[j, i] - 2*a[j, j]
            A_ij = a[i, :] + a[:, i] - 2 * np.diag(a)
            
            # Term 3: - A_ii * D_i
            T3 = -A_ii * D_i
            
            # Term 4: - A_ij * D_j
            T4 = -A_ij * D_j
            
            # Term 5: (a[i, j] - a[j, i]) * (M[p[j], u] - M[u, p[j]])
            # DiffA = a[i, :] - a[:, i]
            # M[p[j], u] is col_u[p[j]]
            col_u = M[:, u]
            M_p_u = col_u[p]
            # M[u, p[j]] is M[u, p]
            M_u_p = M[u, p]
            T5 = (a[i, :] - a[:, i]) * (M_p_u - M_u_p)
            
            deltas = SumPart + T3 + T4 + T5
            
            # Exclude j == i
            deltas[i] = np.inf
            
            # Check tabu
            # tabu_list[i, j] > iteration - tabu_tenure means tabu
            tabu_mask = tabu_list[i, :] > iteration - tabu_tenure
            deltas[tabu_mask] = np.inf
            
            j_best = np.argmin(deltas)
            d_best = deltas[j_best]
            
            if d_best < best_delta:
                best_delta = d_best
                best_i = i
                best_j = j_best
        
        if best_i == -1 or np.isinf(best_delta):
            # No valid move found (all tabu or no improvement? We allow worsening moves in ILS, but here we just break if stuck)
            # If we are doing simple local search, we break.
            # If we want ILS, we should perturb.
            # For now, break to avoid infinite loop if no moves.
            break
            
        # Apply move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        
        # Update Cost
        # Instead of recalculating full cost, use delta?
        # The delta formula derived above is for the change in cost.
        # So new_cost = best_cost + best_delta
        # But floating point errors might accumulate.
        # Recalculate every 100 steps? Or just trust delta for now and verify occasionally?
        # Given the integer nature, we should be careful.
        # Let's recalculate cost every step for safety? O(n^2).
        # 256^2 = 65k ops. 150s / 65k ~ 2000 steps.
        # If we do O(n^2) delta search AND O(n^2) cost calc, it's 2 * O(n^2).
        # Still feasible.
        
        cost = calc_cost(p)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update M and P and V incrementally
        # Swap columns best_i and best_j in M
        M[:, best_i], M[:, best_j] = M[:, best_j].copy(), M[:, best_i].copy()
        
        # Update P
        # P = S @ M.T
        # Only rows of P corresponding to columns of M that changed?
        # M columns changed: best_i, best_j.
        # So P[:, best_i] and P[:, best_j] change?
        # P[i, k] = S[i, :] @ M[k, :].
        # M[k, best_i] changed. So P[i, best_i] changes for all i.
        # M[k, best_j] changed. So P[i, best_j] changes for all i.
        P[:, best_i] = S @ M[best_i, :]
        P[:, best_j] = S @ M[best_j, :]
        
        # Update V
        # V[j] = P[j, p[j]]
        # p changed at best_i and best_j.
        # V[best_i] = P[best_i, p[best_i]]
        # V[best_j] = P[best_j, p[best_j]]
        # Note p[best_i] is now old p[best_j] and vice versa.
        V[best_i] = P[best_i, p[best_i]]
        V[best_j] = P[best_j, p[best_j]]
        
        # Update Tabu
        tabu_list[best_i, best_j] = iteration
        tabu_list[best_j, best_i] = iteration
        
        # Optional: Random restart if stuck
        if no_improve_count > 50:
            # Perturb
            for _ in range(5):
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                p[idx1], p[idx2] = p[idx2], p[idx1]
                # Update M, P, V
                M[:, idx1], M[:, idx2] = M[:, idx2].copy(), M[:, idx1].copy()
                P[:, idx1] = S @ M[idx1, :]
                P[:, idx2] = S @ M[idx2, :]
                V[idx1] = P[idx1, p[idx1]]
                V[idx2] = P[idx2, p[idx2]]
            no_improve_count = 0

    return best_p.tolist()
