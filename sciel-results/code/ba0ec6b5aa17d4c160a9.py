# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # --- Precompute helpers ---
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    A_sym = a + a.T  # Symmetric part for vectorized delta

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def two_opt_delta(p, i, j):
        """Compute delta cost for swapping facilities i and j in permutation p."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        d = b[lj, p] - b[li, p]
        d[i] = 0
        d[j] = 0
        a_diff = a[i, :] - a[j, :] + a[:, i] - a[:, j]
        a_diff[i] = 0
        a_diff[j] = 0
        return int(np.dot(a_diff, d))

    def linearized_cost_matrix(p):
        """
        Compute the linearized cost matrix for Hungarian algorithm.
        cost[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
        """
        C1 = b[:, p]  # C1[l, j] = b[l, p[j]]
        C2 = b[p, :]  # C2[j, l] = b[p[j], l]
        term1 = a @ C1.T  # term1[i,l] = sum_j a[i,j] * b[l, p[j]]
        term2 = a.T @ C2  # term2[i,l] = sum_j a[j,i] * b[p[j], l]
        return term1 + term2

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        cost = calc_cost_full(p)
        if max_time is None:
            max_time = deadline
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        break
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def vectorized_local_search(p, max_time=None, max_passes=50):
        """
        Vectorized steepest-descent 2-opt local search.
        Uses A_sym and D matrix for fast delta computation.
        """
        if max_time is None:
            max_time = deadline
        p = p.copy()
        D = b[np.ix_(p, p)].copy()
        cost = int(np.sum(a * D))
        
        passes = 0
        while passes < max_passes and time.time() < max_time:
            passes += 1
            improved = False
            
            for i in range(n - 1):
                if time.time() > max_time:
                    break
                    
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation for swapping i with each j in js
                # Delta(i, j) = sum_{k != i, j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[p[j], p[k]] - b[p[i], p[k]])
                #             + (a[i,j] + a[j,i] - a[j,j] - a[i,i]) * (b[p[j], p[i]] - b[p[i], p[j]])
                # Let's use the formula:
                # delta = sum_{k != i, j} A_sym[i, k] * (D[j, k] - D[i, k]) - A_sym[j, k] * (D[j, k] - D[i, k]) 
                # Wait, simpler:
                # delta = sum_{k != i, j} (A_sym[i, k] - A_sym[j, k]) * (D[j, k] - D[i, k])
                #       + (A_sym[i, j] - A_sym[j, j]) * (D[j, i] - D[i, j])  # correction for k=i, j terms?
                
                # Let's derive carefully.
                # Cost = sum_{u,v} a[u,v] D[u,v]
                # Swap i, j in p.
                # New D': D'[i,k] = D[j,k], D'[j,k] = D[i,k] for k != i,j.
                # D'[i,j] = D[j,i], D'[j,i] = D[i,j]. D'[i,i]=D[i,i], D'[j,j]=D[j,j].
                
                # Change in cost:
                # Terms involving i or j as row or col.
                # 1. Terms where u=i, v!=i,j: a[i,v] (D[j,v] - D[i,v])
                # 2. Terms where u=j, v!=i,j: a[j,v] (D[i,v] - D[j,v])
                # 3. Terms where v=i, u!=i,j: a[u,i] (D[j,u] - D[i,u])
                # 4. Terms where v=j, u!=i,j: a[u,j] (D[i,u] - D[j,u])
                # 5. Terms where {u,v} = {i,j}:
                #    Old: a[i,j]D[i,j] + a[j,i]D[j,i]
                #    New: a[i,j]D[j,i] + a[j,i]D[i,j]
                #    Diff: (a[i,j] - a[j,i])(D[j,i] - D[i,j])
                
                # Sum 1+2+3+4:
                # sum_{k != i,j} [ a[i,k] - a[j,k] + a[k,i] - a[k,j] ] * (D[j,k] - D[i,k])
                # = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                
                # So total delta = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #                 + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                
                # Vectorized:
                # diff_A = A_sym[i, js] - A_sym[js, :]  # shape (len(js), n) -> need column k
                # Actually:
                # diff_A_k = A_sym[i, k] - A_sym[j, k] for k in js
                # diff_D_k = D[j, k] - D[i, k] for k in js
                
                # We need sum over k != i, j.
                # Let's compute for all k in js (which are != i).
                # But k can be i? No, js starts from i+1. So k != i.
                # However, k can be j? No, j is one of the elements in js?
                # Wait, we are iterating i, and js = i+1..n-1.
                # One of the candidates j is in js.
                # The formula sum_{k != i, j} excludes k=j as well.
                # So we must exclude k=j from the sum over k in js.
                
                # Let's compute S = sum_{k in js, k != j} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                # This is hard to vectorize for all j simultaneously because the exclusion depends on j.
                
                # Alternative: Compute for all k in js including j, then subtract the term for k=j.
                # Sum_all = sum_{k in js} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                # Term_j = (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
                # Sum_excl = Sum_all - Term_j
                
                # Then add the cross term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                
                # Let's vectorize over j in js.
                # For each j in js:
                # A_i = A_sym[i, :]  # (n,)
                # A_j = A_sym[j, :]  # (n,)
                # diff_A = A_i - A_j  # (n,)
                # D_j = D[j, :]  # (n,)
                # D_i = D[i, :]  # (n,)
                # diff_D = D_j - D_i  # (n,)
                # Sum_all = dot(diff_A, diff_D)
                # Term_j = diff_A[j] * diff_D[j]
                # Sum_excl = Sum_all - Term_j
                # Cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                # Delta = Sum_excl + Cross
                
                # We can compute this for all j in js at once.
                # js: (m,)
                # A_sym[i, js]: (m,)
                # A_sym[js, :]: (m, n)
                # diff_A_matrix = A_sym[i, js][:, None] - A_sym[js, :]  # (m, n)
                # D[j, :]: (m, n) -> D[js, :]
                # D[i, :]: (n,) -> D[i, :][None, :]
                # diff_D_matrix = D[js, :] - D[i, :][None, :]  # (m, n)
                
                # Sum_all = np.sum(diff_A_matrix * diff_D_matrix, axis=1)  # (m,)
                # Term_j = diff_A_matrix[np.arange(m), js] * diff_D_matrix[np.arange(m), js]  # (m,)
                # Sum_excl = Sum_all - Term_j
                
                # Cross term:
                # a[i, js]: (m,)
                # a[js, i]: (m,)
                # diff_a_cross = a[i, js] - a[js, i]
                # D[js, i]: (m,)
                # D[i, js]: (m,)
                # diff_D_cross = D[js, i] - D[i, js]
                # Cross = diff_a_cross * diff_D_cross
                
                # Deltas = Sum_excl + Cross
                
                diff_A_matrix = A_sym[i, js][:, None] - A_sym[js, :]
                diff_D_matrix = D[js, :] - D[i, :][None, :]
                
                Sum_all = np.sum(diff_A_matrix * diff_D_matrix, axis=1)
                
                # Term_j: index j in js.
                # js is array of indices.
                # We need diff_A_matrix[row, js[row]]
                # row is 0..m-1. js[row] is the actual index j.
                idx = np.arange(len(js))
                Term_j = diff_A_matrix[idx, js] * diff_D_matrix[idx, js]
                
                Sum_excl = Sum_all - Term_j
                
                diff_a_cross = a[i, js] - a[js, i]
                diff_D_cross = D[js, i] - D[i, js]
                Cross = diff_a_cross * diff_D_cross
                
                Deltas = Sum_excl + Cross
                
                min_idx = np.argmin(Deltas)
                min_delta = Deltas[min_idx]
                
                if min_delta < 0:
                    j = js[min_idx]
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    cost += min_delta
                    improved = True
            
            if not improved:
                break
                
        return p, cost

    # --- Phase 1: Initial Solution via Rank Matching ---
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    p = p.copy()

    # Initial polish
    p, current_cost = vectorized_local_search(p, max_time=start_time + 15.0)
    best_p = p.copy()
    best_cost = current_cost

    # --- Phase 2: Iterated Local Search with Hungarian Reassignment ---
    hungarian_count = 0
    no_improve_rounds = 0
    max_no_improve = 3
    iteration = 0

    while time.time() < deadline - 20.0:
        iteration += 1
        # Adaptive perturbation
        if no_improve_rounds > 10:
            perturb_k = min(15, 3 + no_improve_rounds // 2)
        else:
            perturb_k = 3

        # Perturb best solution
        p = best_p.copy()
        n_swaps = np.random.randint(2, perturb_k + 1)
        for _ in range(n_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        # Local search from perturbed
        p, p_cost = vectorized_local_search(p, max_time=min(deadline - 10.0, time.time() + 5.0))
        
        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            cost_mat = linearized_cost_matrix(p)
            if hungarian_count > 0 and no_improve_rounds > 0:
                noise_scale = 1.0 + no_improve_rounds * 0.5
                cost_mat = cost_mat + np.random.randn(n, n) * noise_scale
                
            try:
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                new_p = np.empty(n, dtype=np.int64)
                new_p[row_ind] = col_ind
                new_cost = calc_cost_full(new_p)
                
                # Polish the Hungarian result
                new_p, new_cost = vectorized_local_search(new_p, max_time=min(deadline - 10.0, time.time() + 5.0))
                
                if new_cost < p_cost:
                    p = new_p
                    p_cost = new_cost
            except Exception:
                pass

        if p_cost < best_cost:
            best_p = p.copy()
            best_cost = p_cost
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

    # --- Phase 3: Final Polish ---
    p_polish = best_p.copy()
    p_polish, final_cost = vectorized_local_search(p_polish, max_time=deadline - 2.0)
    if final_cost < best_cost:
        best_p = p_polish
        best_cost = final_cost

    return best_p.tolist()

# EVOLVE-BLOCK-END
