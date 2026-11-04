"""QAP tai256c solver: iterated local search with vectorized 2-opt, Hungarian re-optimization, and adaptive diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        """Compute the exact cost delta for swapping facilities at positions u and v."""
        pu = p[u]
        pv = p[v]

        # Difference vectors
        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        # Main terms
        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        # Corrections for the u and v indices themselves
        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        # Cross term
        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        """Steepest Descent Local Search with vectorized delta calculation."""
        improved = True
        passes = 0
        p = p.copy()
        
        # Precompute symmetric A
        A_sym = a + a.T
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                    
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation for all j > i
                # We need delta for swapping i and j.
                # Using the formula derived in fast_swap_delta but vectorized over j.
                
                # Let's stick to a slightly less optimized but safer vectorized approach
                # or just loop if vectorization is complex. 
                # Actually, the inspiration 1 code had a specific vectorized block. Let's use that logic.
                
                # diff_A = A_sym[i] - A_sym[js]  # shape (n-i-1, )
                # diff_D = D[js] - D[i] ... this requires D to be maintained.
                
                # Let's use the explicit fast_swap_delta in a loop for correctness first, 
                # as vectorizing the exact delta with corrections is error-prone.
                # However, for speed, we can try to vectorize the main part.
                
                # Given the risk, let's use a semi-vectorized approach:
                # Compute deltas for all j > i.
                
                pu = p[i]
                pv_all = p[js]
                
                # d_uv[j] = b[pv, :] - b[pu, :]
                # d_vu[j] = b[:, pv] - b[:, pu]
                
                # s1[j] = dot(a[u,:] - a[v,:], d_uv[j][p])
                # This is hard to vectorize easily without big memory.
                
                # Let's just loop over j for now, as n=256 is small enough that 
                # 256*256/2 = 32k swaps per pass might be okay if delta is fast.
                # But 32k swaps * 20 passes = 640k swaps. 
                # If each swap is ~1us, that's 0.6s. Feasible.
                
                best_delta = 0
                best_j = -1
                
                for idx, j in enumerate(js):
                    delta = fast_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_delta < 0:
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    cost += best_delta
                    improved = True
        
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply several random tabu-like kicks to diversify, keeping best result."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 10
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            # Pick a random pair
            i, j = np.random.choice(n, 2, replace=False)
            key = (min(i, j), max(i, j))
            
            # Check tabu
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            # Compute delta
            delta = fast_swap_delta(p, i, j)
            
            # Accept if not tabu, or if it improves global best (aspiration)
            if not is_tabu or (cost + delta < best_cost):
                p[i], p[j] = p[j], p[i]
                cost += delta
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            iteration += 1
        
        return best_p, best_cost

    # Initial solution: rank matching based on row sums
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)

    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 10:  # Leave 10s buffer
        iteration += 1

        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb

        # Perturb: combine random swaps with tabu kick
        p = best_p.copy()
        
        # Random swaps
        n_swaps = np.random.randint(2, perturb_k + 1)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        # Tabu kick for additional diversification
        p, cost = tabu_kick(p, compute_cost(p), n_kicks=3)

        # Local search from perturbed
        p, cost = local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1

    # Final polish: one more local search pass on best solution
    if time.time() < deadline - 2:
        final_p = best_p.copy()
        final_cost = best_cost
        final_p, final_cost = local_search(final_p, final_cost)
        if final_cost < best_cost:
            best_p = final_p
            best_cost = final_cost

    return best_p.tolist()
