# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute symmetric A for delta calculations
    A_sym = a + a.T

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

    def two_opt_polish(p, max_time=None):
        """Greedy 2-opt local search. Returns (p, cost)."""
        if max_time is None:
            max_time = deadline
        cost = calc_cost_full(p)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() > max_time:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > max_time:
                        return p, cost
                    delta = two_opt_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def vectorized_local_search(p, max_time=None):
        """
        Faster local search using vectorized delta calculation per row.
        This is the structural change: instead of O(n^2) individual delta calls
        with O(n) cost each (O(n^3) per pass), we compute all deltas for a fixed i
        in O(n) vectorized operations.
        """
        if max_time is None:
            max_time = deadline
        p = p.copy()
        cost = calc_cost_full(p)
        D = b[np.ix_(p, p)].copy()
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(n - 1):
                if time.time() > max_time:
                    return p, cost
                
                js = np.arange(i + 1, n)
                
                # Vectorized delta calculation for swapping i with any j in js
                # Delta(i,j) = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                #             + correction terms for k=i and k=j
                
                # Term 1: sum over k of (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                # We can compute this as dot product of (A_sym[i] - A_sym[js]) and (D[js] - D[i])
                # But D[js] is a matrix, D[i] is a vector.
                # Let's expand:
                # sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
                # = sum_k A_sym[i,k]*D[j,k] - sum_k A_sym[i,k]*D[i,k] - sum_k A_sym[j,k]*D[j,k] + sum_k A_sym[j,k]*D[i,k]
                
                # This is getting complex. Let's use the formula from Inspiration 1.
                # S = np.sum(diff_A * diff_D, axis=1) where diff_A = A_sym[i] - A_sym[js] (shape n-1-i, n)
                # diff_D = D[js] - D[i] (shape n-1-i, n)
                
                diff_A = A_sym[i] - A_sym[js]  # (n-1-i, n)
                diff_D = D[js] - D[i]          # (n-1-i, n)
                
                S = np.sum(diff_A * diff_D, axis=1)  # (n-1-i,)
                
                # Correction terms from Inspiration 1
                # coeff_i = A_sym[i, i] - A_sym[js, i]
                coeff_i = A_sym[i, i] - A_sym[js, i]
                term_i = coeff_i * (D[js, i] - D[i, i])
                
                # coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                term_j = coeff_j * (D[js, js] - D[i, js])
                
                # diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                
                # cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
                cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
                
                deltas = S - term_i - term_j + diag_term + cross_term
                
                min_idx = np.argmin(deltas)
                min_delta = deltas[min_idx]
                
                if min_delta < 0:
                    j = js[min_idx]
                    p[i], p[j] = p[j], p[i]
                    
                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    
                    cost += int(min_delta)
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

    def spectral_init():
        """
        Spectral bisection based initialization.
        Uses the structure of the distance matrix b.
        """
        b_sym = (b + b.T) / 2.0
        d = b_sym.sum(axis=1)
        L = np.diag(d) - b_sym
        
        try:
            from scipy.linalg import eigh
            vals, vecs = eigh(L)
            idx = 1
            fiedler = vecs[:, idx]
        except Exception:
            return np.random.permutation(n).astype(np.int64)
        
        loc_order = np.argsort(fiedler, kind='stable')
        fac_order = np.argsort(a.sum(axis=1), kind='stable')
        
        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    # --- Phase 1: Initial Solutions ---
    best_p = None
    best_cost = float('inf')
    
    # 1. Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    p_rank, c_rank = vectorized_local_search(p_rank, max_time=start_time + 5.0)
    if c_rank < best_cost:
        best_cost = c_rank
        best_p = p_rank.copy()

    # 2. Spectral Init
    if time.time() < start_time + 10.0:
        p_spec = spectral_init()
        p_spec, c_spec = vectorized_local_search(p_spec, max_time=start_time + 15.0)
        if c_spec < best_cost:
            best_cost = c_spec
            best_p = p_spec.copy()

    # 3. Random Init
    if time.time() < start_time + 20.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = vectorized_local_search(p_rand, max_time=start_time + 25.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search with Simulated Annealing ---
    current_p = best_p.copy()
    current_cost = best_cost
    
    # SA Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 1.0
    iteration = 0
    consecutive_no_improve = 0

    while time.time() < deadline - 10.0:
        iteration += 1
        
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, 3 + consecutive_no_improve // 2)
        else:
            perturb_k = 3

        # Perturb
        num_swaps = max(perturb_k, n // 30) 
        p_iter = current_p.copy()
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p_iter[i], p_iter[j] = p_iter[j], p_iter[i]
                
        # Polish with faster vectorized search
        p_iter, c_iter = vectorized_local_search(p_iter, max_time=deadline - 5.0)
        
        # Periodic Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 25.0:
            h_p = hungarian_reopt(p_iter)
            h_p, h_cost = vectorized_local_search(h_p, max_time=deadline - 5.0)
            if h_cost < c_iter:
                p_iter = h_p
                c_iter = h_cost

        # Acceptance Criterion
        if c_iter < best_cost:
            best_cost = c_iter
            best_p = p_iter.copy()
            current_p = p_iter.copy()
            current_cost = c_iter
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            # Simulated Annealing acceptance
            delta = c_iter - current_cost
            if delta < 0:
                # Better than current, accept
                current_p = p_iter.copy()
                current_cost = c_iter
            else:
                # Worse, accept with probability
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_p = p_iter.copy()
                        current_cost = c_iter
                # Else reject (keep current)
        
        T *= cooling_rate

    # --- Phase 3: Final Polish ---
    if time.time() < deadline - 1.0:
        p_final = best_p.copy()
        p_final, c_final = vectorized_local_search(p_final, max_time=deadline - 0.5)
        if c_final < best_cost:
            best_p = p_final
            best_cost = c_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
