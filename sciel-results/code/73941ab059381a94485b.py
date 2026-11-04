# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill

    # Precompute symmetric A for delta calculations
    A_sym = a + a.T

    def calc_cost_full(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def fast_swap_delta(p, u, v):
        """
        Exact O(n) delta calculation for swapping facilities u and v in permutation p.
        """
        pu = p[u]
        pv = p[v]
        if pu == pv:
            return 0

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return int(s1 + s2 + cross)

    def vectorized_local_search(p, max_time=None, exact=False):
        """
        Fast local search using vectorized delta calculation.
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

                if exact:
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    if best_j == -1:
                        continue
                    min_delta = best_delta
                    j = best_j
                else:
                    js = np.arange(i + 1, n)

                    diff_A = A_sym[i] - A_sym[js]
                    diff_D = D[js] - D[i]

                    S = np.sum(diff_A * diff_D, axis=1)

                    coeff_i = A_sym[i, i] - A_sym[js, i]
                    term_i = coeff_i * (D[js, i] - D[i, i])

                    coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                    term_j = coeff_j * (D[js, js] - D[i, js])

                    diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                    cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])

                    deltas = S - term_i - term_j + diag_term + cross_term

                    min_idx = np.argmin(deltas)
                    min_delta = deltas[min_idx]

                    if min_delta < 0:
                        j = js[min_idx]
                    else:
                        continue

                if min_delta < 0:
                    p[i], p[j] = p[j], p[i]

                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()

                    cost += int(min_delta)
                    improved = True

        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T

        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2

        C = M1 + M2

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def spectral_init():
        """
        Spectral bisection based initialization.
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

    def frank_wolfe_init(max_iter=30, tol=1e-6):
        """
        Frank-Wolfe (conditional gradient) method on the Birkhoff polytope.
        This is a structural change: instead of starting from discrete heuristics,
        we optimize over the continuous relaxation and project via Hungarian.
        """
        # Initialize with a doubly stochastic matrix (uniform)
        X = np.ones((n, n), dtype=np.float64) / n

        for it in range(max_iter):
            if time.time() > deadline - 60.0:
                break

            # Compute gradient: G = (A @ X @ B^T + A^T @ X @ B)
            # The QAP objective is f(X) = sum_{i,j,k,l} a[i,k] * b[j,l] * X[i,j] * X[k,l]
            # Gradient: G[i,j] = 2 * sum_{k,l} a[i,k]*b[j,l]*X[k,l] + 2 * sum_{k,l} a[k,i]*b[l,j]*X[k,l]
            # Simplified: G = 2 * (A @ X @ B.T + A.T @ X @ B)
            G = 2.0 * (a @ X @ b.T + a.T @ X @ b)

            # Linear minimization over Birkhoff polytope: find Y that minimizes <G, Y>
            # This is the assignment problem with cost matrix G
            row_ind, col_ind = linear_sum_assignment(G)
            Y = np.zeros((n, n), dtype=np.float64)
            Y[row_ind, col_ind] = 1.0

            # Compute step size
            step = 2.0 / (it + 2.0)  # Standard FW step size

            # Update
            X_new = (1.0 - step) * X + step * Y

            # Check convergence
            if np.max(np.abs(X_new - X)) < tol:
                X = X_new
                break
            X = X_new

        # Project to permutation via Hungarian on the final X
        row_ind, col_ind = linear_sum_assignment(-X)  # Maximize assignment
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # --- Phase 1: Initial Solutions ---
    best_p = None
    best_cost = float('inf')

    # 1. Hungarian Init
    if time.time() < start_time + 5.0:
        p_hung = hungarian_init()
        p_hung, c_hung = vectorized_local_search(p_hung, max_time=start_time + 8.0)
        if c_hung < best_cost:
            best_cost = c_hung
            best_p = p_hung.copy()

    # 2. Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_row_sums, kind='stable')
    central_order = np.argsort(b_row_sums, kind='stable')
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy_order] = central_order
    p_rank, c_rank = vectorized_local_search(p_rank, max_time=start_time + 12.0)
    if c_rank < best_cost:
        best_cost = c_rank
        best_p = p_rank.copy()

    # 3. Spectral Init
    if time.time() < start_time + 15.0:
        p_spec = spectral_init()
        p_spec, c_spec = vectorized_local_search(p_spec, max_time=start_time + 20.0)
        if c_spec < best_cost:
            best_cost = c_spec
            best_p = p_spec.copy()

    # 4. Frank-Wolfe Init (STRUCTURAL CHANGE)
    if time.time() < start_time + 25.0:
        p_fw = frank_wolfe_init(max_iter=20)
        p_fw, c_fw = vectorized_local_search(p_fw, max_time=start_time + 30.0)
        if c_fw < best_cost:
            best_cost = c_fw
            best_p = p_fw.copy()

    # 5. Random Init
    if time.time() < start_time + 32.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = vectorized_local_search(p_rand, max_time=start_time + 38.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search with Simulated Annealing ---
    current_p = best_p.copy()
    current_cost = best_cost

    # SA Parameters
    T = 2000.0
    cooling_rate = 0.997
    min_T = 5.0
    iteration = 0
    consecutive_no_improve = 0

    while time.time() < deadline - 15.0:
        iteration += 1

        # Adaptive perturbation
        if consecutive_no_improve > 20:
            perturb_k = min(20, 4 + consecutive_no_improve // 3)
        elif consecutive_no_improve > 10:
            perturb_k = min(15, 3 + consecutive_no_improve // 2)
        else:
            perturb_k = 3

        p_iter = current_p.copy()

        # Perturb with mix of swaps and insertions
        num_swaps = max(perturb_k, n // 30)
        for _ in range(num_swaps):
            if random.random() < 0.7:
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p_iter[i], p_iter[j] = p_iter[j], p_iter[i]
            else:
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    val = p_iter[i]
                    p_iter = np.delete(p_iter, i)
                    if j > i:
                        j -= 1
                    p_iter = np.insert(p_iter, j, val)

        # Polish with vectorized search
        p_iter, c_iter = vectorized_local_search(p_iter, max_time=deadline - 10.0)

        # Periodic Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 30.0:
            h_p = hungarian_reopt(p_iter)
            h_p, h_cost = vectorized_local_search(h_p, max_time=deadline - 10.0)
            if h_cost < c_iter:
                p_iter = h_p
                c_iter = h_cost

        # Periodic Frank-Wolfe re-optimization (STRUCTURAL CHANGE)
        if iteration % 5 == 0 and time.time() < deadline - 40.0:
            fw_p = frank_wolfe_init(max_iter=10)
            fw_p, fw_cost = vectorized_local_search(fw_p, max_time=deadline - 10.0)
            if fw_cost < c_iter:
                p_iter = fw_p
                c_iter = fw_cost

        # Acceptance
        if c_iter < best_cost:
            best_cost = c_iter
            best_p = p_iter.copy()
            current_p = p_iter.copy()
            current_cost = c_iter
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            delta = c_iter - current_cost
            if delta < 0:
                current_p = p_iter.copy()
                current_cost = c_iter
            else:
                if T > min_T:
                    scaled_delta = min(delta, 50 * T)
                    prob = math.exp(-scaled_delta / T)
                    if random.random() < prob:
                        current_p = p_iter.copy()
                        current_cost = c_iter

        # Periodic restart
        if consecutive_no_improve > 30 and iteration % 10 == 0:
            current_p = best_p.copy()
            current_cost = best_cost
            consecutive_no_improve = 0

        T = max(T * cooling_rate, min_T)

    # --- Phase 3: Final High-Precision Polish ---
    if time.time() < deadline - 2.0:
        p_final = best_p.copy()
        p_final, c_final = vectorized_local_search(p_final, max_time=deadline - 0.5, exact=True)
        if c_final < best_cost:
            best_cost = c_final
            best_p = p_final

    return best_p.tolist()

# EVOLVE-BLOCK-END
