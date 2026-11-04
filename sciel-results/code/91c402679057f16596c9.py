"""QAP tai256c solver: vectorized 2-opt + Hungarian reopt + continuous relaxation restarts."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s

    # Precompute symmetric A for delta calculations
    A_sym = a + a.T

    def calc_cost_full(perm):
        """Compute exact QAP cost for permutation perm."""
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
        If exact=True, uses the O(n) fast_swap_delta for each pair (slower but precise).
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
        """Spectral bisection based initialization."""
        b_sym = (b + b.T) / 2.0
        d = b_sym.sum(axis=1)
        L = np.diag(d) - b_sym

        try:
            from scipy.linalg import eigh
            vals, vecs = eigh(L)
            fiedler = vecs[:, 1]
        except Exception:
            return np.random.permutation(n).astype(np.int64)

        loc_order = np.argsort(fiedler, kind='stable')
        fac_order = np.argsort(a.sum(axis=1), kind='stable')

        p = np.empty(n, dtype=np.int64)
        p[fac_order] = loc_order
        return p

    def continuous_relaxation_init(seed=42):
        """
        Continuous relaxation approach:
        1. Start with a random permutation
        2. Create a soft permutation matrix P (n x n) where P[i,j] indicates facility i at location j
        3. Optimize P via gradient descent on the continuous relaxation of QAP cost
        4. Extract hard permutation via Hungarian or greedy
        """
        rng = np.random.RandomState(seed)
        p_init = rng.permutation(n).astype(np.int64)
        
        # Create soft permutation matrix
        P = np.zeros((n, n), dtype=np.float64)
        P[np.arange(n), p_init] = 1.0
        
        # Temperature for Gumbel-Softmax-like relaxation
        tau = 1.0
        lr = 0.01
        n_iter = 200
        
        for it in range(n_iter):
            if time.time() > deadline - 30:
                break
                
            # Compute continuous cost: sum_{i,j,k,l} a[i,j] * b[k,l] * P[i,k] * P[j,l]
            # This can be written as: trace(P.T @ A @ P @ B.T) or similar
            # Actually: cost = sum_{i,j} a[i,j] * (P @ B @ P.T)[i,j]
            # Let's compute: M = P @ B @ P.T, then cost = sum(a * M)
            
            M = P @ b @ P.T
            cost_cont = np.sum(a * M)
            
            # Gradient of cost w.r.t. P:
            # d/dP[i,k] = sum_{j,l} a[i,j] * (b[k,l] * P[j,l] + P[j,l] * b[l,k]) * ... 
            # More carefully:
            # cost = sum_{i,j,k,l} a[i,j] * P[i,k] * B[k,l] * P[j,l]
            # d/dP[i,k] = sum_{j,l} a[i,j] * B[k,l] * P[j,l] + sum_{j,l} a[j,i] * P[j,k] * B[l,k]
            #           = (A @ (P @ B))[i,k] + (A.T @ (P @ B.T))[i,k]
            
            PB = P @ b  # (n, n)
            grad = a @ PB + a.T @ (P @ b.T)
            
            # Update P with projection to simplex-like constraint (row sums = 1)
            P -= lr * grad
            P = np.clip(P, 0, None)
            
            # Normalize rows to sum to 1
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P /= row_sums
            
            # Also try to encourage column sums to be close to 1 (soft assignment constraint)
            col_sums = P.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            P /= col_sums
            P /= P.sum(axis=1, keepdims=True)  # renormalize rows again
        
        # Extract hard permutation from P using Hungarian
        row_ind, col_ind = linear_sum_assignment(-P)  # maximize P (minimize -P)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

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

    # 4. Continuous Relaxation Init (NEW - structural change)
    if time.time() < start_time + 25.0:
        p_cont = continuous_relaxation_init(seed=123)
        p_cont, c_cont = vectorized_local_search(p_cont, max_time=start_time + 30.0)
        if c_cont < best_cost:
            best_cost = c_cont
            best_p = p_cont.copy()

    # 5. Random Init
    if time.time() < start_time + 32.0:
        p_rand = np.random.permutation(n).astype(np.int64)
        p_rand, c_rand = vectorized_local_search(p_rand, max_time=start_time + 38.0)
        if c_rand < best_cost:
            best_cost = c_rand
            best_p = p_rand.copy()

    # --- Phase 2: Iterated Local Search with Mixed Strategies ---
    rng = np.random.RandomState(42)
    current_p = best_p.copy()
    current_cost = best_cost

    T = 5000.0
    cooling = 0.95
    min_T = 10.0
    iter_count = 0

    while time.time() < deadline - 10.0:
        iter_count += 1
        
        # Choose perturbation strategy based on iteration
        strategy = iter_count % 3
        
        if strategy == 0:
            # Random swaps
            p = current_p.copy()
            n_swaps = rng.randint(2, 6)
            for _ in range(n_swaps):
                i, j = rng.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
        elif strategy == 1:
            # Continuous relaxation restart from current solution
            if time.time() < deadline - 40.0:
                p = continuous_relaxation_init(seed=iter_count)
                # Blend with current: take best of relaxed and current
                p_relaxed, c_relaxed = vectorized_local_search(p, max_time=deadline - 30.0)
                if c_relaxed < current_cost:
                    p = p_relaxed
                else:
                    p = current_p.copy()
                    # Add small perturbation
                    n_swaps = rng.randint(1, 3)
                    for _ in range(n_swaps):
                        i, j = rng.choice(n, 2, replace=False)
                        p[i], p[j] = p[j], p[i]
            else:
                p = current_p.copy()
                n_swaps = rng.randint(2, 4)
                for _ in range(n_swaps):
                    i, j = rng.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
        else:
            # Block move: swap a contiguous block with another
            p = current_p.copy()
            block_len = rng.randint(3, 8)
            i = rng.randint(0, n - block_len)
            j = rng.randint(0, n - block_len)
            if abs(i - j) < block_len:
                j = (j + block_len + 1) % (n - block_len)
            # Swap blocks
            p[i:i+block_len], p[j:j+block_len] = p[j:j+block_len].copy(), p[i:i+block_len].copy()

        # Local search
        p, cost = vectorized_local_search(p, max_time=deadline - 15.0)

        # Hungarian reoptimization step
        if time.time() < deadline - 20.0 and iter_count % 2 == 0:
            p_hung = hungarian_reopt(p)
            p_hung, cost_hung = vectorized_local_search(p_hung, max_time=deadline - 15.0)
            if cost_hung < cost:
                p = p_hung
                cost = cost_hung

        # Accept or reject based on simulated annealing
        if cost < current_cost:
            current_p = p.copy()
            current_cost = cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            delta_cost = cost - current_cost
            if T > min_T:
                prob = np.exp(-delta_cost / T)
                if rng.random_sample() < prob:
                    current_p = p.copy()
                    current_cost = cost
            T *= cooling

    # Final polish: one more local search pass with exact deltas
    if time.time() < deadline - 3.0:
        best_p, best_cost = vectorized_local_search(best_p, max_time=deadline - 1.0, exact=True)

    return best_p.tolist()
