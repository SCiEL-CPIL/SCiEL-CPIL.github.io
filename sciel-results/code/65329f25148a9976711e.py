"""QAP solver for tho150 using MDS/Procrustes initialization with B_perm-accelerated local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Convert to float64 for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    def compute_cost(p):
        """Compute exact cost."""
        return float(np.sum(a_f * b_f[np.ix_(p, p)]))

    def mds_embedding(D, dim=2):
        """Classical MDS embedding."""
        nn = D.shape[0]
        J = np.eye(nn) - np.ones((nn, nn)) / nn
        B = -0.5 * J @ (D ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals = vals[:dim]
        vecs = vecs[:, :dim]
        vals = np.maximum(vals, 0)
        coords = vecs * np.sqrt(vals)
        return coords

    def procrustes_alignment(X, Y):
        """Align X to Y and solve assignment problem."""
        from scipy.optimize import linear_sum_assignment

        U, S, Vt = np.linalg.svd(Y.T @ X)
        Q = U @ Vt
        X_aligned = X @ Q

        X_norms = np.sum(X_aligned ** 2, axis=1)
        Y_norms = np.sum(Y ** 2, axis=1)
        Cross = X_aligned @ Y.T
        C = X_norms[:, None] + Y_norms[None, :] - 2 * Cross

        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def make_B_perm(p):
        """Build permuted B matrix: B_perm[i,j] = b_f[p[i], p[j]]."""
        return b_f[np.ix_(p, p)].copy()

    def swap_in_B_perm(B_perm, i, j):
        """Swap rows i,j and columns i,j of B_perm in place."""
        B_perm[[i, j], :] = B_perm[[j, i], :]
        B_perm[:, [i, j]] = B_perm[:, [j, i]]

    def efficient_local_search(p, current_cost, end_time, seed=12345, tol=-1e-10):
        """
        First-improvement 2-swap local search using B_perm for O(1) row/col access.
        delta = (a[i,:] - a[j,:]) @ (B_perm[j,:] - B_perm[i,:])
              + (a[:,i] - a[:,j]) @ (B_perm[:,j] - B_perm[:,i])
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        B_perm = make_B_perm(curr_p)

        rng = np.random.RandomState(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        while time.time() < end_time:
            improved = False
            rng.shuffle(pairs)

            for i, j in pairs:
                if time.time() > end_time:
                    break

                drow = B_perm[j, :] - B_perm[i, :]
                dcol = B_perm[:, j] - B_perm[:, i]

                a_diff_row = a_f[i, :] - a_f[j, :]
                a_diff_col = a_f[:, i] - a_f[:, j]

                delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

                if delta < tol:
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    swap_in_B_perm(B_perm, i, j)

                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
                    improved = True

            if not improved:
                break

        return best_p, best_cost

    def sa_polish(curr_p, curr_cost, best_p, best_cost, B_perm, end_time, seed=111):
        """SA polish reusing existing B_perm. Slower cooling for longer effective SA."""
        best_p_sa = best_p.copy()
        best_cost_sa = best_cost
        rng_sa = np.random.RandomState(seed)
        T = 500.0
        cooling = 0.9995  # Slower cooling: ~1400 steps to reach T=1, allowing more exploration
        while time.time() < end_time and T > 0.5:
            i, j = rng_sa.choice(n, 2, replace=False)

            drow = B_perm[j, :] - B_perm[i, :]
            dcol = B_perm[:, j] - B_perm[:, i]
            a_diff_row = a_f[i, :] - a_f[j, :]
            a_diff_col = a_f[:, i] - a_f[:, j]
            delta = np.dot(a_diff_row, drow) + np.dot(a_diff_col, dcol)

            if delta < 0 or rng_sa.random_sample() < np.exp(-delta / T):
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                swap_in_B_perm(B_perm, i, j)
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p_sa = curr_p.copy()
            T *= cooling
        return curr_p, curr_cost, best_p_sa, best_cost_sa

    # 1. MDS Initialization (multi-dimension)
    a_sym = (a_f + a_f.T) / 2.0
    b_sym = (b_f + b_f.T) / 2.0

    best_p_init = None
    best_cost_init = np.inf

    for dim in [2, 3, 5]:
        if time.time() > deadline - 50.0:
            break
        X = mds_embedding(a_sym, dim)
        Y = mds_embedding(b_sym, dim)
        p_try = procrustes_alignment(X, Y)
        cost_try = compute_cost(p_try)
        if cost_try < best_cost_init:
            best_cost_init = cost_try
            best_p_init = p_try

    # 2. Rank Matching Initialization
    a_row_sums = a_f.sum(axis=1)
    b_row_sums = b_f.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = compute_cost(p_rank)
    if cost_rank < best_cost_init:
        best_cost_init = cost_rank
        best_p_init = p_rank

    # 3. Random Initialization
    rng_init = np.random.RandomState(42)
    p_rand = rng_init.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost_init:
        best_cost_init = cost_rand
        best_p_init = p_rand

    # 4. Main Local Search Polish
    end_time_lk = deadline - 30.0
    best_p, best_cost = efficient_local_search(best_p_init, best_cost_init, end_time_lk, seed=12345)

    # 5. Build B_perm once and reuse for SA
    B_perm_main = make_B_perm(best_p)

    # 6. SA Polish with longer effective runtime
    if time.time() < deadline - 15.0:
        curr_p = best_p.copy()
        curr_cost = best_cost
        end_time_sa = deadline - 10.0
        curr_p, curr_cost, best_p_sa, best_cost_sa = sa_polish(
            curr_p, curr_cost, best_p, best_cost, B_perm_main, end_time_sa, seed=111
        )
        if best_cost_sa < best_cost:
            best_p = best_p_sa
            best_cost = best_cost_sa
            B_perm_main = make_B_perm(best_p)

    # 7. Second SA restart with different seed if time remains
    if time.time() < deadline - 12.0:
        curr_p2 = best_p.copy()
        curr_cost2 = best_cost
        # Small perturbation before second SA
        rng_perturb = np.random.RandomState(789)
        for _ in range(3):
            i, j = rng_perturb.choice(n, 2, replace=False)
            curr_p2[i], curr_p2[j] = curr_p2[j], curr_p2[i]
            swap_in_B_perm(B_perm_main, i, j)
        # Recompute cost after perturbation
        curr_cost2 = compute_cost(curr_p2)
        end_time_sa2 = deadline - 5.0
        curr_p2, curr_cost2, best_p_sa2, best_cost_sa2 = sa_polish(
            curr_p2, curr_cost2, best_p, best_cost, B_perm_main, end_time_sa2, seed=456
        )
        if best_cost_sa2 < best_cost:
            best_p = best_p_sa2
            best_cost = best_cost_sa2

    # 8. Final high-precision polish pass
    if time.time() < deadline - 2.0:
        end_time_final = deadline - 0.5
        p_final, cost_final = efficient_local_search(best_p, best_cost, end_time_final, seed=999, tol=-1e-12)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final

    return best_p.tolist()
# EVOLVE-BLOCK-END
