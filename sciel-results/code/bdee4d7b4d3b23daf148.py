"""QAP tai256c solver: iterated local search with vectorized 2-opt, Hungarian re-optimization, MCMC diversification, and adaptive perturbation."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    # Precompute symmetric A for vectorized delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a).copy()

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def local_search(p, cost, max_passes=30, exact=False):
        """Steepest Descent Local Search with vectorized delta calculation."""
        p = p.copy()
        D = b[np.ix_(p, p)].copy()
        improved = True
        passes = 0

        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost

                js = np.arange(i + 1, n)

                if exact:
                    # Scalar exact delta for final polish
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        pu = p[i]
                        pv = p[j]
                        d_uv = b[pv, :] - b[pu, :]
                        d_vu = b[:, pv] - b[:, pu]
                        s1 = np.dot(a[i, :] - a[j, :], d_uv[p])
                        s2 = np.dot(a[:, i] - a[:, j], d_vu[p])
                        s1 -= (a[i, i] - a[j, i]) * d_uv[pu]
                        s1 -= (a[i, j] - a[j, j]) * d_uv[pv]
                        s2 -= (a[i, i] - a[i, j]) * d_vu[pu]
                        s2 -= (a[j, i] - a[j, j]) * d_vu[pv]
                        cross = (a[i, j] - a[j, i]) * (b[pv, pu] - b[pu, pv])
                        delta = s1 + s2 + cross
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    if best_j >= 0:
                        j = best_j
                        min_delta = best_delta
                    else:
                        continue
                else:
                    # Vectorized delta calculation
                    diff_A = A_sym[i] - A_sym[js]
                    diff_D = D[js] - D[i]
                    S = np.sum(diff_A * diff_D, axis=1)

                    # Correction terms
                    # k=i term: (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
                    coeff_i = A_sym[i, i] - A_sym[js, i]
                    term_i = coeff_i * (D[js, i] - D[i, i])

                    # k=j term: (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
                    coeff_j = A_sym[i, js] - A_diag[js]
                    term_j = coeff_j * (D[js, js] - D[i, js])

                    # Diagonal swap term
                    diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])

                    # Cross term
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

    def mcmc_perturb(p, current_cost, k_blocks=3):
        """MCMC-based perturbation using block reversals with Metropolis acceptance."""
        p = p.copy()
        cost = current_cost
        for _ in range(k_blocks):
            # Pick a random sub-sequence to reverse
            start = np.random.randint(0, n - 1)
            max_len = n - start
            if max_len < 2:
                continue
            high = min(20, max_len)
            if high <= 2:
                continue
            length = np.random.randint(2, high)
            end = start + length
            # Reverse the block
            p[start:end] = p[start:end][::-1]
            new_cost = compute_cost(p)
            # Metropolis acceptance
            if new_cost < cost or np.random.random() < np.exp(-max(0, new_cost - cost) / 100000.0):
                cost = new_cost
            else:
                # Revert
                p[start:end] = p[start:end][::-1]
        return p, cost

    # Phase 1: Hungarian Initialization
    p_hung = hungarian_init()
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)
    current_p, current_cost = local_search(best_p, best_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Phase 2: Rank matching initialization
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_rank = p_rank.astype(np.int64)
    p_rank_ls, cost_rank = local_search(p_rank, compute_cost(p_rank))
    if cost_rank < best_cost:
        best_p = p_rank_ls.copy()
        best_cost = cost_rank

    # Phase 3: Iterated Local Search with MCMC diversification
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3

    while time.time() < deadline - 10:
        iteration += 1

        # Adaptive perturbation strategy
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb

        # Alternate between random swaps and MCMC block reversals
        if iteration % 2 == 0:
            # MCMC-based perturbation
            p = best_p.copy()
            p, cost = mcmc_perturb(p, best_cost, k_blocks=perturb_k)
        else:
            # Random swap perturbation
            p = best_p.copy()
            n_swaps = np.random.randint(2, perturb_k + 1)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            cost = compute_cost(p)

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

    # Final high-precision polish pass
    if time.time() < deadline - 5:
        p = best_p.copy()
        p, cost = local_search(p, best_cost, max_passes=50, exact=True)
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

        # Try a few more local searches from slightly perturbed best
        remaining = deadline - time.time() - 5
        if remaining > 5:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                p = best_p.copy()
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                p, cost = local_search(p, compute_cost(p), max_passes=30, exact=True)
                if cost < best_cost:
                    best_p = p.copy()
                    best_cost = cost
                if time.time() > deadline - 5:
                    break

    return best_p.tolist()
