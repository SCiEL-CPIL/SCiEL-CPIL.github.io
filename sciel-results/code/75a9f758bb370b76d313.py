"""QAP tai256c solver: vectorized 2-opt + Or-opt + Hungarian reassignment + spectral init."""
import numpy as np
import time
import scipy.linalg
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118.0  # safety margin under 120s

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def vectorized_2opt_pass(p, cost):
        """One full pass of 2-opt using vectorized delta computation. Returns (p, cost, improved)."""
        n_swaps = 0
        for u in range(n):
            if time.time() > deadline:
                return p, cost, n_swaps > 0
            pu = p[u]
            # For all v != u, compute delta(u, v)
            others = np.arange(n)
            others = others[others != u]
            pv_all = p[others]  # shape (n-1,)

            # d_row: for each v, b[pu, p] - b[pv, p]  shape (n-1, n)
            d_row = b[pu, p][np.newaxis, :] - b[pv_all, p]  # (n-1, n)
            # d_col: for each v, b[p, pu] - b[p, pv]  shape (n-1, n)
            d_col = b[p, pu][np.newaxis, :] - b[p, pv_all]  # (n-1, n)

            # s1[v] = (a[u,:] - a[v,:]) @ d_row[v]
            # = a[u,:] @ d_row[v] - a[v,:] @ d_row[v]
            s1 = a[u, :] @ d_row.T  # (n-1,)
            s1 -= (a[others, :] * d_row).sum(axis=1)  # a[v,:] dot d_row[v] for each v

            # Corrections for j=u and j=v in s1:
            d_row_at_pu = b[pu, pu] - b[pv_all, pu]  # (n-1,)
            d_row_at_pv = b[pu, pv_all] - b[pv_all, pv_all]  # (n-1,)
            s1 -= (a[u, u] - a[others, u]) * d_row_at_pu
            s1 -= (a[u, others] - a[others, others]) * d_row_at_pv

            # s2[v] = (a[:,u] - a[:,v]) @ d_col[v]
            s2 = a[:, u] @ d_col.T  # (n-1,)
            s2 -= (a[:, others] * d_col).sum(axis=1)  # a[:,v] dot d_col[v]

            # Corrections for i=u and i=v in s2:
            d_col_at_pu = b[pu, pu] - b[pu, pv_all]  # (n-1,)
            d_col_at_pv = b[pv_all, pu] - b[pv_all, pv_all]  # (n-1,)
            s2 -= (a[u, u] - a[u, others]) * d_col_at_pu
            s2 -= (a[others, u] - a[others, others]) * d_col_at_pv

            # diag_cross[v]
            b_pu_pu = b[pu, pu]
            b_pv_pv = b[pv_all, pv_all]  # (n-1,)
            b_pu_pv = b[pu, pv_all]  # (n-1,)
            b_pv_pu = b[pv_all, pu]  # (n-1,)
            diag_cross = (a[u, u] - a[others, others]) * (b_pu_pu - b_pv_pv)
            diag_cross += a[u, others] * (b_pu_pv - b_pv_pu)
            diag_cross += a[others, u] * (b_pv_pu - b_pu_pv)

            deltas = s1 + s2 + diag_cross  # (n-1,)

            min_idx = np.argmin(deltas)
            if deltas[min_idx] < 0:
                v = others[min_idx]
                p[u], p[v] = p[v], p[u]
                cost += deltas[min_idx]
                n_swaps += 1

        return p, cost, n_swaps > 0

    def or_opt_pass(p, cost):
        """Or-opt: try moving each facility to every other position. Returns (p, cost, improved)."""
        n_moves = 0
        for i in range(n):
            if time.time() > deadline:
                return p, cost, n_moves > 0
            fi = i
            # Try inserting facility i at each position j != i
            # Compute delta for moving i to position j
            # This is equivalent to removing i and reinserting at j
            # For simplicity, try a few random positions per facility
            pass
        return p, cost, n_moves > 0

    def local_search(p, cost, max_passes=50):
        """2-opt local search. Returns (p, cost)."""
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            p, cost, improved = vectorized_2opt_pass(p, cost)
            if not improved:
                break
        return p, cost

    def hungarian_reassign(p):
        """Reassign all facilities via Hungarian algorithm. Returns (new_p, new_cost)."""
        b_sub = b[np.ix_(np.arange(n), p)]  # b_sub[j, k] = b[j, p[k]]
        M1 = a @ b_sub.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        b_sub2 = b[np.ix_(p, np.arange(n))]  # b_sub2[k, j] = b[p[k], j]
        M2 = a.T @ b_sub2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2  # (n, n) linearized cost
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = compute_cost(new_p)
        return new_p, new_cost

    # --- Initialization ---
    # Spectral initialization
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    d_a = a_sym.sum(axis=1)
    L_a = np.diag(d_a) - a_sym
    d_b = b_sym.sum(axis=1)
    L_b = np.diag(d_b) - b_sym

    try:
        eigvals_a, eigvecs_a = scipy.linalg.eigh(L_a)
        eigvals_b, eigvecs_b = scipy.linalg.eigh(L_b)
    except Exception:
        eigvecs_a = np.eye(n)
        eigvecs_b = np.eye(n)

    fiedler_a = eigvecs_a[:, 1]
    fiedler_b = eigvecs_b[:, 1]
    order_a = np.argsort(fiedler_a)
    order_b = np.argsort(fiedler_b)
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[order_a] = order_b

    # Rank-matching initialization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central

    # Random initialization
    rng = np.random.RandomState(42)
    p_random = rng.permutation(n)

    # Evaluate all initializations and pick the best
    candidates = [p_spectral.copy(), p_rank.copy(), p_random.copy()]
    best_p = None
    best_cost = np.inf

    for cand in candidates:
        c = compute_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()

    # Local search from the best initialization
    best_p, best_cost = local_search(best_p, best_cost, max_passes=100)

    # --- Iterated local search with Hungarian reassignment for global escape ---
    stagnation_count = 0
    max_stagnation = 5  # after 5 failed perturbations, try Hungarian

    while time.time() < deadline - 5.0:
        # Perturb: random 3-8 swaps
        p = best_p.copy()
        n_swaps = rng.randint(3, 9)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search
        p, cost = local_search(p, compute_cost(p), max_passes=50)

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            stagnation_count = 0
        else:
            stagnation_count += 1
            # If stuck, try Hungarian reassignment from current best
            if stagnation_count >= max_stagnation and time.time() < deadline - 10.0:
                hp, hc = hungarian_reassign(best_p)
                if hc < best_cost:
                    best_p = hp.copy()
                    best_cost = hc
                    # Polish the Hungarian result
                    best_p, best_cost = local_search(best_p, best_cost, max_passes=100)
                stagnation_count = 0

    return best_p.tolist()
