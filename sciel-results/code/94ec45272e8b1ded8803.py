"""QAP tai256c solver: iterated local search with vectorized 2-opt, Hungarian re-optimization, and MCMC diversification."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0

    # Precompute symmetric A for vectorized delta calculation
    A_sym = a + a.T

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20, exact=False):
        improved = True
        passes = 0
        p = p.copy()
        
        D = b[np.ix_(p, p)].copy()
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                    
                if exact:
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    if best_j >= 0:
                        j = best_j
                        min_delta = best_delta
                    else:
                        continue
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
                    
                    cost += min_delta
                    improved = True
        
        return p, cost

    def hungarian_reopt(p):
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
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def mcmc_perturb(p, cost, n_steps=50, temperature=1000.0):
        p = p.copy()
        current_cost = cost
        best_p = p.copy()
        best_cost = cost
        
        for _ in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            block_size = np.random.randint(2, 11)
            if block_size >= n:
                block_size = n - 1
            
            start = np.random.randint(0, n - block_size)
            end = start + block_size
            
            new_p = p.copy()
            new_p[start:end+1] = p[start:end+1][::-1]
            
            new_cost = compute_cost(new_p)
            
            delta = new_cost - current_cost
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                p = new_p
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        return best_p, best_cost

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

    # Phase 3: Iterated Local Search
    iteration = 0
    consecutive_no_improve = 0
    base_perturb = 3
    mcmc_temperature = 5000.0

    while time.time() < deadline - 10:
        iteration += 1

        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
            mcmc_temperature = 5000.0 + consecutive_no_improve * 1000.0
        else:
            perturb_k = base_perturb
            mcmc_temperature = 5000.0

        p = best_p.copy()
        
        n_swaps = np.random.randint(2, perturb_k + 1)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
        
        p, cost = mcmc_perturb(p, compute_cost(p), n_steps=20, temperature=mcmc_temperature)

        p, cost = local_search(p, cost)

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
        remaining = deadline - time.time() - 5
        p, cost = local_search(p, best_cost, max_passes=50, exact=True)
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

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
