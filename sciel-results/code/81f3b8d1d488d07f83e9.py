# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = 118.0  # safety margin before 120s/150s kill

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def get_delta(p, i, j):
        """Delta cost for swapping facilities i and j in permutation p.
        Assumes symmetric a and b for simplification, but works generally if we sum all terms.
        For QAPLIB instances, a and b are symmetric.
        """
        u = p[i]
        v = p[j]
        
        # Diagonal terms
        diag_delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + 2 * a[i, j] * (b[v, u] - b[u, v])
        
        # Off-diagonal terms
        # Sum over k != i, j of (a[i,k] - a[j,k]) * (b[v,k] - b[u,k])
        # Since we assume symmetry, a[k,i] = a[i,k] and b[k,v] = b[v,k].
        # The total change is 2 * sum_{k != i,j} (a[i,k] - a[j,k]) * (b[v,k] - b[u,k])
        
        # We compute the sum over all k, then subtract k=i and k=j
        # Sum_all = sum_k (a[i,k] - a[j,k]) * (b[v,k] - b[u,k])
        
        # Efficient calculation:
        # b_v_minus_b_u = b[v, :] - b[u, :]
        # diff_a = a[i, :] - a[j, :]
        # sum_all = dot(diff_a, b_v_minus_b_u)
        
        b_v_minus_b_u = b[v, :] - b[u, :]
        diff_a = a[i, :] - a[j, :]
        sum_all = np.dot(diff_a, b_v_minus_b_u)
        
        # Subtract k=i term: (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # Subtract k=j term: (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        off_diag_delta = 2 * (sum_all - term_i - term_j)
        
        return diag_delta + off_diag_delta

    def local_search(p, current_cost):
        """First-improvement 2-opt local search."""
        p = p.copy()
        while True:
            if time.time() - t_start > deadline:
                break
            moved = False
            for i in range(n):
                if time.time() - t_start > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() - t_start > deadline:
                        break
                    delta = get_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        moved = True
                        break  # Restart scan
                if moved:
                    break
            if not moved:
                break
        return p, current_cost

    # Spectral initialization
    A_sym = (a + a.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    idx_a = np.argmax(eigvals)
    v_a = eigvecs[:, idx_a]

    B_sym = (b + b.T) / 2.0
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    idx_b = np.argmax(eigvals_b)
    v_b = eigvecs_b[:, idx_b]

    fac_order = np.argsort(-v_a, kind="stable")
    loc_order = np.argsort(v_b, kind="stable")
    p_spectral = np.empty(n, dtype=np.int64)
    p_spectral[fac_order] = loc_order

    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central

    cost_spectral = compute_cost(p_spectral)
    cost_rank = compute_cost(p_rank)
    if cost_rank < cost_spectral:
        p = p_rank.copy()
        best_cost = cost_rank
    else:
        p = p_spectral.copy()
        best_cost = cost_spectral

    # Run local search on initial solution
    p, best_cost = local_search(p, best_cost)
    best_p = p.copy()

    # Iterated Local Search with adaptive perturbation and reset-to-best
    rng = np.random.RandomState(42)
    no_improve_count = 0

    while time.time() - t_start < deadline - 5.0:
        # Adaptive perturbation strength
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 12

        p_pert = best_p.copy()
        current_cost = best_cost
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            delta = get_delta(p_pert, i, j)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            current_cost += delta

        # Local search from perturbed solution
        p_new, cost_new = local_search(p_pert, current_cost)

        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_p.tolist()
# EVOLVE-BLOCK-END
