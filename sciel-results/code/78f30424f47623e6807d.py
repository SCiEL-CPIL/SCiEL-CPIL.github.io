"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline):
    """First-improvement 2-opt local search with time guard."""
    n = len(p)
    cost = _cost(a, b, p)
    improved = True
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Randomize the order of j for this i to avoid cycling
            js = np.random.permutation(range(i + 1, n))
            for j in js:
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
    return p.copy(), cost


def _best_improvement_2opt(a, b, p, deadline):
    """Repeated best-improvement 2-opt until convergence or deadline."""
    n = len(p)
    cost = _cost(a, b, p)
    
    while time.time() < deadline:
        best_delta = 0
        best_move = None
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
        if best_move is None:
            break
        i, j = best_move
        p[i], p[j] = p[j], p[i]
        cost += best_delta
    
    return p.copy(), cost


def _polish_pass(a, b, p, deadline):
    """
    High-precision polish: best-improvement 2-opt followed by limited 3-opt,
    then final 2-opt cleanup.
    """
    n = len(p)
    
    # Phase 1: Best-improvement 2-opt
    p, cost = _best_improvement_2opt(a, b, p.copy(), deadline)
    
    # Phase 2: Limited 3-opt (only if time remains)
    remaining = deadline - time.time()
    if remaining > 2.0:
        cost_3 = cost
        improved_3 = True
        max_3opt_rounds = 3
        for _round in range(max_3opt_rounds):
            if time.time() > deadline - 1.0:
                break
            found = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 1.0:
                        break
                    for k in range(j + 1, n):
                        if time.time() > deadline - 1.0:
                            break
                        old_pi, old_pj, old_pk = p[i], p[j], p[k]
                        
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        mask[j] = False
                        mask[k] = False
                        others = p[mask]
                        
                        old_with_others = (
                            np.sum(a[i, mask] * b[old_pi, others]) +
                            np.sum(a[j, mask] * b[old_pj, others]) +
                            np.sum(a[k, mask] * b[old_pk, others]) +
                            np.sum(a[mask, i] * b[others, old_pi]) +
                            np.sum(a[mask, j] * b[others, old_pj]) +
                            np.sum(a[mask, k] * b[others, old_pk])
                        )
                        
                        new_with_others = (
                            np.sum(a[i, mask] * b[old_pj, others]) +
                            np.sum(a[j, mask] * b[old_pk, others]) +
                            np.sum(a[k, mask] * b[old_pi, others]) +
                            np.sum(a[mask, i] * b[others, old_pj]) +
                            np.sum(a[mask, j] * b[others, old_pk]) +
                            np.sum(a[mask, k] * b[others, old_pi])
                        )
                        
                        old_internal = (
                            a[i, j] * b[old_pi, old_pj] + a[j, i] * b[old_pj, old_pi] +
                            a[j, k] * b[old_pj, old_pk] + a[k, j] * b[old_pk, old_pj] +
                            a[k, i] * b[old_pk, old_pi] + a[i, k] * b[old_pi, old_pk] +
                            a[i, i] * b[old_pi, old_pi] + a[j, j] * b[old_pj, old_pj] +
                            a[k, k] * b[old_pk, old_pk]
                        )
                        
                        new_internal = (
                            a[i, j] * b[old_pj, old_pk] + a[j, i] * b[old_pk, old_pj] +
                            a[j, k] * b[old_pk, old_pi] + a[k, j] * b[old_pi, old_pk] +
                            a[k, i] * b[old_pi, old_pj] + a[i, k] * b[old_pj, old_pi] +
                            a[i, i] * b[old_pj, old_pj] + a[j, j] * b[old_pk, old_pk] +
                            a[k, k] * b[old_pi, old_pi]
                        )
                        
                        delta_3 = (new_with_others - old_with_others) + (new_internal - old_internal)
                        
                        if delta_3 < 0:
                            p[i], p[j], p[k] = old_pj, old_pk, old_pi
                            cost_3 += int(delta_3)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        if cost_3 < cost:
            cost = cost_3
            improved_3 = True
    
    # Phase 3: Final 2-opt cleanup
    if time.time() < deadline - 0.5:
        p, cost = _best_improvement_2opt(a, b, p.copy(), deadline)
    
    return p.copy(), cost


def _spectral_init(a, b, k=None, rng=None):
    """Spectral initialization: embed facilities and locations, match by nearest neighbor."""
    n = a.shape[0]
    if k is None:
        k = min(10, n)
    if rng is None:
        rng = np.random.default_rng(42)
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
        
        idx_a = np.argsort(evals_a)
        fac_emb = evecs_a[:, idx_a[:k]]
        
        idx_b = np.argsort(-evals_b)
        loc_emb = evecs_b[:, idx_b[:k]]
        
        fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
        fac_norm[fac_norm == 0] = 1
        fac_emb = fac_emb / fac_norm
        
        loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
        loc_norm[loc_norm == 0] = 1
        loc_emb = loc_emb / loc_norm
        
        # Add small random noise for diversity
        noise_scale = rng.uniform(0.01, 0.05)
        fac_emb = fac_emb + noise_scale * rng.standard_normal(fac_emb.shape)
        loc_emb = loc_emb + noise_scale * rng.standard_normal(loc_emb.shape)
        
        cost_matrix = -fac_emb @ loc_emb.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except:
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _simulated_annealing(a, b, p, deadline, rng=None):
    """Batched simulated annealing with slower cooling for better exploration."""
    if rng is None:
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    
    n = len(p)
    cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    T_initial = max(1.0, abs(cost) * 0.01)
    T_final = max(0.05, abs(cost) * 1e-7)
    cooling_rate = 0.999  # Slower cooling
    batch_size = 50  # Process moves in batches to reduce time-check overhead
    
    T = T_initial
    no_improve = 0
    max_no_improve = 8000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Reheat
            T = max(T_initial * 0.1, T_final * 10)
            no_improve = 0
            if T < T_final:
                break
        
        # Batch process: generate batch_size candidate moves at once
        batch = min(batch_size, max(1, int((deadline - time.time()) * 200)))
        accepted = 0
        for _ in range(batch):
            if time.time() > deadline:
                break
            
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            
            delta = _delta_swap(a, b, p, i, j)
            
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                cost += delta
                accepted += 1
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if T > 1e-10:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        accepted += 1
                no_improve += 1
        
        T *= cooling_rate ** batch
        if T < T_final:
            T = T_final
    
    return best_p, best_cost


def _tabu_search(a, b, p, deadline, tabu_tenure=10):
    """Vectorized simulated annealing to escape local minima (replaces O(n^3) tabu)."""
    return _simulated_annealing(a, b, p, deadline)


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin

    best_p = None
    best_cost = np.inf
    rng_main = np.random.default_rng(42)

    def _update_best(p, cost):
        nonlocal best_p, best_cost
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    # 1. Multi-Start Spectral Initialization (diverse starting points)
    n_starts = 4
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        p_spec = _spectral_init(a, b, k=min(10, n), rng=rng_main)
        cost_spec = _cost(a, b, p_spec)
        _update_best(p_spec, cost_spec)
        
        if time.time() < deadline - 25:
            p_ls, cost_ls = _local_search(a, b, p_spec.copy(), deadline - 25)
            _update_best(p_ls, cost_ls)

    # 2. Linear Assignment Initialization (Surrogate)
    if time.time() < deadline - 20:
        surrogate = a @ b + a.T @ b
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        cost_lap = _cost(a, b, p_lap)
        _update_best(p_lap, cost_lap)

        if time.time() < deadline - 15:
            p_ls, cost_ls = _local_search(a, b, p_lap.copy(), deadline - 15)
            _update_best(p_ls, cost_ls)

    # 3. Simulated Annealing from best solution (batched, slower cooling)
    if best_p is not None and time.time() < deadline - 25:
        p_sa, cost_sa = _simulated_annealing(a, b, best_p.copy(), deadline - 25, rng=rng_main)
        _update_best(p_sa, cost_sa)
        
        # Local search after SA
        if time.time() < deadline - 15:
            p_ls, cost_ls = _local_search(a, b, p_sa.copy(), deadline - 15)
            _update_best(p_ls, cost_ls)

    # 4. Iterated Local Search (ILS) with surrogate-guided perturbation
    if best_p is not None and time.time() < deadline - 20:
        current_p = best_p.copy()
        current_cost = best_cost
        
        surrogate = a @ b + a.T @ b
        
        while time.time() < deadline - 20:
            n_perturb = rng_main.integers(5, max(6, n // 5))
            subset = rng_main.choice(n, size=min(n_perturb, n), replace=False)
            
            p_perturbed = current_p.copy()
            old_locs = p_perturbed[subset]
            
            fixed_locs = set(p_perturbed.tolist()) - set(old_locs.tolist())
            available_locs = [l for l in range(n) if l not in fixed_locs]
            
            if len(available_locs) > 0 and len(subset) > 0:
                sub_cost = np.zeros((len(subset), len(available_locs)))
                for ii, fac in enumerate(subset):
                    for jj, loc in enumerate(available_locs):
                        sub_cost[ii, jj] = surrogate[fac, loc]
                
                row_ind, col_ind = linear_sum_assignment(sub_cost)
                for ii, jj in zip(row_ind, col_ind):
                    p_perturbed[subset[ii]] = available_locs[jj]
            
            p_ls, cost_ls = _local_search(a, b, p_perturbed, deadline - 20)
            
            if cost_ls < current_cost:
                current_p = p_ls
                current_cost = cost_ls
                _update_best(current_p, current_cost)

    # 5. Final high-precision polish pass
    if best_p is not None and time.time() < deadline - 0.5:
        p_polish, cost_polish = _polish_pass(a, b, best_p.copy(), deadline - 0.5)
        _update_best(p_polish, cost_polish)

    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
