"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)

def _2opt_pass(a, b, p, max_time_remaining):
    """Perform 2-opt local search. Returns improved p and cost."""
    n = len(p)
    cost = _compute_cost(a, b, p)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            best_j = -1
            best_delta = 1e-10
            pi = p[i]
            for j in range(n):
                if i == j:
                    continue
                pj = p[j]
                # Compute delta for swapping p[i] and p[j]
                # Delta = sum_{k != i,j} a[i,k]*(b[pj,pk]-b[pi,pk]) + sum_{k != i,j} a[j,k]*(b[pi,pk]-b[pj,pk]) + a[i,j]*(b[pi,pi]+b[pj,pj]-2*b[pi,pj])
                # Vectorized:
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                term_i = np.sum(a[i, mask] * (b[pj, p[mask]] - b[pi, p[mask]]))
                term_j = np.sum(a[j, mask] * (b[pi, p[mask]] - b[pj, p[mask]]))
                term_ij = a[i, j] * (b[pi, pi] + b[pj, pj] - 2 * b[pi, pj])
                delta = term_i + term_j + term_ij
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            if best_j >= 0 and best_delta < 0:
                p[i], p[best_j] = p[best_j], p[i]
                cost += best_delta
                improved = True
                # Check time
                if time.time() - start_time > max_time_remaining - 1.0:
                    return p, cost
        # Check time at end of pass
        if time.time() - start_time > max_time_remaining - 1.0:
            break
    return p, cost

def _perturb(p, n, strength=5):
    """Randomly perturb the permutation by swapping strength pairs."""
    p = p.copy()
    for _ in range(strength):
        i, j = np.random.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    total_budget = 110.0  # safety margin before 120s limit
    start_time = time.time()
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_best = np.empty(n, dtype=np.int64)
    p_best[heavy] = central
    best_cost = _compute_cost(a, b, p_best)
    
    remaining = total_budget - (time.time() - start_time)
    if remaining < 10:
        return p_best.tolist()
    
    # 2-opt local search from initial solution
    p, cost = _2opt_pass(a, b, p_best.copy(), remaining)
    if cost < best_cost:
        best_cost = cost
        p_best = p.copy()
    
    remaining = total_budget - (time.time() - start_time)
    if remaining < 5:
        return p_best.tolist()
    
    # Iterated local search with perturbations
    rng = np.random.default_rng(42)
    iter_count = 0
    while True:
        remaining = total_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Perturb the best solution
        p_current = _perturb(p_best, n, strength=5)
        
        # Run 2-opt from perturbed solution
        p, cost = _2opt_pass(a, b, p_current, remaining - 2)
        if cost < best_cost:
            best_cost = cost
            p_best = p.copy()
            iter_count = 0
        else:
            iter_count += 1
            if iter_count > 50:
                break
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
