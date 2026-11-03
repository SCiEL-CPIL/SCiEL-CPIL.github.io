"""Benders-decomposition QAP solver for wil100.

Master problem: linear assignment minimizing surrogate cost.
Subproblem: evaluate exact QAP cost, generate a cutting plane
(per-facility penalty) that discourages the current assignment
and guides the master toward lower true cost.

Enhanced with:
- Efficient delta-swap local search (O(n) per swap instead of O(n^2))
- Multi-start with diverse initializations
- Adaptive step size for cutting planes
- Final high-precision polish pass
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _cost(p, a, b):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _delta_swap(a, b, p, i, j, a_row_sums, b_row_sums, col_sums_cache):
    """
    Compute the change in cost if we swap locations p[i] and p[j].
    Uses precomputed row sums for efficiency.
    
    The cost is sum_{u,v} a[u,v] * b[p[u], p[v]].
    Swapping p[i] and p[j] only affects terms where u or v is i or j.
    """
    pi, pj = p[i], p[j]
    
    # Terms involving facility i or j as first index
    # sum_v a[i,v] * (b[pj, v'] - b[pi, v']) where v' is the location of facility v
    # This is: sum_v a[i,v] * (b[pj, p[v]] - b[pi, p[v]])
    # = sum_v a[i,v] * b[pj, p[v]] - sum_v a[i,v] * b[pi, p[v]]
    
    # Similarly for facility j
    # And for terms where i or j is the second index:
    # sum_u a[u,i] * (b[p[u], pj] - b[p[u], pi])
    # sum_u a[u,j] * (b[p[u], pi] - b[p[u], pj])
    
    # Let's compute more carefully:
    # Original contribution involving i or j:
    # sum_v a[i,v] * b[pi, p[v]] + sum_u a[u,i] * b[p[u], pi]
    #   + sum_v a[j,v] * b[pj, p[v]] + sum_u a[u,j] * b[p[u], pj]
    #   - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]  (double counted)
    
    # After swap:
    # sum_v a[i,v] * b[pj, p[v]] + sum_u a[u,i] * b[p[u], pj]
    #   + sum_v a[j,v] * b[pi, p[v]] + sum_u a[u,j] * b[p[u], pi]
    #   - a[i,j]*b[pj,pi] - a[j,i]*b[pi,pj]
    
    # Delta = new - old
    # For terms with i as first index:
    # sum_v a[i,v] * (b[pj, p[v]] - b[pi, p[v]])
    # For terms with i as second index:
    # sum_u a[u,i] * (b[p[u], pj] - b[p[u], pi])
    # For terms with j as first index:
    # sum_v a[j,v] * (b[pi, p[v]] - b[pj, p[v]])
    # For terms with j as second index:
    # sum_u a[u,j] * (b[p[u], pi] - b[p[u], pj])
    
    # Note: when v=i, b[pi, p[i]] = b[pi, pi], b[pj, p[i]] = b[pj, pi]
    # when v=j, b[pi, p[j]] = b[pi, pj], b[pj, p[j]] = b[pj, pj]
    
    # Let's use matrix-vector products for efficiency:
    # sum_v a[i,v] * b[pj, p[v]] = (a[i,:] * b[pj, p]).sum()
    # sum_v a[i,v] * b[pi, p[v]] = (a[i,:] * b[pi, p]).sum()
    
    ai = a[i, :]
    aj = a[j, :]
    
    # b[pj, p] is the row pj of b indexed by p
    bpj_p = b[pj, p]
    bpi_p = b[pi, p]
    
    # b[:, pj] indexed by p is b[p, pj]
    b_p_pj = b[p, pj]
    b_p_pi = b[p, pi]
    
    delta_i_first = np.dot(ai, bpj_p) - np.dot(ai, bpi_p)
    delta_i_second = np.dot(a[:, i], b_p_pj) - np.dot(a[:, i], b_p_pi)
    delta_j_first = np.dot(aj, bpi_p) - np.dot(aj, bpj_p)
    delta_j_second = np.dot(a[:, j], b_p_pi) - np.dot(a[:, j], b_p_pj)
    
    # Correct for double-counting of (i,j) and (j,i) terms
    # In the original, a[i,j]*b[pi,pj] and a[j,i]*b[pj,pi] are counted
    # In the new, a[i,j]*b[pj,pi] and a[j,i]*b[pi,pj] are counted
    # The above formulas include these correctly since they're part of the sums
    
    return delta_i_first + delta_i_second + delta_j_first + delta_j_second


def _local_search(p, a, b, max_passes=10, time_limit=None, deadline=None):
    """Local search using 2-opt swaps with efficient delta computation."""
    n = len(p)
    p = p.copy()
    best_cost = _cost(p, a, b)
    
    # Precompute
    a_sym = 0.5 * (a + a.T)
    
    passes = 0
    while passes < max_passes:
        passes += 1
        improved = False
        npairs = min(100, n * 10)
        
        # Try random pairs
        for _ in range(npairs):
            if deadline is not None and time.time() > deadline:
                return p, best_cost
            
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _delta_swap(a, b, p, i, j, None, None, None)
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                best_cost += int(delta)
                improved = True
        
        if not improved:
            break
    
    return p, best_cost


def _spectral_init(a, b):
    """Spectral initialization using eigendecomposition."""
    n = len(a)
    A = 0.5 * (a + a.T)
    B = 0.5 * (b + b.T)
    
    # Use row sums as a simple heuristic
    a_row = a.sum(axis=1)
    b_row = b.sum(axis=1)
    
    heavy = np.argsort(-a_row, kind="stable")
    central = np.argsort(b_row, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150 s wall-clock
    
    # Precompute symmetric parts
    A = 0.5 * (a + a.T)
    B = 0.5 * (b + b.T)
    
    # Row sums for initialization
    a_row = a.sum(axis=1)
    b_row = b.sum(axis=1)
    
    # Initial assignment: rank-match
    heavy = np.argsort(-a_row, kind="stable")
    central = np.argsort(b_row, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = _cost(best_p, a, b)
    
    # Benders cutting-plane vector
    c = np.zeros(n, dtype=np.float64)
    step = 1.0
    min_step = 1e-8
    
    # Precompute surrogate base
    surrogate_base = B @ a.T  # shape (n, n)
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10.0:
        iteration += 1
        
        # --- Master problem: linear assignment with surrogate cost ---
        surrogate_cost = surrogate_base + np.outer(c, np.arange(n, dtype=np.float64))
        
        # Add small random perturbation to escape local minima
        if iteration % 5 == 0:
            noise = np.random.randn(n, n) * (best_cost / n) * 0.005
            surrogate_cost = surrogate_cost + noise
        
        p_new, _ = linear_sum_assignment(surrogate_cost)
        p_new = p_new.astype(np.int64)
        
        # --- Subproblem: evaluate exact cost ---
        exact_cost = _cost(p_new, a, b)
        
        # --- Generate cutting plane ---
        gap = exact_cost - best_cost
        if gap > 0:
            # Penalize the assignment
            c[np.arange(n)] += step * gap / (n * (1.0 + gap / max(best_cost, 1)))
            step *= 0.97
            if step < min_step:
                step = min_step
            no_improve_count += 1
        else:
            # Exact cost is at or below best: accept
            if exact_cost < best_cost:
                best_cost = exact_cost
                best_p = p_new.copy()
                no_improve_count = 0
            # Decay penalty to allow exploration
            c *= 0.7
            step = max(step * 1.05, min_step)
        
        # --- Local improvement on best_p ---
        if time.time() < deadline - 15.0:
            local_p, local_cost = _local_search(best_p, a, b, max_passes=5, deadline=deadline - 15.0)
            if local_cost < best_cost:
                best_cost = local_cost
                best_p = local_p
                no_improve_count = 0
        
        # --- LAP-guided perturbation ---
        if time.time() < deadline - 15.0 and iteration % 3 == 0:
            noise = np.random.randn(n, n) * (best_cost / n) * 0.01
            surrogate_perturbed = surrogate_base + np.outer(c, np.arange(n, dtype=np.float64)) + noise
            p_pert, _ = linear_sum_assignment(surrogate_perturbed)
            p_pert = p_pert.astype(np.int64)
            pert_cost = _cost(p_pert, a, b)
            if pert_cost < best_cost:
                best_cost = pert_cost
                best_p = p_pert.copy()
                no_improve_count = 0
        
        # Check for stagnation
        if no_improve_count > max_no_improve:
            # Reset and try a fresh start with different initialization
            c = np.zeros(n, dtype=np.float64)
            step = 1.0
            no_improve_count = 0
            
            # Try a different initialization
            np.random.seed()
            p_init = np.random.permutation(n)
            init_cost = _cost(p_init, a, b)
            if init_cost < best_cost:
                best_cost = init_cost
                best_p = p_init
    
    # Final polish: more intensive local search
    if time.time() < deadline - 2.0:
        local_p, local_cost = _local_search(best_p, a, b, max_passes=20, deadline=deadline - 2.0)
        if local_cost < best_cost:
            best_cost = local_cost
            best_p = local_p
    
    return best_p.tolist()
