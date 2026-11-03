"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _build_cost_matrix(a, b, p):
    """Build cost matrix C[i,l] = sum_j a[i,j] * b[l, p[j]]."""
    V = b[:, p]
    C = a @ V.T
    return C


def _swap_delta_fast(C, a, b, p, i, j):
    """Compute swap delta using precomputed cost matrix C with correction."""
    pi = p[i]
    pj = p[j]
    delta_approx = C[i, pj] - C[i, pi] + C[j, pi] - C[j, pj]
    correction = (a[i, j] * (b[pj, pi] - b[pj, pj]) +
                  a[j, i] * (b[pi, pj] - b[pi, pi]) -
                  a[i, i] * (b[pj, pi] - b[pi, pi]) -
                  a[j, j] * (b[pi, pj] - b[pj, pj]))
    return int(delta_approx + correction)


def _update_cost_matrix(C, a, b, p, i, j):
    """Update cost matrix C after swapping p[i] and p[j]."""
    pi = p[i]
    pj = p[j]
    C += np.outer(a[:, i] - a[:, j], b[:, pj] - b[:, pi])
    p[i], p[j] = p[j], p[i]


def _local_search(C, a, b, p, current_cost, deadline, max_passes=50):
    """Steepest descent local search using precomputed cost matrix."""
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 2:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                d = _swap_delta_fast(C, a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        _update_cost_matrix(C, a, b, p, best_i, best_j)
        current_cost += best_d
    return p, current_cost, C


def _spectral_initial(a, b, n, k=10, rng=None):
    """Generate initial permutation using spectral projection of flow and distance matrices."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    
    # SVD for top-k components
    try:
        Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
        k = min(k, n)
        A_proj = (Ua[:, :k] * Sa[:k]) @ Va[:k, :]
        B_proj = (Ub[:, :k] * Sb[:k]) @ Vb[:k, :]
    except:
        A_proj = a_centered
        B_proj = b_centered
    
    # Build surrogate cost matrix: C[i,l] = sum_k A_proj[i,k] * B_proj[l,k]
    C = A_proj @ B_proj.T
    
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _lap_initial(a, b, n):
    """Get initial assignment using LAP on linearized cost."""
    C = a @ b
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _rank_initial(a, b, n):
    """Rank-matching initial assignment."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _benders_cut(a, b, p, cut_coeff, n):
    """Generate a Benders cut from current assignment p."""
    L = a @ b[:, p].T
    np.minimum(cut_coeff, L, out=cut_coeff)
    return L


def _surrogate_assignment(cut_coeff, n, rng):
    """Assign facilities to locations using the surrogate cost matrix."""
    assigned = np.zeros(n, dtype=bool)
    p = np.empty(n, dtype=np.int64)
    order = rng.permutation(n)
    for idx in order:
        row = cut_coeff[idx].copy()
        row[assigned] = np.inf
        best_loc = np.argmin(row)
        p[idx] = best_loc
        assigned[best_loc] = True
    return p


def _sa_run(a, b, p_init, T_start, T_min, cooling, max_iters, deadline, best_cost, best_p):
    """Run one SA iteration and return updated best."""
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    C = _build_cost_matrix(a, b, p)
    T = T_start
    rng = np.random.default_rng()
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    for it in range(max_iters):
        if time.time() >= deadline - 3:
            break
        if T < T_min:
            break
        
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        d = _swap_delta_fast(C, a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            _update_cost_matrix(C, a, b, p, i, j)
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling
    
    return best_cost, best_p


def solve(a, b):
    n = len(a)
    deadline = time.time() + 118
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions
    p_lap = _lap_initial(a, b, n)
    p_rank = _rank_initial(a, b, n)
    p_spec = _spectral_initial(a, b, n, k=8, rng=rng)
    
    candidates = [p_lap, p_rank, p_spec]
    best_p = candidates[0].copy()
    best_cost = _qap_cost(a, b, best_p)
    
    for pc in candidates[1:]:
        c = _qap_cost(a, b, pc)
        if c < best_cost:
            best_cost = c
            best_p = pc.copy()
    
    # Benders cut storage
    cut_coeff = np.full((n, n), np.inf, dtype=np.float64)
    _benders_cut(a, b, best_p, cut_coeff, n)
    
    # Phase 1: Local search from best initial
    C = _build_cost_matrix(a, b, best_p)
    p_ls = best_p.copy()
    cost_ls = best_cost
    C_ls = C.copy()
    p_ls, cost_ls, C_ls = _local_search(C_ls, a, b, p_ls, cost_ls, deadline, max_passes=40)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
        C = C_ls.copy()
        _benders_cut(a, b, best_p, cut_coeff, n)
    
    # Phase 2: Multi-start SA with iterated local search
    T_start = 2500.0
    T_min = 0.3
    cooling = 0.99995
    max_iters = 8000
    
    while time.time() < deadline - 20:
        # Choose starting point: mix of best, surrogate, and perturbed
        r = rng.random()
        if r < 0.3:
            p = _surrogate_assignment(cut_coeff, n, rng)
        elif r < 0.5:
            p = best_p.copy()
            # Perturb best solution
            for _ in range(rng.integers(5, 15)):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    p[i], p[j] = p[j], p[i]
        else:
            # Spectral restart with randomization
            p = _spectral_initial(a, b, n, k=rng.integers(5, 12), rng=rng)
        
        best_cost, best_p = _sa_run(a, b, p, T_start, T_min, cooling, max_iters, deadline, best_cost, best_p)
        
        # Polish with local search
        if time.time() < deadline - 10:
            C_pol = _build_cost_matrix(a, b, best_p)
            p_pol = best_p.copy()
            cost_pol = best_cost
            p_pol, cost_pol, C_pol = _local_search(C_pol, a, b, p_pol, cost_pol, deadline, max_passes=25)
            if cost_pol < best_cost:
                best_cost = cost_pol
                best_p = p_pol.copy()
                C = C_pol.copy()
                _benders_cut(a, b, best_p, cut_coeff, n)
    
    # Final polish: multiple passes
    for _ in range(8):
        if time.time() >= deadline - 3:
            break
        C_pol = _build_cost_matrix(a, b, best_p)
        p_pol = best_p.copy()
        cost_pol = best_cost
        p_pol, cost_pol, C_pol = _local_search(C_pol, a, b, p_pol, cost_pol, deadline, max_passes=15)
        if cost_pol < best_cost:
            best_cost = cost_pol
            best_p = p_pol.copy()
            C = C_pol.copy()
        else:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
