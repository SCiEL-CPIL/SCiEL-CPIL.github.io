"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _compute_delta_matrix(a, b, p):
    """
    Compute the O(n x n) matrix D where D[i, j] is the change in cost
    if we swap the locations assigned to facilities i and j in p.
    Uses vectorized NumPy operations for speed.
    """
    n = len(p)
    # M[i, l] = sum_k a[i, k] * b[l, p[k]]
    # M = a @ b[:, p].T
    M = a @ b[:, p].T
    # T[i, l] = sum_k a[k, i] * b[p[k], l]
    # T = a.T @ b[p, :]
    T = a.T @ b[p, :]
    
    # D[i, j] = M[i, p[j]] + T[i, p[j]] - M[i, p[i]] - T[i, p[i]]
    #         - M[j, p[i]] - T[j, p[i]] + M[j, p[j]] + T[j, p[j]]
    D = (M[np.arange(n), p] + T[np.arange(n), p]
         - M[np.arange(n)[:, None], p[None, :]] - T[np.arange(n)[:, None], p[None, :]]
         - M[p[None, :], np.arange(n)[None, :]] - T[p[None, :], np.arange(n)[None, :]]
         + M[p, np.arange(n)] + T[p, np.arange(n)])
    return D


def _local_search_vec(a, b, p, deadline, max_passes=50):
    """
    Vectorized first-improvement local search.
    """
    n = len(p)
    cost = _cost(a, b, p)
    p = p.copy()
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best negative delta (excluding diagonal)
        np.fill_diagonal(D, 0)
        min_val = np.min(D)
        
        if min_val >= 0:
            break
        
        idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = idx
        
        p[i], p[j] = p[j], p[i]
        cost += int(min_val)
        
    return p, cost


def _frank_wolfe_qap(a, b, n, deadline):
    """
    Frank-Wolfe (Conditional Gradient) descent on the Birkhoff polytope.
    Returns a permutation.
    """
    # Start with a random doubly stochastic matrix (or identity)
    X = np.eye(n, dtype=np.float64)
    step_size = 1.0
    
    for t in range(1000):
        if time.time() > deadline:
            break
            
        # Compute gradient of the QAP objective at X
        # f(X) = sum_{i,j,k,l} a[i,j] X[i,k] b[k,l] X[j,l]
        # grad[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        # Let G1[i,k] = (A * (X^T B X^T)^T)[i,k] ... let's use matrix mults
        # Term 1: (A @ (X.T @ B @ X)) ? No.
        # sum_j a[i,j] (sum_l b[k,l] X[j,l]) = sum_j a[i,j] (B @ X.T)[k,j].T ?
        # Let Y = X.T @ B @ X.T  (n x n)
        # Y[k,l] = sum_{j,m} X[j,k] b[k,l] X[m,l] ... no
        # Let's stick to the standard formula:
        # grad = A @ (X.T @ B @ X) + A.T @ (X.T @ B @ X).T
        # Wait, f(X) = trace(X^T A X B^T) is for a different formulation.
        
        # Let's use the explicit derivative:
        # df/dX[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        
        # Compute S1[i,k] = sum_j a[i,j] (sum_l b[k,l] X[j,l])
        # Let M1 = B @ X.T  (M1[k,l] = sum_j b[k,l] X[j,l] is wrong, M1[k,j] = sum_l b[k,l] X[j,l])
        # M1 = B @ X.T
        # S1 = A @ M1  (S1[i,k] = sum_j a[i,j] M1[j,k] = sum_j a[i,j] sum_l b[k,l] X[j,l])
        # This matches the first term if we swap indices correctly.
        # Actually, grad[i,k] = S1[i,k] + S2[i,k]
        
        M1 = B @ X.T  # M1[k, j] = sum_l b[k, l] X[j, l]
        S1 = A @ M1   # S1[i, k] = sum_j a[i, j] M1[j, k] ... wait, M1 is B @ X.T, so M1[j, k] is sum_l b[j, l] X[k, l]? No.
        
        # Let's be precise.
        # X is n x n. X[i, k] is prob fac i is at loc k.
        # f(X) = sum_{i,j} a[i,j] sum_{k,l} b[k,l] X[i,k] X[j,l]
        # grad[i,k] = sum_{j,l} a[i,j] b[k,l] X[j,l] + sum_{j,l} a[j,i] b[l,k] X[j,l]
        
        # Term 1: T1[i,k] = sum_j a[i,j] (sum_l b[k,l] X[j,l])
        # Let V1[j, l] = X[j, l].
        # Inner sum: sum_l b[k,l] X[j,l] = (B @ X.T)[k, j]
        # So T1[i,k] = sum_j a[i,j] (B @ X.T)[k, j]
        # T1 = A @ (B @ X.T).T = A @ X @ B.T
        
        # Term 2: T2[i,k] = sum_j a[j,i] (sum_l b[l,k] X[j,l])
        # Inner sum: sum_l b[l,k] X[j,l] = (B.T @ X.T)[k, j] ? No.
        # sum_l b[l,k] X[j,l] = (X @ B)[j, k]
        # So T2[i,k] = sum_j a[j,i] (X @ B)[j, k]
        # T2 = A.T @ (X @ B)
        
        grad = A @ X @ B.T + A.T @ X @ B
        
        # Frank-Wolfe step
        # Linear minimization: min_{Y in Birkhoff} <grad, Y>
        # This is the Hungarian algorithm on grad
        row_ind, col_ind = linear_sum_assignment(grad)
        S = np.zeros((n, n), dtype=np.float64)
        S[row_ind, col_ind] = 1.0
        
        # Update
        alpha = 2.0 / (t + 2.0)
        X = (1.0 - alpha) * X + alpha * S
        
    # Discretize
    row_ind, col_ind = linear_sum_assignment(X)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _greedy_init(a, b, n, rng):
    """
    Greedy initialization based on row sums.
    """
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow
    fac_order = np.argsort(-a_sums)
    # Sort locations by ascending distance sum (central)
    loc_order = np.argsort(b_sums)
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    
    # Add some noise
    if rng.random() < 0.5:
        # Randomly swap a few
        for _ in range(n // 10):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                p[i], p[j] = p[j], p[i]
                
    return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    
    # 1. Greedy
    p1 = _greedy_init(a, b, n, rng)
    inits.append(p1)
    
    # 2. Random
    p2 = rng.permutation(n).astype(np.int64)
    inits.append(p2)
    
    # 3. Frank-Wolfe
    if time.time() < deadline - 30.0:
        p3 = _frank_wolfe_qap(a, b, n, deadline - 20.0)
        inits.append(p3)
    
    # 4. Reverse Greedy
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    fac_order = np.argsort(a_sums)
    loc_order = np.argsort(-b_sums)
    p4 = np.empty(n, dtype=np.int64)
    p4[fac_order] = loc_order
    inits.append(p4)
    
    # Phase 2: Local Search from each init
    for p_init in inits:
        if time.time() > deadline - 10.0:
            break
            
        p_ls, c_ls = _local_search_vec(a, b, p_init, deadline - 5.0)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
    
    # Phase 3: Iterated Local Search (ILS)
    current_p = best_p.copy()
    current_cost = best_cost
    
    ilse_deadline = deadline - 5.0
    temp = 1.0
    
    while time.time() < ilse_deadline:
        # Perturb
        p_pert = current_p.copy()
        k_swaps = rng.integers(2, 6)
        for _ in range(k_swaps):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Local Search
        p_ls, c_ls = _local_search_vec(a, b, p_pert, ilse_deadline)
        
        # Accept/Reject
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
            current_p = p_ls
            current_cost = c_ls
        elif c_ls < current_cost:
            # Always accept improvement
            current_p = p_ls
            current_cost = c_ls
        else:
            # Metropolis acceptance
            delta = c_ls - current_cost
            if delta < 0:
                current_p = p_ls
                current_cost = c_ls
            else:
                accept_prob = np.exp(-delta / (temp * 100.0))
                if rng.random() < accept_prob:
                    current_p = p_ls
                    current_cost = c_ls
        
        temp *= 0.95
        if temp < 0.1:
            temp = 0.1
    
    # Phase 4: Final Polish
    if time.time() < deadline - 2.0:
        final_p, final_cost = _local_search_vec(a, b, best_p, deadline - 1.0)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
