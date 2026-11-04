# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    n = X.shape[0]
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _init_state(a, b, p):
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    RC = b_float[:, p]
    CC = b_float[p, :]
    S1 = a_float @ RC.T
    S2 = a_float.T @ CC
    U = S1 + S2
    return S1, S2, U, RC, CC, a_float, b_float


def _update_state(S1, S2, U, RC, CC, a_float, i, j, p):
    li, lj = p[i], p[j]
    
    # Swap RC columns
    RC[:, i], RC[:, j] = RC[:, j].copy(), RC[:, i].copy()
    # Swap CC rows
    CC[i, :], CC[j, :] = CC[j, :].copy(), CC[i, :].copy()
    
    # Update S1 columns
    S1[:, i] = a_float @ RC[:, i]
    S1[:, j] = a_float @ RC[:, j]
    
    # Update S2
    # Delta S2 = outer(a[i, :] - a[j, :], CC[i, :] - CC[j, :])
    v = a_float[i, :] - a_float[j, :]
    w = CC[i, :] - CC[j, :]
    Delta_S2 = np.outer(v, w)
    S2 += Delta_S2
    
    # Update U
    U = S1 + S2
    
    return S1, S2, U, RC, CC


def _two_opt_search(a, b, p, time_limit):
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    
    S1, S2, U, RC, CC, a_float, b_float = _init_state(a, b, p)
    
    start = time.time()
    idx = np.arange(n)
    
    while time.time() - start < time_limit:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            li = p[i]
            vec1 = U[i, p]
            vec2 = U[idx, p]
            vec3 = U[:, li]
            
            deltas = vec1 - vec2 + vec3 - U[i, li]
            deltas[i] = 0
            
            min_idx = np.argmin(deltas)
            if deltas[min_idx] < best_delta:
                best_delta = deltas[min_idx]
                best_i = i
                best_j = min_idx
        
        if best_i == -1:
            break
        
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        cost += best_delta
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        S1, S2, U, RC, CC = _update_state(S1, S2, U, RC, CC, a_float, i, j, p)
        
    return best_p, best_cost


def _simulated_annealing(a, b, p, time_limit):
    n = len(p)
    p = p.copy()
    cost = _qap_cost(a, b, p)
    best_cost = cost
    best_p = p.copy()
    
    S1, S2, U, RC, CC, a_float, b_float = _init_state(a, b, p)
    
    start = time.time()
    T = 1000.0
    T_min = 1.0
    cooling = 0.99
    idx = np.arange(n)
    
    while T > T_min and (time.time() - start) < time_limit:
        i, j = np.random.choice(n, 2, replace=False)
        if i == j: continue
        
        li = p[i]
        vec1 = U[i, p]
        vec2 = U[idx, p]
        vec3 = U[:, li]
        # We only need delta for specific j
        # delta = vec1[j] - vec2[j] + vec3[j] - U[i, li]
        delta = vec1[j] - vec2[j] + vec3[j] - U[i, li]
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost += delta
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
            S1, S2, U, RC, CC = _update_state(S1, S2, U, RC, CC, a_float, i, j, p)
        
        T *= cooling
        
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start = time.time()
    deadline = start + 115

    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0

    k = min(30, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    p = _procrustes_align(X, Y)
    cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cost

    time_left = deadline - time.time()
    if time_left > 10:
        p, cost = _two_opt_search(a, b, p, time_left - 5)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _simulated_annealing(a, b, best_p, time_left - 1)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p.tolist()
# EVOLVE-BLOCK-END
