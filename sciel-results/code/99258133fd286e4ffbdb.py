import scipy
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check for integer/half-integer format
    def is_valid_format(arr):
        arr2 = arr * 2
        rounded = np.round(arr2)
        return np.allclose(arr2, rounded, atol=1e-9)
    
    if not (is_valid_format(U) and is_valid_format(V) and is_valid_format(W)):
        return False, None
        
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def snap_to_half_integers(U, V, W, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    def snap(arr):
        snapped = np.round(arr * 2) / 2.0
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return arr
    
    return snap(U), snap(V), snap(W)

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    def add_strassen(i_off, j_off, k_off, col):
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    col = add_strassen(0, 1, 0, col)
    
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)
    
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def compute_tensor_fast(U, V, W, n, m, p):
    """Compute the tensor from factor matrices using numpy broadcasting."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # tensor[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        # For each (i,j,k): u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return tensor

def search_lower_rank(n, m, p, base_U, base_V, base_W, target_rank, time_budget):
    """
    Search for a lower-rank decomposition using the base solution as initialization.
    Uses least_squares with LM method and multiple strategies (drop column, random restart).
    """
    start_time = time.time()
    
    if target_rank >= base_U.shape[1]:
        return None
    
    base_U = np.array(base_U, dtype=float)
    base_V = np.array(base_V, dtype=float)
    base_W = np.array(base_W, dtype=float)
    base_rank = base_U.shape[1]
    
    num_u = n * m * target_rank
    num_v = m * p * target_rank
    num_w = p * n * target_rank
    
    def reshape(x):
        U = x[:num_u].reshape(n * m, target_rank)
        V = x[num_u:num_u+num_v].reshape(m * p, target_rank)
        W = x[num_u+num_v:].reshape(p * n, target_rank)
        return U, V, W
    
    def residuals(x):
        U, V, W = reshape(x)
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return (T - 1.0).flatten()
    
    best_result = None
    best_cost = np.inf
    
    # Strategy 1: Drop columns from base
    for merge_idx in range(base_rank):
        if time.time() - start_time > time_budget:
            break
        keep = [i for i in range(base_rank) if i != merge_idx]
        U0 = base_U[:, keep].copy()
        V0 = base_V[:, keep].copy()
        W0 = base_W[:, keep].copy()
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res.x
            if best_cost < 1e-10:
                break
        except Exception:
            pass
            
    # Strategy 2: Random restarts with half-integer initialization
    num_restarts = 10
    for _ in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        U0 = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(n*m, target_rank))
        V0 = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(m*p, target_rank))
        W0 = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=(p*n, target_rank))
        # Avoid zero norms
        if np.all(U0 == 0): U0[0,0] = 1.0
        if np.all(V0 == 0): V0[0,0] = 1.0
        if np.all(W0 == 0): W0[0,0] = 1.0
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        try:
            res = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res.x
            if best_cost < 1e-10:
                break
        except Exception:
            pass

    if best_result is not None and best_cost < 1e-8:
        U, V, W = reshape(best_result)
        U, V, W = snap_to_half_integers(U, V, W, tol=1e-4)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        valid, _ = verify_decomposition(U_list, V_list, W_list, n, m, p)
        if valid:
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_rank = float('inf')
    
    # Fallback: Schoolbook (Rank 30) - Always valid
    rank_school = n * m * p
    U_school = [[0.0] * rank_school for _ in range(n * m)]
    V_school = [[0.0] * rank_school for _ in range(m * p)]
    W_school = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    
    valid_school, _ = verify_decomposition(U_school, V_school, W_school, n, m, p)
    if valid_school:
        best_result = {'U': U_school, 'V': V_school, 'W': W_school}
        best_rank = rank_school
    else:
        # Should never happen
        return {'U': [[1.0]], 'V': [[1.0]], 'W': [[1.0]]}
    
    # Try Rank 28 construction directly
    if time.time() < deadline:
        try:
            U28, V28, W28 = build_rank_28(n, m, p)
            valid28, _ = verify_decomposition(U28, V28, W28, n, m, p)
            if valid28:
                best_result = {'U': U28, 'V': V28, 'W': W28}
                best_rank = 28
        except Exception:
            pass
    
    # Try Rank 27 if we have a rank-28 base and time remains
    if best_rank == 28 and time.time() < deadline:
        base_U = np.array(best_result['U'])
        base_V = np.array(best_result['V'])
        base_W = np.array(best_result['W'])
        time_budget = min(deadline - time.time(), 60.0)
        if time_budget > 5.0:
            try:
                result = search_lower_rank(n, m, p, base_U, base_V, base_W, 27, time_budget)
                if result is not None:
                    U, V, W = result
                    valid, _ = verify_decomposition(U, V, W, n, m, p)
                    if valid:
                        best_result = {'U': U, 'V': V, 'W': W}
                        best_rank = 27
            except Exception:
                pass

    return best_result
