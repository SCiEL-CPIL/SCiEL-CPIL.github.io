import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape for efficient tensor contraction
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    # We want to check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # This is a tensor contraction.
    
    # Reshape
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This can be done via einsum
    # T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
    # But indices for W are k,n,i. W_t is (p, n, rank). So W[k, i, r].
    # So: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    except Exception:
        return False

    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-8):
        return False
    return True

def build_rank_28(n, m, p):
    """Construct a rank-28 decomposition for 2x3x5 matrix multiplication."""
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0
        col += 1

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M2
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = -1.0
        col += 1
        # M3
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M4
        U[(i_off+1)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+0), col] = -1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        W[(k_off+0)*n + (i_off+1), col] = 1.0
        col += 1
        # M5
        U[(i_off+0)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = -1.0
        W[(k_off+1)*n + (i_off+0), col] = 1.0
        col += 1
        # M6
        U[(i_off+1)*m + (j_off+0), col] = 1.0
        U[(i_off+0)*m + (j_off+0), col] = -1.0
        V[(j_off+0)*p + (k_off+0), col] = 1.0
        V[(j_off+0)*p + (k_off+1), col] = 1.0
        W[(k_off+1)*n + (i_off+1), col] = 1.0
        col += 1
        # M7
        U[(i_off+0)*m + (j_off+1), col] = 1.0
        U[(i_off+1)*m + (j_off+1), col] = -1.0
        V[(j_off+1)*p + (k_off+0), col] = 1.0
        V[(j_off+1)*p + (k_off+1), col] = 1.0
        W[(k_off+0)*n + (i_off+0), col] = 1.0
        col += 1

    # 1. j=0 block: rank 10
    for k in range(p):
        add_schoolbook(0, 0, k)
        add_schoolbook(1, 0, k)
    
    # 2. j=1,2, k=0,1 block: rank 7
    add_strassen(0, 1, 0)

    # 3. j=1,2, k=2 block: rank 4
    for i in range(n):
        for j in [1, 2]:
            add_schoolbook(i, j, 2)

    # 4. j=1,2, k=3,4 block: rank 7
    add_strassen(0, 1, 3)

    assert col == rank
    return U, V, W

def try_find_lower_rank(n, m, p, target_rank, time_limit=100):
    """Attempt to find a decomposition of lower rank using numerical optimization."""
    start_time = time.time()
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    total_vars = target_rank * (num_u + num_v + num_w)
    
    # We flatten U, V, W into a single vector x
    # x[0 : target_rank*num_u] is U
    # x[target_rank*num_u : target_rank*(num_u+num_v)] is V
    # x[target_rank*(num_u+num_v) : total_vars] is W
    
    def residuals(x):
        U_flat = x[:target_rank*num_u]
        V_flat = x[target_rank*num_u : target_rank*(num_u+num_v)]
        W_flat = x[target_rank*(num_u+num_v) : total_vars]
        
        U = U_flat.reshape(num_u, target_rank)
        V = V_flat.reshape(num_v, target_rank)
        W = W_flat.reshape(num_w, target_rank)
        
        U_t = U.reshape(n, m, target_rank)
        V_t = V.reshape(m, p, target_rank)
        W_t = W.reshape(p, n, target_rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        expected = np.ones((n, m, p))
        return (T - expected).flatten()

    # Try a few random initializations
    np.random.seed(42)
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        x0 = np.random.randn(total_vars) * 0.1
        # Scale to help convergence?
        
        try:
            res = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if res.cost < 1e-12:
                # Found a solution
                U_flat = res.x[:target_rank*num_u]
                V_flat = res.x[target_rank*num_u : target_rank*(num_u+num_v)]
                W_flat = res.x[target_rank*(num_u+num_v) : total_vars]
                
                U = U_flat.reshape(num_u, target_rank)
                V = V_flat.reshape(num_v, target_rank)
                W = W_flat.reshape(num_w, target_rank)
                
                # Verify
                if verify_decomposition(U, V, W, n, m, p):
                    return U, V, W
        except Exception:
            pass

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, try to find a rank 27 decomposition
    # This is the "stronger algorithm" attempt
    try:
        result = try_find_lower_rank(n, m, p, 27, time_limit=100)
        if result is not None:
            U, V, W = result
            # Convert to list of lists as per interface
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass

    # Fallback to known optimal rank 28
    U, V, W = build_rank_28(n, m, p)
    if verify_decomposition(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Ultimate fallback: Schoolbook rank 30
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
