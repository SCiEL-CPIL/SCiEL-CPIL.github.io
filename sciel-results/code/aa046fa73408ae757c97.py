"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """M is a numpy array. Returns numpy array of half-integers."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_pass_fast(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Fast ALS to refine U, V, W using vectorized matrix operations."""
    prev_err = float('inf')
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute index arrays for vectorized access
    idx_U = np.arange(n*m).reshape(n, m)  # (n, m)
    idx_V = np.arange(m*p).reshape(m, p)  # (m, p)
    idx_W = np.arange(p*n).reshape(p, n)  # (p, n)
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve V[j,:] * W[:,i] -> 1
        for i in range(n):
            for j in range(m):
                # A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                A_mat = V[idx_V[j, :], :] * W[idx_W[:, i], :]
                b_vec = np.ones(p)
                try:
                    U[idx_U[i, j], :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve U[:,j] * W[k,:] -> 1
        for j in range(m):
            for k in range(p):
                # A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                A_mat = U[idx_U[:, j], :] * W[idx_W[k, :], :]
                b_vec = np.ones(n)
                try:
                    V[idx_V[j, k], :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve U[:,i] * V[k,:] -> 1
        for i in range(n):
            for k in range(p):
                # A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_mat = U[idx_U[i, :], :] * V[idx_V[:, k], :]
                b_vec = np.ones(m)
                try:
                    W[idx_W[k, i], :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[idx_U[i, j], :], V[idx_V[j, k], :] * W[idx_W[k, i], :])
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _try_rank(rank, n, m, p, seed=42, max_iters=80, polish_iters=100):
    """Try to find a decomposition at given rank."""
    # Use rank-65 construction as base, then project to target rank
    U65, V65, W65 = _construct_rank_65(n, m, p)
    U65 = np.array(U65)
    V65 = np.array(V65)
    W65 = np.array(W65)
    
    rng = np.random.RandomState(seed)
    
    if rank >= 65:
        # Pad with zeros if needed
        if rank > 65:
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            U[:, :65] = U65
            V[:, :65] = V65
            W[:, :65] = W65
        else:
            U = U65.copy()
            V = V65.copy()
            W = W65.copy()
    else:
        # Random projection from 65 to rank
        P_U = rng.randn(65, rank)
        P_V = rng.randn(65, rank)
        P_W = rng.randn(65, rank)
        U = U65 @ P_U
        V = V65 @ P_V
        W = W65 @ P_W
    
    U, V, W = _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters)
    
    U_rounded = _round_matrix_to_half_int(U)
    V_rounded = _round_matrix_to_half_int(V)
    W_rounded = _round_matrix_to_half_int(W)
    
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try ranks from lower to higher, starting from known good range
    for rank in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 60, 65]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iters=50, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
