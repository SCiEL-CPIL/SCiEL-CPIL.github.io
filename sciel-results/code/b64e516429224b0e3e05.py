import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _add_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0, start_col):
    """Add Strassen decomposition for 2x2x2 block."""
    col = start_col
    
    def u_idx(ii, jj):
        return (i0 + ii) * m + (j0 + jj)
    
    def v_idx(jj, kk):
        return (j0 + jj) * p + (k0 + kk)
    
    def w_idx(kk, ii):
        return (k0 + kk) * n + (i0 + ii)
    
    # M1 = (A00+A11)(B00+B11)
    U[u_idx(0, 0)][col] = 1
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 0)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = 1
    W[w_idx(1, 1)][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[u_idx(1, 0)][col] = 1
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 0)][col] = 1
    W[w_idx(0, 1)][col] = 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[u_idx(0, 0)][col] = 1
    V[v_idx(1, 0)][col] = 1
    V[v_idx(1, 1)][col] = -1
    W[w_idx(1, 0)][col] = 1
    col += 1
    
    # M4 = A11(B01-B00)
    U[u_idx(1, 1)][col] = 1
    V[v_idx(0, 1)][col] = 1
    V[v_idx(0, 0)][col] = -1
    W[w_idx(0, 0)][col] = 1
    W[w_idx(0, 1)][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[u_idx(0, 0)][col] = 1
    U[u_idx(0, 1)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = -1
    W[w_idx(1, 0)][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[u_idx(1, 0)][col] = 1
    U[u_idx(0, 0)][col] = -1
    V[v_idx(0, 0)][col] = 1
    V[v_idx(0, 1)][col] = 1
    W[w_idx(1, 1)][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[u_idx(0, 1)][col] = 1
    U[u_idx(1, 1)][col] = -1
    V[v_idx(1, 0)][col] = 1
    V[v_idx(1, 1)][col] = 1
    W[w_idx(0, 0)][col] = 1
    col += 1
    
    return col

def _build_rank34():
    """Build the rank-34 decomposition."""
    n, m, p = 3, 3, 4
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, col)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 2, col)
    
    # Schoolbook for remaining blocks
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
    for i in range(2):
        for k in range(4):
            U[(0 + i) * 3 + 2][col] = 1
            V[2 * 4 + k][col] = 1
            W[k * 3 + (0 + i)][col] = 1
            col += 1
            
    # Block 4: A[2:3, 0:2] * B[0:2, 0:4]
    for j in range(2):
        for k in range(4):
            U[2 * 3 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    # Block 5: A[2:3, 2:3] * B[2:3, 0:4]
    for k in range(4):
        U[2 * 3 + 2][col] = 1
        V[2 * 4 + k][col] = 1
        W[k * 3 + 2][col] = 1
        col += 1
        
    if col != rank:
        raise ValueError(f"Rank mismatch: {col} != {rank}")
        
    return U, V, W

def _try_optimize(rank, start_U, start_V, start_W, deadline):
    """Try to optimize for a given rank."""
    n, m, p = 3, 3, 4
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    # Flatten start
    x0 = np.concatenate([start_U.flatten(), start_V.flatten(), start_W.flatten()])
    # If start rank > target rank, truncate
    if len(x0) > total_size:
        x0 = x0[:total_size]
    elif len(x0) < total_size:
        # Pad with zeros
        x0 = np.concatenate([x0, np.zeros(total_size - len(x0))])
        
    def residual(x):
        Uf = x[:size_U].reshape(n*m, rank)
        Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
        Wf = x[size_U+size_V:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    try:
        res = least_squares(residual, x0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
        if res.cost < 1e-10:
            x_sol = res.x
            U_sol = x_sol[:size_U].reshape(n*m, rank)
            V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
            W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
            
            # Round to nearest integer
            U_r = np.round(U_sol)
            V_r = np.round(V_sol)
            W_r = np.round(W_sol)
            
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-8):
                return U_r, V_r, W_r
    except Exception:
        pass
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Get the rank 34 solution
    U34, V34, W34 = _build_rank34()
    
    # Try to improve to rank 32
    for rank in [32, 31, 30]:
        if time.time() > deadline:
            break
        
        # Initialize with truncated 34 solution
        # This is a rough initialization
        U_init = np.zeros((n*m, rank))
        V_init = np.zeros((m*p, rank))
        W_init = np.zeros((p*n, rank))
        
        # Copy first 'rank' columns
        U_init[:, :] = U34[:, :rank]
        V_init[:, :] = V34[:, :rank]
        W_init[:, :] = W34[:, :rank]
        
        result = _try_optimize(rank, U_init, V_init, W_init, deadline)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 34
    return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}
