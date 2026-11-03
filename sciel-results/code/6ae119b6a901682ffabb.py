import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse import lil_matrix, csr_matrix

def _build_target(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.ones((n, m, p))
    return T

def _tensor_to_vec(T, n, m, p):
    """Flatten tensor to vector using convention: index = i*m*p + j*p + k."""
    return T.reshape(-1)

def _vec_to_tensor(v, n, m, p):
    return v.reshape((n, m, p))

def _decomp_to_tensor(U, V, W, n, m, p):
    """Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    return T

def _verify(U, V, W, n, m, p, tol=1e-8):
    T = _decomp_to_tensor(U, V, W, n, m, p)
    return np.allclose(T, 1.0, atol=tol)

def _block_decomposition_356(n, m, p):
    """Block decomposition with rank 44."""
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1
    
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1
    
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = 1; S_W[3, 4] = -1
    
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = -1
    
    col = 0
    # Block 1: 2x2x6
    for bk in range(3):
        ko = bk * 2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j, col] += S_U[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[j*p+(ko+k), col] += S_V[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(ko+k)*n+i, col] += S_W[k*2+i, r]
            col += 1
    
    # Block 2: 1x5x6, rank 1
    for j in range(5):
        U[2*m+j, col] = 1
    for j in range(5):
        for k in range(6):
            V[j*p+k, col] = 1
    for k in range(6):
        W[k*n+2, col] = 1
    col += 1
    
    # Block 3: 2x2x6
    for bk in range(3):
        ko = bk * 2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+(2+j), col] += S_U[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*p+(ko+k), col] += S_V[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(ko+k)*n+i, col] += S_W[k*2+i, r]
            col += 1
    
    # Block 4: 2x1x6, rank 1
    for i in range(2):
        U[i*m+4, col] = 1
    for k in range(6):
        V[4*p+k, col] = 1
    for k in range(6):
        for i in range(2):
            W[k*n+i, col] = 1
    col += 1
    
    return U, V, W

def _try_reduce_rank_sparse(U, V, W, n, m, p, target_rank, max_iter=50, tol=1e-10, time_limit=20):
    """
    Try to reduce rank using sparse linear algebra approach.
    This is a linear problem: find U', V', W' of rank target_rank such that
    the tensor decomposition equals the target.
    """
    start = time.time()
    cur_rank = U.shape[1]
    
    if target_rank >= cur_rank:
        return U, V, W
    
    # For a given rank r, we need to find U', V', W' such that
    # sum_{r'} U'[i*m+j, r'] * V'[j*p+k, r'] * W'[k*n+i, r'] = 1 for all i,j,k
    
    # This is a bilinear system. We can fix two matrices and solve for the third.
    # Strategy: randomly initialize U' and V', then solve for W'
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Build the linear system for W given U and V
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is: sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    
    # The coefficient matrix A has rows indexed by (i,j,k) and columns by (k*n+i, r)
    # A[(i,j,k), (k*n+i, r)] = U[i*m+j, r] * V[j*p+k, r]
    
    # Number of equations: n*m*p
    # Number of unknowns: nW * target_rank
    
    num_eq = n * m * p
    num_unk = nW * target_rank
    
    if num_unk >= num_eq:
        # Underdetermined or square system, may have solutions
        pass
    else:
        # Overdetermined, may not have exact solutions
        return None, None, None
    
    # Try multiple random initializations
    best_result = None
    best_residual = np.inf
    
    for trial in range(3):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U_try = np.random.randn(nU, target_rank) * 0.5
        V_try = np.random.randn(nV, target_rank) * 0.5
        
        # Build the linear system A * W_vec = b
        # A has shape (num_eq, num_unk)
        # For equation (i,j,k): index = i*m*p + j*p + k
        # For unknown (k*n+i, r): index = (k*n+i) * target_rank + r
        
        A_rows = []
        A_cols = []
        A_vals = []
        b = np.ones(num_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(target_rank):
                        coeff = U_try[i * m + j, r] * V_try[j * p + k, r]
                        if abs(coeff) > 1e-12:
                            unk_idx = (k * n + i) * target_rank + r
                            A_rows.append(eq_idx)
                            A_cols.append(unk_idx)
                            A_vals.append(coeff)
        
        A_sparse = lil_matrix((num_eq, num_unk), dtype=float)
        for row, col, val in zip(A_rows, A_cols, A_vals):
            A_sparse[row, col] = val
        A_csr = A_sparse.tocsr()
        
        # Solve using least squares (may be over/under-determined)
        try:
            from scipy.sparse.linalg import lsqr
            W_vec, istop, itn, normr, anorm, resid, cond = lsqr(A_csr, b, atol=tol, btol=tol, iter_lim=1000)
            residual = np.linalg.norm(A_csr @ W_vec - b)
            
            if residual < best_residual:
                best_residual = residual
                if residual < 1e-8:
                    W_try = W_vec.reshape(nW, target_rank)
                    if _verify(U_try, V_try, W_try, n, m, p, tol=1e-8):
                        return U_try, V_try, W_try
                    best_result = (U_try.copy(), V_try.copy(), W_try.copy())
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    return None, None, None

def construct():
    n, m, p = 3, 5, 6
    start = time.time()
    deadline = start + 140
    
    U, V, W = _block_decomposition_356(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback: trivial
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank using sparse linear algebra
    for target_rank in range(best_rank - 1, max(best_rank - 15, 20), -1):
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - time.time() - 15
        if remaining_time < 5:
            break
        
        result = _try_reduce_rank_sparse(U, V, W, n, m, p, target_rank, 
                                          max_iter=50, tol=1e-10, 
                                          time_limit=min(20, remaining_time))
        if result[0] is not None:
            if _verify(result[0], result[1], result[2], n, m, p):
                best_U, best_V, best_W = result
                best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
