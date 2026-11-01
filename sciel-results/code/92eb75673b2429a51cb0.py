"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, T):
    """Verify exact decomposition using rational arithmetic."""
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim1)]
    V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim2)]
    W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim3)]
    
    # Check all entries
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                expected = T[i, j, k]
                actual = sum(U_frac[i][c] * V_frac[j][c] * W_frac[k][c] for c in range(rank))
                if abs(float(actual) - expected) > 1e-10:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-12, seed=42):
    """ALS decomposition with structured initialization."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = n*m, m*p, p*n
    
    T = _build_target_tensor(n, m, p)
    T_u1 = T.reshape(dim1, dim2 * dim3)
    T_u2 = T.reshape(dim2, dim1 * dim3)
    T_u3 = T.reshape(dim3, dim1 * dim2)
    
    # Structured initialization
    U = rng.randn(dim1, rank) * 0.5
    V = rng.randn(dim2, rank) * 0.5
    W = rng.randn(dim3, rank) * 0.5
    
    prev_err = float('inf')
    for it in range(max_iter):
        # Update U
        VW = np.zeros((dim2 * dim3, rank))
        for c in range(rank):
            VW[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        U_new = np.linalg.lstsq(VW, T_u1.T, rcond=None)[0].T
        U = 0.7 * U + 0.3 * U_new
        
        # Update V
        UW = np.zeros((dim1 * dim3, rank))
        for c in range(rank):
            UW[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        V_new = np.linalg.lstsq(UW, T_u2.T, rcond=None)[0].T
        V = 0.7 * V + 0.3 * V_new
        
        # Update W
        UV = np.zeros((dim1 * dim2, rank))
        for c in range(rank):
            UV[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        W_new = np.linalg.lstsq(UV, T_u3.T, rcond=None)[0].T
        W = 0.7 * W + 0.3 * W_new
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _local_search(U, V, W, T, max_steps=300, time_limit=8):
    """Local search on half-integer grid."""
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    start_time = time.time()
    
    U_cur = np.round(U * 2) / 2.0
    V_cur = np.round(V * 2) / 2.0
    W_cur = np.round(W * 2) / 2.0
    
    if _verify_exact(U_cur, V_cur, W_cur, T):
        return U_cur, V_cur, W_cur, True
    
    recon = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
    err = np.linalg.norm(T - recon)
    
    rng = np.random.RandomState(123)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for _ in range(30):
            mat_idx = rng.randint(3)
            if mat_idx == 0:
                mat, dim = U_cur, dim1
            elif mat_idx == 1:
                mat, dim = V_cur, dim2
            else:
                mat, dim = W_cur, dim3
            
            i, c = rng.randint(dim), rng.randint(rank)
            orig = mat[i, c]
            
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                mat[i, c] = orig + delta * 0.5
                recon_new = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
                err_new = np.linalg.norm(T - recon_new)
                
                if err_new < err - 1e-12:
                    err = err_new
                    improved = True
                    if err < 1e-10:
                        if _verify_exact(U_cur, V_cur, W_cur, T):
                            return U_cur, V_cur, W_cur, True
                    break
                else:
                    mat[i, c] = orig
        
        if not improved:
            for _ in range(15):
                mat_idx = rng.randint(3)
                if mat_idx == 0:
                    mat, dim = U_cur, dim1
                elif mat_idx == 1:
                    mat, dim = V_cur, dim2
                else:
                    mat, dim = W_cur, dim3
                
                i, c = rng.randint(dim), rng.randint(rank)
                orig = mat[i, c]
                new_val = rng.randint(-4, 5) * 0.5
                mat[i, c] = new_val
                recon_new = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
                err_new = np.linalg.norm(T - recon_new)
                
                if err_new < err - 1e-12:
                    err = err_new
                    improved = True
                    if err < 1e-10:
                        if _verify_exact(U_cur, V_cur, W_cur, T):
                            return U_cur, V_cur, W_cur, True
                else:
                    mat[i, c] = orig
            
            if not improved:
                break
    
    if err < 1e-10 and _verify_exact(U_cur, V_cur, W_cur, T):
        return U_cur, V_cur, W_cur, True
    
    return None, None, None, False

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    dim1, dim2, dim3 = n*m, m*p, p*n
    U = [[0] * rank for _ in range(dim1)]
    V = [[0] * rank for _ in range(dim2)]
    W = [[0] * rank for _ in range(dim3)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    T = _build_target_tensor(n, m, p)
    dim1, dim2, dim3 = n*m, m*p, p*n
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in [24, 22, 20, 18, 16, 14, 12, 10, 8]:
        if time.time() > deadline - 15:
            break
        
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            U, V, W, err = _als_decompose(n, m, p, target_rank, 
                                          max_iter=150, tol=1e-12, seed=seed)
            
            if err < 1e-8:
                U_r, V_r, W_r, ok = _local_search(U, V, W, T, 
                                                  max_steps=200, time_limit=5)
                if ok:
                    U_list = [[float(U_r[i, c]) for c in range(target_rank)] for i in range(dim1)]
                    V_list = [[float(V_r[i, c]) for c in range(target_rank)] for i in range(dim2)]
                    W_list = [[float(W_r[i, c]) for c in range(target_rank)] for i in range(dim3)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
# EVOLVE-BLOCK-END
