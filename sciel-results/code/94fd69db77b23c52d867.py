"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = _build_tensor(n, m, p)
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _als_decompose_fast(T, n, m, p, rank, max_iter=100, seed=0, tol=1e-9):
    """
    Fast Alternating Least Squares for tensor decomposition using vectorized ops.
    T is the target tensor (n*m, m*p, p*n).
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Precompute shapes
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    
    # T is (dim_u, dim_v, dim_w)
    # We need to solve for U, V, W iteratively.
    
    prev_error = np.inf
    for _ in range(max_iter):
        # 1. Solve for U
        # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
        # For fixed j, k: T[:, j, k] = U * (V[j, :] * W[k, :])
        # A_U shape: (dim_v * dim_w, rank)
        # A_U[j*dim_w + k, c] = V[j, c] * W[k, c]
        # B_U shape: (dim_v * dim_w, dim_u)
        # B_U[j*dim_w + k, i] = T[i, j, k]
        
        # Construct A_U
        # V is (dim_v, rank), W is (dim_w, rank)
        # We want outer product of rows, then element-wise multiply? No.
        # V[j, c] * W[k, c].
        # V[:, None, :] * W[None, :, :] -> (dim_v, dim_w, rank)
        A_U = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
        
        # Construct B_U
        # T is (dim_u, dim_v, dim_w). We want (dim_v, dim_w, dim_u) -> (dim_v*dim_w, dim_u)
        B_U = np.transpose(T, (1, 2, 0)).reshape(-1, dim_u)
        
        # Solve A_U^T U^T = B_U^T  => U^T = (A_U^T)^+ B_U^T
        # U = B_U (A_U^T)^+^T
        # Using lstsq: U^T = lstsq(A_U.T, B_U.T)
        U_new, _, _, _ = np.linalg.lstsq(A_U.T, B_U.T, rcond=None)
        U = U_new.T
        
        # 2. Solve for V
        # T_{ijk} = sum_c V_{jc} U_{ic} W_{kc}
        # For fixed i, k: T[i, :, k] = V * (U[i, :] * W[k, :])
        # A_V shape: (dim_u * dim_w, rank)
        # A_V[i*dim_w + k, c] = U[i, c] * W[k, c]
        # B_V shape: (dim_u * dim_w, dim_v)
        # B_V[i*dim_w + k, j] = T[i, j, k]
        
        A_V = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
        B_V = np.transpose(T, (0, 2, 1)).reshape(-1, dim_v)
        
        V_new, _, _, _ = np.linalg.lstsq(A_V.T, B_V.T, rcond=None)
        V = V_new.T
        
        # 3. Solve for W
        # T_{ijk} = sum_c W_{kc} U_{ic} V_{jc}
        # For fixed i, j: T[i, j, :] = W * (U[i, :] * V[j, :])
        # A_W shape: (dim_u * dim_v, rank)
        # A_W[i*dim_v + j, c] = U[i, c] * V[j, c]
        # B_W shape: (dim_u * dim_v, dim_w)
        # B_W[i*dim_v + j, k] = T[i, j, k]
        
        A_W = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
        B_W = np.transpose(T, (0, 1, 2)).reshape(-1, dim_w) # T is already (u, v, w), so just reshape?
        # T[i, j, k]. We want row index i*dim_v + j.
        # np.transpose(T, (0, 1, 2)) is T. Reshape to (-1, dim_w) works if C-contiguous.
        # T[i, j, k] -> flat index i*dim_v*dim_w + j*dim_w + k.
        # Row i*dim_v + j corresponds to flat block i*dim_v*dim_w + j*dim_w.
        # Yes, this works.
        
        W_new, _, _, _ = np.linalg.lstsq(A_W.T, B_W.T, rcond=None)
        W = W_new.T
        
        # Check convergence
        # Calculate error
        T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
        error = np.linalg.norm(T_decomp - T)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
        
    return U, V, W

def _find_decomposition(n, m, p, deadline):
    T = _build_tensor(n, m, p)
    # Try ranks from 44 down to 30
    # Schoolbook is 45.
    # Known lower bound for 3x3x3 is 20? No, Strassen is 7.
    # For 3x3x5, it's likely around 40-45.
    # We try high ranks first as they are easier to find.
    for r in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
        # Use more seeds for lower ranks? Or just a fixed number.
        # With fast ALS, we can afford more seeds.
        num_seeds = 50 if r > 40 else 100
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            try:
                U, V, W = _als_decompose_fast(T, n, m, p, r, max_iter=100, seed=seed)
                
                # Round to half-integers
                # Check if rounding to half-integers yields an exact solution
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist(), "rank": r}
                    
                # Also check integer rounding just in case
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist(), "rank": r}
                    
            except Exception:
                continue
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    # Try to find a better decomposition
    result = _find_decomposition(n, m, p, deadline)
    if result:
        return result
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
