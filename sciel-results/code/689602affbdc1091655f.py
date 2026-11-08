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

def _als_decompose(T, n, m, p, rank, max_iter=100, seed=0):
    """
    Alternating Least Squares for tensor decomposition.
    T is the target tensor (n*m, m*p, p*n).
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    T_flat = T.flatten()
    
    for _ in range(max_iter):
        # Solve for U
        # T_flat = sum_c U[:,c] outer (V[:,c] outer W[:,c])
        # Let M_c = (V[:,c] * W[:,c]).flatten()
        # T_flat = sum_c U[:,c] * M_c  (element-wise? No, this is not right)
        # T_{ijk} = sum_c U_{i,c} V_{j,c} W_{k,c}
        # Flatten: T_flat[idx] = sum_c U_{i,c} * (V_{j,c} W_{k,c})
        # We can form a matrix A where A[idx, c] = V_{j,c} W_{k,c} for the idx corresponding to (i,j,k)
        # But idx depends on i.
        # T_flat is indexed by (i, j, k).
        # For fixed j, k, T[:, j, k] = sum_c U[:, c] * (V[j, c] W[k, c])
        # This is a linear system for U[:, c].
        
        # Construct A for U
        A_U = np.zeros((n*m, m*p, rank))
        for j in range(m*p):
            for k in range(p*n):
                # We need to map j, k to i? No, j and k are fixed.
                # The equation is for all i.
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # So for each (j, k), we have a vector of length n*m (which is n*m? No, i ranges 0..n*m-1? No, i is index of U row)
                # U has rows n*m. i in T is 0..n*m-1.
                # So T[:, j, k] is a vector of length n*m.
                # The coefficient vector for U is V[j, :] * W[k, :] (element-wise).
                A_U[:, j, k] = V[:, j] * W[:, k] # This is wrong shape.
                # V is (m*p, rank). V[j, c] is scalar.
                # So for a fixed j, k, the coefficient for U[i, c] is V[j, c] * W[k, c].
                # We want to solve T[:, j, k] = U * (V[j, :] * W[k, :])^T ?
                # T[:, j, k] is (n*m,). U is (n*m, rank).
                # Let b_{jk} = V[j, :] * W[k, :] (length rank).
                # Then T[:, j, k] = U * b_{jk}.
                # This is a linear system U * b_{jk} = T[:, j, k].
                # We can stack all (j, k) pairs.
                # There are m*p * p*n such pairs.
                # We have n*m * rank unknowns.
                # This is an overdetermined system.
                
        # Efficient construction:
        # A_U is (n*m, m*p * p*n)
        # A_U[i, j*p*n + k] = ?
        # We need A_U such that A_U @ U_flat = T_flat.
        # U_flat is (n*m * rank,).
        # This is not straightforward.
        
        # Let's use the block structure.
        # For each c, U[:, c] is a vector.
        # T = sum_c U_c outer V_c outer W_c.
        # T_flat = sum_c (U_c * (V_c * W_c).flatten()).flatten()
        # Let B_c = (V_c * W_c).flatten()
        # T_flat = sum_c U_c * B_c (element-wise? No, U_c is length n*m, B_c is length m*p*p*n)
        # This doesn't match dimensions.
        
        # Correct formulation:
        # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
        # Let's fix j, k.
        # T_{ijk} = sum_c U_{ic} (V_{jc} W_{kc})
        # Let w_{jk} = V[:, j] * W[:, k] (vector of length rank)
        # Then T[:, j, k] = U * w_{jk}
        # We have m*p * p*n equations of the form U * w = t.
        # We can solve for U using least squares.
        # Construct A (size m*p*p*n x rank) and B (size m*p*p*n x n*m)
        # A[row, c] = w_{jk, c}
        # B[row, i] = T[i, j, k]
        # Then U = B * A^+ (pseudo-inverse)
        
        rows = []
        cols = []
        vals = []
        # We can use sparse matrix construction if needed, but dense is fine for small sizes.
        A = np.zeros((m*p * p*n, rank))
        B = np.zeros((m*p * p*n, n*m))
        
        idx = 0
        for j in range(m*p):
            for k in range(p*n):
                A[idx, :] = V[:, j] * W[:, k]
                B[idx, :] = T[:, j, k]
                idx += 1
                
        # Solve A^T U^T = B^T
        # U^T = (A^T)^+ B^T
        # U = B (A^T)^+^T
        # Use np.linalg.lstsq
        U_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        U = U_sol.T
        
        # Solve for V
        # T_{ijk} = sum_c V_{jc} (U_{ic} W_{kc})
        # Fix i, k.
        # T[i, :, k] = V * (U[i, :] * W[k, :])
        A = np.zeros((n*m * p*n, rank))
        B = np.zeros((n*m * p*n, m*p))
        idx = 0
        for i in range(n*m):
            for k in range(p*n):
                A[idx, :] = U[i, :] * W[k, :]
                B[idx, :] = T[i, :, k]
                idx += 1
        V_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        V = V_sol.T
        
        # Solve for W
        # T_{ijk} = sum_c W_{kc} (U_{ic} V_{jc})
        # Fix i, j.
        # T[i, j, :] = W * (U[i, :] * V[j, :])
        A = np.zeros((n*m * m*p, rank))
        B = np.zeros((n*m * m*p, p*n))
        idx = 0
        for i in range(n*m):
            for j in range(m*p):
                A[idx, :] = U[i, :] * V[j, :]
                B[idx, :] = T[i, j, :]
                idx += 1
        W_sol, _, _, _ = np.linalg.lstsq(A.T, B.T, rcond=None)
        W = W_sol.T
        
    return U, V, W

def _find_decomposition(n, m, p, deadline):
    T = _build_tensor(n, m, p)
    # Try ranks from 44 down to 30
    for r in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            try:
                U, V, W = _als_decompose(T, n, m, p, r, max_iter=50, seed=seed)
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Convert to lists
                def to_list(mat):
                    return [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in mat]
                
                U_list = to_list(U_rounded)
                V_list = to_list(V_rounded)
                W_list = to_list(W_rounded)
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
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
