"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Target: T[i, j, k] = 1 for all i, j, k
        
        # Efficient verification using matrix multiplication
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape to matrices
        U_mat = U_arr.reshape(n, m, rank)
        V_mat = V_arr.reshape(m, p, rank)
        W_mat = W_arr.reshape(p, n, rank)
        
        # For each (i, k), we need sum_j sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        # Let's compute for each c, the contribution to T[i,j,k]
        # and sum over c.
        
        # Alternative: For each c, define:
        # A_c = U[:,c] reshaped to (n,m)
        # B_c = V[:,c] reshaped to (m,p)
        # C_c = W[:,c] reshaped to (p,n)
        # Then T = sum_c A_c[i,j] * B_c[j,k] * C_c[k,i]
        
        T_computed = np.zeros((n, m, p), dtype=float)
        
        for c in range(rank):
            u_vec = U_arr[:, c]
            v_vec = V_arr[:, c]
            w_vec = W_arr[:, c]
            
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This can be computed as: for each i, k, 
            # T[i, :, k] += u_mat[i, :] * (v_mat * w_mat[k, :])^T ... no
            # T[i, j, k] = u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # = u_mat[i, j] * (v_mat @ w_mat)[j, i] ... no, w_mat is (p, n)
            # Let's just do it directly for small sizes
            
            for i in range(n):
                for k in range(p):
                    w_val = w_mat[k, i]
                    if w_val == 0:
                        continue
                    # T[i, :, k] += w_val * u_mat[i, :] * v_mat[:, k]
                    # u_mat[i, :] is (m,), v_mat[:, k] is (m,)
                    # product is (m,)
                    T_computed[i, :, k] += w_val * (u_mat[i, :] * v_mat[:, k])
        
        T_target = np.ones((n, m, p), dtype=float)
        
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True
    except Exception:
        return False

def construct():
    n, m, p = DIMENSIONS
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
