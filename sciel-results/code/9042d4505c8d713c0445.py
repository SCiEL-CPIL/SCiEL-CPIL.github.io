import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # 1. Generate the Schoolbook Decomposition (rank 45)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_sb[i*m+j, :]
                v_val = V_sb[j*p+k, :]
                w_val = W_sb[k*n+i, :]
                s = np.sum(u_vec * v_val * w_val)
                if not np.isclose(s, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find rank-44 decomposition
    # Strategy: Fix U and V to be 9x44 and 15x44, solve for W
    # Use schoolbook as starting point, remove one column, solve for W
    
    def try_rank44():
        if time.time() - start_time > deadline - 5:
            return None
        
        r = 44
        # Start with schoolbook, remove last column
        U = U_sb[:, :r].copy()
        V = V_sb[:, :r].copy()
        
        # Build linear system for W
        # Variables: W has p*n*r = 15*44 = 660 variables
        # Equations: one for each (i,j,k), total 45 equations
        # For equation (i,j,k): sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        
        num_eq = n * m * p  # 45
        num_var = p * n * r  # 15 * 44 = 660
        
        A = np.zeros((num_eq, num_var))
        b = np.ones(num_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        var_idx = (k * n + i) * r + c
                        A[eq_idx, var_idx] = U[i*m+j, c] * V[j*p+k, c]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape(p*n, r)
        except:
            return None
        
        # Round to nearest half-integer and verify
        W_rounded = np.round(W * 2) / 2
        
        # Verify decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W_rounded[k*n+i, :])
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return None
        
        return U, V, W_rounded
    
    result = try_rank44()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
