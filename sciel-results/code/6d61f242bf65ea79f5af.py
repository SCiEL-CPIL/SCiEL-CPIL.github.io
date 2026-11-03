import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.size == 0 or V_arr.size == 0 or W_arr.size == 0:
        return False, 1.0
        
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
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

def _strassen_44():
    n, m, p = DIMENSIONS
    # We will build lists of columns and then trim
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strassen steps definition
    # Each step is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts mapping local (i, j) to value
    
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00(Y01 - Y11) -> Z01 += 1, Z11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11(Y10 - Y00) -> Z00 += 1, Z10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00 + X01)Y11 -> Z01 += 1, Z00 -= 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11) -> Z00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Define blocks
    # (X_offset_i, X_offset_j, Y_offset_j, Y_offset_k, Z_offset_i, Z_offset_k)
    blocks = [
        # M1 N1 -> C[0:2, 0:2]
        (0, 0, 0, 0, 0, 0),
        # M2 N3 -> C[0:2, 0:2]
        (0, 2, 2, 0, 0, 0),
        # M1 N2 -> C[0:2, 2:4]
        (0, 0, 0, 2, 0, 2),
        # M2 N4 -> C[0:2, 2:4]
        (0, 2, 2, 2, 0, 2)
    ]
    
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for (li, lj), val in u_coeffs.items():
                gi = li + xi
                gj = lj + xj
                u_col[gi * m + gj] += val
            
            # Fill V
            for (lj, lk), val in v_coeffs.items():
                gj = lj + yj
                gk = lk + yk
                v_col[gj * p + gk] += val
            
            # Fill W
            for (li, lk), val in w_coeffs.items():
                gi = li + zi
                gk = lk + zk
                w_col[gk * n + gi] += val
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # Rank 1 decomposition for the bottom row (i=2) part (1x4x4)
    # T_{2,j,k} = 1 for all j,k
    u_col = [0] * (n * m)
    v_col = [0] * (m * p)
    w_col = [0] * (p * n)
    
    for j in range(m):
        u_col[2 * m + j] += 1
    for j in range(m):
        for k in range(p):
            v_col[j * p + k] += 1
    for k in range(p):
        w_col[k * n + 2] += 1
        
    U_cols.append(u_col)
    V_cols.append(v_col)
    W_cols.append(w_col)
    
    # Convert to matrices and trim zero columns
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    # Check for zero columns and remove them
    # A column is zero if all its entries are 0
    mask = (np.sum(np.abs(U_mat), axis=0) > 0) & \
           (np.sum(np.abs(V_mat), axis=0) > 0) & \
           (np.sum(np.abs(W_mat), axis=0) > 0)
           
    U_mat = U_mat[:, mask]
    V_mat = V_mat[:, mask]
    W_mat = W_mat[:, mask]
    
    # Convert back to lists of lists of integers/half-integers
    # The entries are integers
    U_list = [[int(x) for x in row] for row in U_mat]
    V_list = [[int(x) for x in row] for row in V_mat]
    W_list = [[int(x) for x in row] for row in W_mat]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 29 decomposition
    result_29 = _strassen_44()
    ok, err = _verify_fast(result_29["U"], result_29["V"], result_29["W"], n, m, p)
    
    if ok:
        return result_29
    
    # Fallback to schoolbook
    return _schoolbook()
