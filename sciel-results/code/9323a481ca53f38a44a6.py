import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def _build_strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2(U_rows, V_rows, W_rows, offset):
        U_e = []
        V_e = []
        W_e = []
        col = offset
        
        # M1 = (A00+A11)*(B00+B11)
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], -1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M2 = (A10+A11)*B00
        U_e.append((U_rows[2], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M3 = A00*(B10-B11)
        U_e.append((U_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        V_e.append((V_rows[3], -1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[2], 1, col))
        col += 1
        
        # M4 = A11*(B01-B10)
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[1], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M5 = (A00+A10)*B11
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[2], -1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        W_e.append((W_rows[0], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[3], -1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        return U_e, V_e, W_e, col

    U_e1, V_e1, W_e1, _ = strassen_2x2([0,1,2,3], [0,1,2,3], [0,1,2,3], 0)
    U_e2, V_e2, W_e2, _ = strassen_2x2([4,5,6,7], [8,9,10,11], [0,1,2,3], 7)
    U_e3, V_e3, W_e3, _ = strassen_2x2([0,1,2,3], [12,13,14,15], [4,5,6,7], 14)
    U_e4, V_e4, W_e4, _ = strassen_2x2([4,5,6,7], [12,13,14,15], [4,5,6,7], 21)
    
    for row, val, col in U_e1 + U_e2 + U_e3 + U_e4:
        U[row][col] += val
    for row, val, col in V_e1 + V_e2 + V_e3 + V_e4:
        V[row][col] += val
    for row, val, col in W_e1 + W_e2 + W_e3 + W_e4:
        W[row][col] += val
        
    return U, V, W

def _tensor_error(params, n, m, p, rank):
    """Calculate the error of the tensor decomposition."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U_flat = params[:rank * size_u]
    V_flat = params[rank * size_u : rank * (size_u + size_v)]
    W_flat = params[rank * (size_u + size_v) :]
    
    U = U_flat.reshape(rank, size_u).T
    V = V_flat.reshape(rank, size_v).T
    W = W_flat.reshape(rank, size_w).T
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _search_rank_27(U28, V28, W28, max_time=60.0):
    """Search for a rank-27 decomposition using optimization."""
    start_time = time.time()
    n, m, p = 2, 4, 4
    rank = 27
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Convert to NumPy arrays to allow proper slicing and operations
    U28_np = np.array(U28, dtype=float)
    V28_np = np.array(V28, dtype=float)
    W28_np = np.array(W28, dtype=float)
    
    # Build the matrix of tensor entries for each column.
    # Each column r defines a tensor T_r = U_r \otimes V_r \otimes W_r.
    # We want to see if T_27 is a linear combination of T_0...T_26.
    
    tensors = []
    for r in range(28):
        t = np.zeros(32)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Indexing NumPy arrays
                    t[i*m*p + j*p + k] = U28_np[i*m+j, r] * V28_np[j*p+k, r] * W28_np[k*n+i, r]
        tensors.append(t)
        
    T_mat = np.array(tensors).T # Shape (32, 28)
    
    # Check if the last column is in the span of the first 27
    try:
        x, residuals, rank_s, sv = np.linalg.lstsq(T_mat[:, :27], T_mat[:, 27], rcond=None)
        if residuals[0] < 1e-9:
            # Linear dependence found.
            # T_27 = sum_{r=0}^{26} x_r T_r
            # T_total = sum_{r=0}^{26} T_r + T_27 = sum_{r=0}^{26} (1 + x_r) T_r
            
            new_U = []
            new_V = []
            new_W = []
            for i in range(size_u):
                row_u = []
                for r in range(27):
                    # Scale U by 1 (no change)
                    row_u.append(U28_np[i, r])
                new_U.append(row_u)
            for i in range(size_v):
                row_v = []
                for r in range(27):
                    # Scale V by 1 (no change)
                    row_v.append(V28_np[i, r])
                new_V.append(row_v)
            for i in range(size_w):
                row_w = []
                for r in range(27):
                    # Scale W by (1 + x_r)
                    row_w.append(W28_np[i, r] * (1 + x[r]))
                new_W.append(row_w)
            
            # Verify
            params_new = np.concatenate([np.array(new_U).flatten(), np.array(new_V).flatten(), np.array(new_W).flatten()])
            err = _tensor_error(params_new, n, m, p, 27)
            if err < 1e-9:
                return new_U, new_V, new_W
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    deadline = 140.0
    
    U, V, W = _build_strassen_28()
    
    # Try to reduce to 27
    if time.time() - start_time < deadline - 10:
        result = _search_rank_27(U, V, W, max_time=10.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}

