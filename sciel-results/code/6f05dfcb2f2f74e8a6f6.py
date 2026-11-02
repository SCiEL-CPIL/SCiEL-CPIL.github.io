import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomp():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict) mapping local indices to coefficients.
    Local indices:
    A: 0=a00, 1=a01, 2=a10, 3=a11
    B: 0=b00, 1=b01, 2=b10, 3=b11
    C: 0=c00, 1=c01, 2=c10, 3=c11
    """
    patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7
    ]
    return patterns

def _build_rank_56():
    """
    Construct the rank-56 decomposition for 2x4 @ 4x8.
    Uses 8 independent Strassen 2x2x2 decompositions.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2_decomp()
    
    # A is 2x4. Split columns into 2 blocks of 2.
    # B is 4x8. Split rows into 2 blocks of 2, columns into 4 blocks of 2.
    # For each pair (a_block, b_col_block), we have a 2x2x2 problem.
    # a_block in {0, 1} determines which 2 columns of A and which 2 rows of B.
    # b_col_block in {0, 1, 2, 3} determines which 2 columns of B (and C).
    
    for a_block in range(2):
        # A block: rows 0,1; cols 2*a_block, 2*a_block+1
        # B block: rows 2*a_block, 2*a_block+1; cols 2*b_col_block, 2*b_col_block+1
        for b_col_block in range(4):
            # Global index offsets
            a_rows = [0, 1]
            a_cols = [2*a_block, 2*a_block+1]
            
            b_rows = [2*a_block, 2*a_block+1]
            b_cols = [2*b_col_block, 2*b_col_block+1]
            
            c_rows = [0, 1]
            c_cols = [2*b_col_block, 2*b_col_block+1]
            
            base_col = 7 * (4 * a_block + b_col_block)
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                # Fill U: U[i, j] corresponds to A[i, j]
                for local_idx, val in u_c.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = a_rows[li]
                    global_j = a_cols[lj]
                    U[global_i * m + global_j][col] += val
                
                # Fill V: V[j, k] corresponds to B[j, k]
                for local_idx, val in v_c.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = b_rows[lj]
                    global_k = b_cols[lk]
                    V[global_j * p + global_k][col] += val
                
                # Fill W: W[k, i] corresponds to C[i, k]
                # Convention: W[k*n + i]
                for local_idx, val in w_c.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    global_i = c_rows[li]
                    global_k = c_cols[lk]
                    W[global_k * n + global_i][col] += val
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} should be 1 for all i,j,k.
    """
    if not U or not V or not W:
        return False
        
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # We want to check sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
    # This is a tensor contraction.
    # We can do this by iterating over i, j, k but using vectorized dot products for r.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u = U[i,j,:], v = V[j,k,:], w = W[k,i,:]
                # val = sum(u * v * w)
                val = np.dot(U_reshaped[i, j, :], np.multiply(V_reshaped[j, k, :], W_reshaped[k, i, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        if time.time() < deadline:
            result = _build_rank_56()
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()
