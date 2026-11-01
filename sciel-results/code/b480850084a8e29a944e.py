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
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    patterns = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7
    ]
    return patterns

def _build_rank_56():
    """
    Construct the rank-56 decomposition for 2x4 @ 4x8.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2_decomp()
    
    # A is 2x4. Split into 2 blocks of 2x2 (columns 0-1, 2-3).
    # B is 4x8. Split into 4 blocks of 2x2 (rows 0-1/2-3, cols 0-1/2-3/4-5/6-7).
    # Actually, B is split by rows into 2 blocks of 2x8, and each 2x8 into 4 blocks of 2x2.
    # So 2 (A blocks) * 4 (B blocks) = 8 Strassen instances.
    
    for a_block in range(2):
        # A block columns: 2*a_block, 2*a_block+1
        # A rows: 0, 1
        for b_block in range(4):
            # B block rows: 2*(b_block//2) ... wait.
            # B is 4x8.
            # Let's index B blocks by (row_block, col_block).
            # row_block in {0, 1} -> rows 0-1, 2-3.
            # col_block in {0, 1, 2, 3} -> cols 0-1, 2-3, 4-5, 6-7.
            # We iterate a_block in {0,1} and b_block in {0,1,2,3}.
            # But we need to match A columns with B rows.
            # A[a_block] has columns 2*a_block, 2*a_block+1.
            # B must have rows 2*a_block, 2*a_block+1 to multiply with A[a_block].
            # So for a given a_block, we only use B rows corresponding to that a_block.
            # B rows are 0,1,2,3.
            # If a_block=0, use B rows 0,1.
            # If a_block=1, use B rows 2,3.
            # So b_block should represent the column block of B, and the row block is fixed by a_block.
            
            row_block = a_block
            col_block = b_block
            
            # Global indices for A block
            a_row_offsets = [0, 1] # i=0,1
            a_col_offsets = [2*a_block, 2*a_block+1] # j=2a, 2a+1
            
            # Global indices for B block
            b_row_offsets = [2*row_block, 2*row_block+1] # j=2a, 2a+1
            b_col_offsets = [2*col_block, 2*col_block+1] # k=2c, 2c+1
            
            # Global indices for C block
            c_row_offsets = [0, 1] # i=0,1
            c_col_offsets = [2*col_block, 2*col_block+1] # k=2c, 2c+1
            
            base_col = 7 * (4 * a_block + b_block)
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                # Fill U
                for local_idx, val in u_c.items():
                    # local_idx 0..3 -> (li, lj)
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = c_row_offsets[li] # A row
                    global_j = a_col_offsets[lj] # A col
                    U[global_i * m + global_j][col] += val
                
                # Fill V
                for local_idx, val in v_c.items():
                    # local_idx 0..3 -> (lj, lk)
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = b_row_offsets[lj] # B row
                    global_k = b_col_offsets[lk] # B col
                    V[global_j * p + global_k][col] += val
                
                # Fill W
                for local_idx, val in w_c.items():
                    # local_idx 0..3 -> (li, lk)
                    li = local_idx // 2
                    lk = local_idx % 2
                    global_i = c_row_offsets[li] # C row
                    global_k = c_col_offsets[lk] # C col
                    W[global_k * n + global_i][col] += val
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} should be 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
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
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        if time.time() < deadline:
            result = _build_rank_56()
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()
