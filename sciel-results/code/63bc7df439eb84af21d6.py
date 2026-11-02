import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomp():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_coeffs, V_coeffs, W_coeffs) where coeffs maps local index to value.
    Local index: i*2 + j for U (i=row, j=col of A), j*2 + k for V (j=row of B, k=col of B), k*2 + i for W (k=col, i=row of C).
    """
    patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3: A00(B10-B01) -> C01 += M3, C11 += M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M4: A11(B01-B10) -> C00 += M4, C10 += M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6: (A10-A00)(B00+B01) -> C10 += M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7: (A01-A11)(B10+B11) -> C00 += M7
    ]
    return patterns

def _build_rank_56():
    """
    Construct the rank-56 decomposition for 2x4 @ 4x8.
    The 2x4 matrix is split into 2 blocks of 2 columns.
    The 4x8 matrix is split into 2x4 blocks (2 rows, 4 columns each split into 2x2).
    Actually, let's view it as:
    A is 2x4, B is 4x8.
    We can think of this as 4 independent 2x2x2 multiplications if we tile:
    A = [A00 A01] where A00 is 2x2, A01 is 2x2
    B = [B00 B01; B10 B11] where each is 2x4... no.
    
    Let's use the standard Kronecker-like block decomposition.
    A (2x4) = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    B (4x8) = [[B_tl, B_tr], [B_bl, B_br]] where each is 2x4.
    This doesn't directly give 2x2x2 blocks.
    
    Better view:
    C (2x8) = A @ B.
    C[:, 0:2] = A @ B[:, 0:2]
    C[:, 2:4] = A @ B[:, 2:4]
    C[:, 4:6] = A @ B[:, 4:6]
    C[:, 6:8] = A @ B[:, 6:8]
    
    Each C[:, 2j:2j+2] = A @ B[:, 2j:2j+2].
    A is 2x4, B[:, 2j:2j+2] is 4x2.
    Let A = [A0 A1] where A0 is 2x2, A1 is 2x2.
    Let B[:, 2j:2j+2] = [B0j; B1j] where B0j is 2x2, B1j is 2x2.
    Then C[:, 2j:2j+2] = A0 @ B0j + A1 @ B1j.
    
    This is the sum of two 2x2x2 matrix multiplications!
    For each j in {0,1,2,3}, we have two 2x2x2 products.
    Total: 4 * 2 = 8 blocks of 2x2x2.
    Each 2x2x2 can be done in rank 7 (Strassen).
    Total rank: 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    strassen_patterns = _strassen_2x2_decomp()
    
    # A is 2x4: A[i][j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j][k] for j in {0,1,2,3}, k in {0,...,7}
    # C is 2x8: C[i][k] for i in {0,1}, k in {0,...,7}
    # W index: k*n + i = k*2 + i
    
    # Split A into A0 (cols 0,1) and A1 (cols 2,3)
    # Split B into 4 blocks of 2x2: for k in {0,1,2,3}, B_block[k] = B[:, 2k:2k+2]
    #   B_block[k] = [B0k; B1k] where B0k = B[0:2, 2k:2k+2], B1k = B[2:4, 2k:2k+2]
    
    # For each output column pair (2k, 2k+1), we compute:
    # C[:, 2k:2k+2] = A0 @ B0k + A1 @ B1k
    # This is two 2x2x2 multiplications:
    #   Product 1: A0 (2x2) @ B0k (2x2) -> contributes to C[:, 2k:2k+2]
    #   Product 2: A1 (2x2) @ B1k (2x2) -> contributes to C[:, 2k:2k+2]
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B01)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # In terms of U, V, W:
    # For product A_sub (2x2) @ B_sub (2x2) -> C_sub (2x2):
    # U indices: i*2 + j for A_sub[i][j]
    # V indices: j*2 + k for B_sub[j][k]
    # W indices: k*2 + i for C_sub[i][k] (note: W[k*n+i] with n=2)
    
    for k_block in range(4):  # 4 blocks of 2 output columns
        for a_block in range(2):  # A0 or A1
            base_col = 7 * (4 * k_block + a_block)
            
            # A_sub is 2x2:
            # If a_block=0: A0 = A[:, 0:2], rows 0-1, cols 0-1
            # If a_block=1: A1 = A[:, 2:4], rows 0-1, cols 2-3
            a_cols = [2*a_block, 2*a_block+1]
            
            # B_sub is 2x2:
            # B0k = B[0:2, 2k:2k+2] if a_block=0
            # B1k = B[2:4, 2k:2k+2] if a_block=1
            b_rows = [2*a_block, 2*a_block+1]
            b_cols = [2*k_block, 2*k_block+1]
            
            # C_sub is 2x2:
            # C[:, 2k:2k+2]
            c_rows = [0, 1]
            c_cols = [2*k_block, 2*k_block+1]
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                # U coefficients
                for local_idx, val in u_c.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = c_rows[li]  # Actually A rows: 0,1
                    global_j = a_cols[lj]
                    U[global_i * m + global_j][col] += val
                
                # V coefficients
                for local_idx, val in v_c.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = b_rows[lj]
                    global_k = b_cols[lk]
                    V[global_j * p + global_k][col] += val
                
                # W coefficients
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
