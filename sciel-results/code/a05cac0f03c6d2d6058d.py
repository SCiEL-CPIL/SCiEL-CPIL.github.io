import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    dim_u = n * m  # 8
    dim_v = m * p  # 20
    dim_w = p * n  # 10
    
    # Total rank is 36
    rank_total = 36
    U = np.zeros((dim_u, rank_total))
    V = np.zeros((dim_v, rank_total))
    W = np.zeros((dim_w, rank_total))
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_2x2x2(a_row_start, b_row_start, c_col_start):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        A block is A[a_row_start:a_row_start+2, :]
        B block is B[:, b_row_start:b_row_start+2] -> No, B is 4x5.
        The block structure is:
        A is 2x4. Split into A0 (cols 0-1) and A1 (cols 2-3).
        B is 4x5. Split into B0 (rows 0-1) and B1 (rows 2-3).
        
        Term 1: A0 * B0. A0 is 2x2 (rows 0-1 of A, cols 0-1). B0 is 2x5 (rows 0-1 of B).
        We split B0 into B0_0 (cols 0-1, 2x2) and B0_1 (cols 2-4, 2x3).
        
        So this function handles A (2x2) * B (2x2).
        a_row_start: start row of A in the global 2x4? No, A is fixed 2 rows.
        The 'block' is defined by which columns of A and rows of B.
        
        Let's define the parameters as:
        a_cols: tuple of 2 col indices for A (e.g., (0,1))
        b_rows: tuple of 2 row indices for B (e.g., (0,1))
        c_cols: tuple of 2 col indices for C (e.g., (0,1))
        """
        nonlocal col
        
        a_c0, a_c1 = a_cols
        b_r0, b_r1 = b_rows
        c_c0, c_c1 = c_cols
        
        # Strassen 7 products
        # M1 = (A00 + A11) * (B00 + B11)
        U[u_idx(0, a_c0), col] += 1
        U[u_idx(1, a_c1), col] += 1
        V[v_idx(b_r0, c_c0), col] += 1
        V[v_idx(b_r1, c_c1), col] += 1
        W[w_idx(c_c0, 0), col] += 1
        W[w_idx(c_c1, 1), col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[u_idx(1, a_c0), col] += 1
        U[u_idx(1, a_c1), col] += 1
        V[v_idx(b_r0, c_c0), col] += 1
        W[w_idx(c_c0, 1), col] += 1
        W[w_idx(c_c1, 1), col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        U[u_idx(0, a_c0), col] += 1
        V[v_idx(b_r0, c_c1), col] += 1
        V[v_idx(b_r1, c_c1), col] -= 1
        W[w_idx(c_c1, 0), col] += 1
        W[w_idx(c_c1, 1), col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        U[u_idx(1, a_c1), col] += 1
        V[v_idx(b_r1, c_c0), col] += 1
        V[v_idx(b_r0, c_c0), col] -= 1
        W[w_idx(c_c0, 0), col] += 1
        W[w_idx(c_c1, 0), col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[u_idx(0, a_c0), col] += 1
        U[u_idx(0, a_c1), col] += 1
        V[v_idx(b_r1, c_c1), col] += 1
        W[w_idx(c_c0, 0), col] -= 1
        W[w_idx(c_c1, 0), col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[u_idx(1, a_c0), col] += 1
        U[u_idx(0, a_c0), col] -= 1
        V[v_idx(b_r0, c_c0), col] += 1
        V[v_idx(b_r0, c_c1), col] += 1
        W[w_idx(c_c1, 1), col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[u_idx(0, a_c1), col] += 1
        U[u_idx(1, a_c1), col] -= 1
        V[v_idx(b_r1, c_c0), col] += 1
        V[v_idx(b_r1, c_c1), col] += 1
        W[w_idx(c_c0, 1), col] += 1
        col += 1
        
    def add_trivial_2x2x1(a_cols, b_row, c_col):
        """
        Adds a rank-4 trivial decomposition for A (2x2) * B (2x1) -> C (2x1).
        a_cols: tuple of 2 col indices for A
        b_row: row index for B
        c_col: col index for C
        """
        nonlocal col
        a_c0, a_c1 = a_cols
        
        # A00 * B0
        U[u_idx(0, a_c0), col] += 1
        V[v_idx(b_row, c_col), col] += 1
        W[w_idx(c_col, 0), col] += 1
        col += 1
        
        # A01 * B1
        U[u_idx(0, a_c1), col] += 1
        V[v_idx(b_row + 1, c_col), col] += 1
        W[w_idx(c_col, 0), col] += 1
        col += 1
        
        # A10 * B0
        U[u_idx(1, a_c0), col] += 1
        V[v_idx(b_row, c_col), col] += 1
        W[w_idx(c_col, 1), col] += 1
        col += 1
        
        # A11 * B1
        U[u_idx(1, a_c1), col] += 1
        V[v_idx(b_row + 1, c_col), col] += 1
        W[w_idx(c_col, 1), col] += 1
        col += 1

    # Decomposition Plan:
    # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
    # B is 4x5. Split into B0 (rows 0,1) and B1 (rows 2,3).
    # C is 2x5.
    
    # Term 1: A0 * B0
    # B0 is 2x5. Split into B0_0 (cols 0,1) and B0_1 (cols 2,3,4).
    # A0 * B0_0 is 2x2 * 2x2. Rank 7.
    add_strassen_2x2x2((0, 1), (0, 1), (0, 1))
    
    # A0 * B0_1 is 2x2 * 2x3. Rank 11.
    # Split B0_1 into B0_1_0 (col 2) and B0_1_1 (cols 3,4)?
    # My add_trivial_2x2x1 handles 2x1.
    # So we call it 3 times for cols 2, 3, 4.
    add_trivial_2x2x1((0, 1), 0, 2)
    add_trivial_2x2x1((0, 1), 0, 3)
    add_trivial_2x2x1((0, 1), 0, 4)
    
    # Term 2: A1 * B1
    # B1 is 2x5. Split into B1_0 (cols 0,1) and B1_1 (cols 2,3,4).
    # A1 * B1_0 is 2x2 * 2x2. Rank 7.
    add_strassen_2x2x2((2, 3), (2, 3), (0, 1))
    
    # A1 * B1_1 is 2x2 * 2x3. Rank 11.
    add_trivial_2x2x1((2, 3), 2, 2)
    add_trivial_2x2x1((2, 3), 2, 3)
    add_trivial_2x2x1((2, 3), 2, 4)
    
    assert col == rank_total, f"Expected rank {rank_total}, got {col}"
    
    # Verify
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[u_idx(i, j), v_idx(j, k), w_idx(k, i)] = 1.0
                
    if not np.allclose(T, T_decomp, atol=1e-10):
        # Fallback to trivial rank 40
        rank_t = 40
        U_t = np.zeros((dim_u, rank_t))
        V_t = np.zeros((dim_v, rank_t))
        W_t = np.zeros((dim_w, rank_t))
        col_t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_t[u_idx(i,j), col_t] = 1.0
                    V_t[v_idx(j,k), col_t] = 1.0
                    W_t[w_idx(k,i), col_t] = 1.0
                    col_t += 1
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
