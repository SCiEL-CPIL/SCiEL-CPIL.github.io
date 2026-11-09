"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the matrices U, V, W.
    # Estimated max rank: 54
    # Let's allocate a bit more just in case, then trim.
    max_rank = 60
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    col = 0

    def add_strassen_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds the Strassen decomposition for a 2x2 matrix multiplication.
        A is 2x2 at A[a_row_offset:a_row_offset+2, a_col_offset:a_col_offset+2]
        B is 2x2 at B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        C is 2x2 at C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        
        Returns the number of columns used.
        """
        nonlocal col
        
        # Indices in global U, V, W
        # U index: i * m + j
        # V index: j * p + k
        # W index: k * n + i
        
        # Local A indices (i, j) -> Global U index (a_row_offset + i) * m + (a_col_offset + j)
        # Local B indices (j, k) -> Global V index (b_row_offset + j) * p + (b_col_offset + k)
        # Local C indices (i, k) -> Global W index (c_col_offset + k) * n + (c_row_offset + i)
        
        def get_u(i, j):
            return (a_row_offset + i) * m + (a_col_offset + j)
            
        def get_v(j, k):
            return (b_row_offset + j) * p + (b_col_offset + k)
            
        def get_w(i, k):
            return (c_col_offset + k) * n + (c_row_offset + i)

        # Strassen Terms
        # P1 = (A00 + A11) * (B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=-1
        U[get_u(0,0), col] += 1
        U[get_u(1,1), col] += 1
        V[get_v(0,0), col] += 1
        V[get_v(1,1), col] += 1
        W[get_w(0,0), col] += 1
        W[get_w(1,1), col] += -1
        col += 1

        # P2 = (A00 + A11) * B11
        # U: A00=1, A11=1
        # V: B11=1
        # W: C00=-1
        U[get_u(0,0), col] += 1
        U[get_u(1,1), col] += 1
        V[get_v(1,1), col] += 1
        W[get_w(0,0), col] += -1
        col += 1

        # P3 = A00 * (B00 - B10)
        # U: A00=1
        # V: B00=1, B10=-1
        # W: C00=1, C10=1
        U[get_u(0,0), col] += 1
        V[get_v(0,0), col] += 1
        V[get_v(1,0), col] += -1
        W[get_w(0,0), col] += 1
        W[get_w(1,0), col] += 1
        col += 1

        # P4 = A11 * (B11 - B01)
        # U: A11=1
        # V: B11=1, B01=-1
        # W: C01=1, C11=1
        U[get_u(1,1), col] += 1
        V[get_v(1,1), col] += 1
        V[get_v(0,1), col] += -1
        W[get_w(0,1), col] += 1
        W[get_w(1,1), col] += 1
        col += 1

        # P5 = (A00 + A01) * B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C01=1, C11=-1
        U[get_u(0,0), col] += 1
        U[get_u(0,1), col] += 1
        V[get_v(1,1), col] += 1
        W[get_w(0,1), col] += 1
        W[get_w(1,1), col] += -1
        col += 1

        # P6 = (A10 - A00) * (B00 + B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C00=1, C11=1
        U[get_u(1,0), col] += 1
        U[get_u(0,0), col] += -1
        V[get_v(0,0), col] += 1
        V[get_v(0,1), col] += 1
        W[get_w(0,0), col] += 1
        W[get_w(1,1), col] += 1
        col += 1

        # P7 = (A01 - A11) * (B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C10=1
        U[get_u(0,1), col] += 1
        U[get_u(1,1), col] += -1
        V[get_v(1,0), col] += 1
        V[get_v(1,1), col] += 1
        W[get_w(1,0), col] += 1
        col += 1

        return col

    def add_schoolbook_block(a_row_offset, a_col_offset, a_rows, a_cols, b_row_offset, b_col_offset, b_rows, b_cols, c_row_offset, c_col_offset):
        """
        Adds schoolbook decomposition for a rectangular block.
        A is a_rows x a_cols
        B is b_rows x b_cols (where a_cols == b_rows)
        C is a_rows x b_cols
        """
        nonlocal col
        for i in range(a_rows):
            for j in range(a_cols):
                for k in range(b_cols):
                    u_idx = (a_row_offset + i) * m + (a_col_offset + j)
                    v_idx = (b_row_offset + j) * p + (b_col_offset + k)
                    w_idx = (c_col_offset + k) * n + (c_row_offset + i)
                    
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return col

    # Partition M=5 into 2, 2, 1
    # Partition P=6 into 2, 2, 2
    # N=2 is single block

    # Block A1 (rows 0-1, cols 0-1)
    # Block A2 (rows 0-1, cols 2-3)
    # Block A3 (rows 0-1, col 4)

    # Block B1 (rows 0-1, cols 0-5) -> Split into B11 (cols 0-1), B12 (cols 2-3), B13 (cols 4-5)
    # Block B2 (rows 2-3, cols 0-5) -> Split into B21 (cols 0-1), B22 (cols 2-3), B23 (cols 4-5)
    # Block B3 (row 4, cols 0-5)   -> Split into B31 (cols 0-1), B32 (cols 2-3), B33 (cols 4-5)

    # 1. A1 * B11 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    # 2. A1 * B12 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    # 3. A1 * B13 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 0, 0, 4, 0, 4)
    
    # 4. A2 * B21 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    # 5. A2 * B22 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    # 6. A2 * B23 (2x2 * 2x2) -> Strassen
    add_strassen_2x2x2(0, 2, 2, 4, 0, 4)
    
    # 7. A3 * B31 (2x1 * 1x2) -> Schoolbook (Rank 4)
    add_schoolbook_block(0, 4, 2, 1, 4, 0, 1, 2, 0, 0)
    # 8. A3 * B32 (2x1 * 1x2) -> Schoolbook (Rank 4)
    add_schoolbook_block(0, 4, 2, 1, 4, 2, 1, 2, 0, 2)
    # 9. A3 * B33 (2x1 * 1x2) -> Schoolbook (Rank 4)
    add_schoolbook_block(0, 4, 2, 1, 4, 4, 1, 2, 0, 4)
    
    # Trim matrices to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
