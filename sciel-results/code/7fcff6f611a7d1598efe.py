import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 120s limit, 20s safety margin
    
    n, m, p = 2, 4, 5
    rank = 36
    
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m) x rank = 8x36
    # V: (m*p) x rank = 20x36
    # W: (p*n) x rank = 10x36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to U, V, W
    # row_u, row_v, row_w, col_index, val_u, val_v, val_w
    def add_term(u_idx, v_idx, w_idx, c_idx, vu, vv, vw):
        U[u_idx][c_idx] += vu
        V[v_idx][c_idx] += vv
        W[w_idx][c_idx] += vw

    # Pre-compute indices for the 4 Strassen blocks and 2 Schoolbook blocks
    # We will explicitly list the 36 rank-1 tensors
    
    # --- Strassen Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] ---
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in U (size 8, m=4, so row*4+col)
    # A00=0, A01=1, A10=4, A11=5
    # B00=0, B01=1, B10=2, B11=3 in V (size 20, p=5, so row*5+col)
    # B00=0, B01=1, B10=5, B11=6
    # C indices in W (size 10, p*n=10, col*n+row? No, task says W[k*n+i, r] where k is col? 
    # Task: W[k*n+i, r]. Usually C is 2x5. k=0..4 (col), i=0..1 (row).
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[0,1] -> k=0, i=1 -> idx 1
    # C[1,0] -> k=1, i=0 -> idx 5
    # C[1,1] -> k=1, i=1 -> idx 6
    # C[2,0] -> k=2, i=0 -> idx 10
    # C[2,1] -> k=2, i=1 -> idx 11
    # C[3,0] -> k=3, i=0 -> idx 15
    # C[3,1] -> k=3, i=1 -> idx 16
    # C[4,0] -> k=4, i=0 -> idx 20
    # C[4,1] -> k=4, i=1 -> idx 21
    
    # Let's map indices for Block 1 (Cols 0,1 of C; Rows 0,1 of A; Rows 0,1 of B)
    # A: 0,1,4,5
    # B: 0,1,5,6
    # C: 0,1,5,6
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 0, 1, 1, 1)
    add_term(5, 6, 1, 0, 1, 1, 1)
    add_term(5, 6, 5, 0, 1, 1, 1)
    add_term(0, 0, 6, 0, 1, 1, 1)
    
    # Wait, standard Strassen adds to specific C positions.
    # Let's use the explicit formula from the previous code to ensure correctness, 
    # but structure it as a list of operations.
    
    # Clear previous partial adds for clarity in static list approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_row, a_col, b_row, b_col, c_row, c_col, offset):
        """
        Adds 7 rank-1 terms for a 2x2 * 2x2 block.
        A is 2x4, so a_row, a_col define the 2x2 window.
        B is 4x5, so b_row, b_col define the 2x2 window.
        C is 2x5, so c_row, c_col define the 2x2 window in C (c_row is always 0 here, c_col is col start).
        Note: c_row in task description is 0..1. c_col is 0..4.
        W index is c_col * n + c_row.
        """
        # Local indices
        # U indices (row * 4 + col)
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        # V indices (row * 5 + col)
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        # W indices (col * 2 + row)
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + (c_row + 1)
        w11 = (c_col + 1) * n + (c_row + 1)
        
        # M1: (A00 + A11) * (B00 + B11) -> C00 + C11
        U[u00][offset] += 1; U[u11][offset] += 1
        V[v00][offset] += 1; V[v11][offset] += 1
        W[w00][offset] += 1; W[w11][offset] += 1
        
        # M2: (A10 + A11) * B00 -> C10 - C11
        U[u10][offset+1] += 1; U[u11][offset+1] += 1
        V[v00][offset+1] += 1
        W[w10][offset+1] += 1; W[w11][offset+1] -= 1
        
        # M3: A00 * (B01 - B11) -> C01 + C11
        U[u00][offset+2] += 1
        V[v01][offset+2] += 1; V[v11][offset+2] -= 1
        W[w01][offset+2] += 1; W[w11][offset+2] += 1
        
        # M4: A11 * (B10 - B00) -> C00 + C10
        U[u11][offset+3] += 1
        V[v10][offset+3] += 1; V[v00][offset+3] -= 1
        W[w00][offset+3] += 1; W[w10][offset+3] += 1
        
        # M5: (A00 + A01) * B11 -> -C00 + C01
        U[u00][offset+4] += 1; U[u01][offset+4] += 1
        V[v11][offset+4] += 1
        W[w00][offset+4] -= 1; W[w01][offset+4] += 1
        
        # M6: (A10 - A00) * (B00 + B01) -> C11
        U[u10][offset+5] += 1; U[u00][offset+5] -= 1
        V[v00][offset+5] += 1; V[v01][offset+5] += 1
        W[w11][offset+5] += 1
        
        # M7: (A01 - A11) * (B10 + B11) -> C00
        U[u01][offset+6] += 1; U[u11][offset+6] -= 1
        V[v10][offset+6] += 1; V[v11][offset+6] += 1
        W[w00][offset+6] += 1

    def add_schoolbook_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col, offset):
        """
        Adds 4 rank-1 terms for a 2x2 * 2x1 block.
        A is 2x2, B is 2x1, C is 2x1.
        """
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + (c_row + 1)
        
        # C0 = A00*B0 + A01*B1
        U[u00][offset] += 1; V[v0][offset] += 1; W[w0][offset] += 1
        U[u01][offset+1] += 1; V[v1][offset+1] += 1; W[w0][offset+1] += 1
        # C1 = A10*B0 + A11*B1
        U[u10][offset+2] += 1; V[v0][offset+2] += 1; W[w1][offset+2] += 1
        U[u11][offset+3] += 1; V[v1][offset+3] += 1; W[w1][offset+3] += 1

    # Block 1: Cols 0-1 of C. A Cols 0-1, B Rows 0-1 Cols 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: Cols 0-1 of C. A Cols 2-3, B Rows 2-3 Cols 0-1
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    
    # Block 3: Cols 2-3 of C. A Cols 0-1, B Rows 0-1 Cols 2-3
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    
    # Block 4: Cols 2-3 of C. A Cols 2-3, B Rows 2-3 Cols 2-3
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    # Block 5: Col 4 of C. A Cols 0-1, B Rows 0-1 Col 4
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4, 28)
    
    # Block 6: Col 4 of C. A Cols 2-3, B Rows 2-3 Col 4
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4, 32)
    
    return {"U": U, "V": V, "W": W}
