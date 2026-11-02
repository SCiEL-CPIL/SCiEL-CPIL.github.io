"""Matrix multiplication 2,3,7 tensor decomposition via Strassen blocks."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal (or low-rank) tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.
    
    Strategy: Split the 2x3 matrix A into two parts:
      - A1: first 2 columns (2x2)
      - A2: last 1 column (2x1)
    
    Correspondingly split B (3x7) into:
      - B1: first 2 rows (2x7)
      - B2: last 1 row (1x7)
    
    Then C = A*B = A1*B1 + A2*B2
    
    We use Strassen for the 2x2 * 2x2 sub-blocks of A1*B1 and trivial for the rest.
    Indices are global row-major for A and B, and W[k*n+i] for C.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_2x2_2x2(A_indices, B_indices, C_indices):
        """
        Add Strassen decomposition.
        A_indices: [A11, A12, A21, A22] - global indices in U
        B_indices: [B11, B12, B21, B22] - global indices in V
        C_indices: [C11, C12, C21, C22] - global indices in W
        """
        products = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,0]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1]),
        ]
        
        for u_c, v_c, w_c in products:
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[A_indices[0]] = u_c[0]
            u_col[A_indices[1]] = u_c[1]
            u_col[A_indices[2]] = u_c[2]
            u_col[A_indices[3]] = u_c[3]
            
            v_col[B_indices[0]] = v_c[0]
            v_col[B_indices[1]] = v_c[1]
            v_col[B_indices[2]] = v_c[2]
            v_col[B_indices[3]] = v_c[3]
            
            w_col[C_indices[0]] = w_c[0]
            w_col[C_indices[1]] = w_c[1]
            w_col[C_indices[2]] = w_c[2]
            w_col[C_indices[3]] = w_c[3]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    def add_trivial_2x2_2x1(A_indices, B_indices, C_indices):
        """
        Add trivial decomposition for 2x2 * 2x1.
        A_indices: [A11, A12, A21, A22]
        B_indices: [B1, B2]
        C_indices: [C1, C2]
        """
        products = [
            ([1,0,0,0], [1,0], [1,0]),
            ([0,1,0,0], [0,1], [1,0]),
            ([0,0,1,0], [1,0], [0,1]),
            ([0,0,0,1], [0,1], [0,1]),
        ]
        
        for u_c, v_c, w_c in products:
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[A_indices[0]] = u_c[0]
            u_col[A_indices[1]] = u_c[1]
            u_col[A_indices[2]] = u_c[2]
            u_col[A_indices[3]] = u_c[3]
            
            v_col[B_indices[0]] = v_c[0]
            v_col[B_indices[1]] = v_c[1]
            
            w_col[C_indices[0]] = w_c[0]
            w_col[C_indices[1]] = w_c[1]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    # Part 1: A1 * B1
    # A1 is 2x2 block from A (cols 0,1)
    # Global A indices:
    # A00=0, A01=1
    # A10=3, A11=4
    A1_indices = [0, 1, 3, 4]
    
    # B1 is 2x7 block from B (rows 0,1)
    # Split into 2x2 blocks (cols 0,1; 2,3; 4,5) and 2x1 block (col 6)
    
    # Block 1: cols 0,1
    # B indices: B00=0, B01=1, B10=7, B11=8
    B1a_indices = [0, 1, 7, 8]
    # C indices for cols 0,1:
    # C00=0, C01=2, C10=1, C11=3
    C1a_indices = [0, 2, 1, 3]
    add_strassen_2x2_2x2(A1_indices, B1a_indices, C1a_indices)
    
    # Block 2: cols 2,3
    # B indices: B02=2, B03=3, B12=9, B13=10
    B1b_indices = [2, 3, 9, 10]
    # C indices for cols 2,3:
    # C02=4, C03=6, C12=5, C13=7
    C1b_indices = [4, 6, 5, 7]
    add_strassen_2x2_2x2(A1_indices, B1b_indices, C1b_indices)
    
    # Block 3: cols 4,5
    # B indices: B04=4, B05=5, B14=11, B15=12
    B1c_indices = [4, 5, 11, 12]
    # C indices for cols 4,5:
    # C04=8, C05=10, C14=9, C15=11
    C1c_indices = [8, 10, 9, 11]
    add_strassen_2x2_2x2(A1_indices, B1c_indices, C1c_indices)
    
    # Block 4: col 6
    # B indices: B06=6, B16=13
    B1d_indices = [6, 13]
    # C indices for col 6:
    # C06=12, C16=13
    C1d_indices = [12, 13]
    add_trivial_2x2_2x1(A1_indices, B1d_indices, C1d_indices)
    
    # Part 2: A2 * B2
    # A2 is 2x1 block from A (col 2)
    # Global A indices:
    # A02=2
    # A12=5
    
    # B2 is 1x7 block from B (row 2)
    # Global B indices:
    # B20=14, ..., B26=20
    
    for i in range(2):
        # A index for row i, col 2
        u_idx = i * 3 + 2
        for k in range(7):
            # B index for row 2, col k
            v_idx = 2 * 7 + k
            # C index for row i, col k
            w_idx = k * 2 + i
            
            u_col = [0.0] * 6
            v_col = [0.0] * 21
            w_col = [0.0] * 14
            
            u_col[u_idx] = 1.0
            v_col[v_idx] = 1.0
            w_col[w_idx] = 1.0
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
    # Convert to row-major format: U has shape (6, rank), so transpose.
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(6)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(21)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(14)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
