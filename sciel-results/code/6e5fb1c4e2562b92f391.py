import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    # We use a rank-22 decomposition:
    # - Strassen (7 terms) for the 2x2 A-block (cols 0,1) with 2x2 B-blocks (rows 0,1) for cols 0,1 and cols 2,3
    # - Schoolbook (2*4 = 8 terms) for the 2x1 A-block (col 2) with 1x4 B-block (row 2)
    # - Schoolbook (1*4*2... wait, let me recount)
    # Actually: 
    #   A is 2x3, B is 3x4, C is 2x4
    #   Split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2)
    #   Split B into B_top (2x4, rows 0,1) and B_bottom (1x4, row 2)
    #   Split B_top into B_top_left (2x2, rows 0,1; cols 0,1) and B_top_right (2x2, rows 0,1; cols 2,3)
    #   
    #   C = A_left * B_top + A_right * B_bottom
    #   C_left (2x2, cols 0,1) = A_left * B_top_left
    #   C_right (2x2, cols 2,3) = A_left * B_top_right
    #   C += A_right * B_bottom (full 2x4)
    #
    #   Rank for A_left * B_top_left: Strassen = 7
    #   Rank for A_left * B_top_right: Strassen = 7
    #   Rank for A_right * B_bottom: schoolbook = 2*1*4 = 8
    #   Total rank = 7 + 7 + 8 = 22

    rank = 22
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*4 = 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 4*2 = 8 rows

    def add_term(u_entries, v_entries, w_entries, col):
        """Add a single tensor term to columns of U, V, W."""
        for (i, j), val in u_entries.items():
            U[i * m + j][col] += val
        for (j, k), val in v_entries.items():
            V[j * p + k][col] += val
        for (k, i), val in w_entries.items():
            W[k * n + i][col] += val

    col = 0

    # --- Strassen terms for A_left * B_top_left -> C_left (cols 0,1) ---
    # A_left: rows 0,1; cols 0,1. Indices: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # B_top_left: rows 0,1; cols 0,1. Indices: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # C_left: rows 0,1; cols 0,1. Indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3

    # M1 = (A00 + A11) * (B00 + B11)
    add_term({(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1},
             {(0,0): 1, (1,1): 1}, col)
    col += 1

    # M2 = (A10 + A11) * B00
    add_term({(1,0): 1, (1,1): 1},
             {(0,0): 1},
             {(0,0): -1, (0,1): 1, (1,0): 1, (1,1): 1}, col)
    col += 1

    # M3 = A00 * (B10 + B11)
    add_term({(0,0): 1},
             {(1,0): 1, (1,1): 1},
             {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, col)
    col += 1

    # M4 = A11 * (B01 + B10)
    add_term({(1,1): 1},
             {(0,1): 1, (1,0): 1},
             {(0,0): -1, (0,1): 1, (1,0): -1, (1,1): 1}, col)
    col += 1

    # M5 = (A00 + A01) * B11
    add_term({(0,0): 1, (0,1): 1},
             {(1,1): 1},
             {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1}, col)
    col += 1

    # M6 = (A10 - A00) * (B00 + B01)
    add_term({(1,0): 1, (0,0): -1},
             {(0,0): 1, (0,1): 1},
             {(0,1): 1, (1,1): 1}, col)
    col += 1

    # M7 = (A01 - A11) * (B10 + B11)
    add_term({(0,1): 1, (1,1): -1},
             {(1,0): 1, (1,1): 1},
             {(0,0): -1, (0,1): -1, (1,0): -1, (1,1): 1}, col)
    col += 1

    # --- Strassen terms for A_left * B_top_right -> C_right (cols 2,3) ---
    # B_top_right: rows 0,1; cols 2,3. Indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # C_right: rows 0,1; cols 2,3. Indices: (0,2)=4, (0,3)=5, (1,2)=6, (1,3)=7

    # M8 = (A00 + A11) * (B02 + B13)
    add_term({(0,0): 1, (1,1): 1},
             {(0,2): 1, (1,3): 1},
             {(2,0): 1, (3,1): 1}, col)
    col += 1

    # M9 = (A10 + A11) * B02
    add_term({(1,0): 1, (1,1): 1},
             {(0,2): 1},
             {(2,0): -1, (2,1): 1, (3,0): 1, (3,1): 1}, col)
    col += 1

    # M10 = A00 * (B12 + B13)
    add_term({(0,0): 1},
             {(1,2): 1, (1,3): 1},
             {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): 1}, col)
    col += 1

    # M11 = A11 * (B03 + B12)
    add_term({(1,1): 1},
             {(0,3): 1, (1,2): 1},
             {(2,0): -1, (2,1): 1, (3,0): -1, (3,1): 1}, col)
    col += 1

    # M12 = (A00 + A01) * B13
    add_term({(0,0): 1, (0,1): 1},
             {(1,3): 1},
             {(2,0): 1, (2,1): 1, (3,0): 1, (3,1): -1}, col)
    col += 1

    # M13 = (A10 - A00) * (B02 + B03)
    add_term({(1,0): 1, (0,0): -1},
             {(0,2): 1, (0,3): 1},
             {(2,1): 1, (3,1): 1}, col)
    col += 1

    # M14 = (A01 - A11) * (B12 + B13)
    add_term({(0,1): 1, (1,1): -1},
             {(1,2): 1, (1,3): 1},
             {(2,0): -1, (2,1): -1, (3,0): -1, (3,1): 1}, col)
    col += 1

    # --- Schoolbook terms for A_right * B_bottom -> C (full 2x4) ---
    # A_right: rows 0,1; col 2. Indices: (0,2)=2, (1,2)=5
    # B_bottom: row 2; cols 0,1,2,3. Indices: (2,0)=8, (2,1)=9, (2,2)=10, (2,3)=11
    # C: rows 0,1; cols 0,1,2,3. Indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (0,2)=4, (0,3)=5, (1,2)=6, (1,3)=7

    for j in range(2):  # A_right row index: 0,1
        for k in range(4):  # B_bottom col index: 0,1,2,3
            add_term({(j, 2): 1},
                     {(2, k): 1},
                     {(k, j): 1}, col)
            col += 1

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Reconstruct tensor: T[i,m+j,p+k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check a few random entries
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        # Expected: 1 if i,j,k form a valid product term (always 1 in standard matmul)
        # Actually, the matmul tensor T[i,j,k] = 1 for all i,j,k? No.
        # T[i,j,k] contributes to C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        expected = 1
        computed = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
        if abs(computed - expected) > 1e-10:
            # Fall back to schoolbook
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

    return {"U": U, "V": V, "W": W}
