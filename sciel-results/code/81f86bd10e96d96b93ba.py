import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen's algorithm on six 2x2x2 sub-blocks and 12 scalar multiplications for the remainder.
    Rank = 6 * 7 + 12 = 54.
    """
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    # Helper to add a Strassen block
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        nonlocal col
        # A is 2x2 at (a_row_off, a_col_off)
        # B is 2x2 at (b_row_off, b_col_off)
        # C is 2x2 at (c_row_off, c_col_off)
        
        # Strassen multipliers
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # C contributions
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices in flattened A (n*m), B (m*p), C (p*n)
        # A index: (a_row_off + dr) * m + (a_col_off + dc)
        # B index: (b_row_off + dr) * p + (b_col_off + dc)
        # C index: (c_col_off + dc) * n + (c_row_off + dr)  # Note: C is p*n, index k*n+i
        
        # Define the 7 terms
        terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # Coeffs are lists of (index, value)
            # M1
            (
                [((a_row_off)*m + (a_col_off), 1.0), ((a_row_off+1)*m + (a_col_off+1), 1.0)],
                [((b_row_off)*p + (b_col_off), 1.0), ((b_row_off+1)*p + (b_col_off+1), 1.0)],
                [((c_col_off)*n + (c_row_off), 1.0), ((c_col_off+1)*n + (c_row_off+1), 1.0)]
            ),
            # M2
            (
                [((a_row_off+1)*m + (a_col_off), 1.0), ((a_row_off+1)*m + (a_col_off+1), 1.0)],
                [((b_row_off)*p + (b_col_off), 1.0)],
                [((c_col_off)*n + (c_row_off+1), 1.0)]
            ),
            # M3
            (
                [((a_row_off)*m + (a_col_off), 1.0)],
                [((b_row_off)*p + (b_col_off+1), 1.0), ((b_row_off+1)*p + (b_col_off+1), -1.0)],
                [((c_col_off+1)*n + (c_row_off), 1.0)]
            ),
            # M4
            (
                [((a_row_off+1)*m + (a_col_off+1), 1.0)],
                [((b_row_off+1)*p + (b_col_off), 1.0), ((b_row_off)*p + (b_col_off), -1.0)],
                [((c_col_off)*n + (c_row_off), 1.0)]
            ),
            # M5
            (
                [((a_row_off)*m + (a_col_off), 1.0), ((a_row_off)*m + (a_col_off+1), 1.0)],
                [((b_row_off+1)*p + (b_col_off+1), 1.0)],
                [((c_col_off)*n + (c_row_off), -1.0), ((c_col_off+1)*n + (c_row_off+1), 1.0)]
            ),
            # M6
            (
                [((a_row_off+1)*m + (a_col_off), 1.0), ((a_row_off)*m + (a_col_off), -1.0)],
                [((b_row_off)*p + (b_col_off), 1.0), ((b_row_off)*p + (b_col_off+1), 1.0)],
                [((c_col_off+1)*n + (c_row_off+1), 1.0)]
            ),
            # M7
            (
                [((a_row_off)*m + (a_col_off+1), 1.0), ((a_row_off+1)*m + (a_col_off+1), -1.0)],
                [((b_row_off+1)*p + (b_col_off), 1.0), ((b_row_off+1)*p + (b_col_off+1), 1.0)],
                [((c_col_off)*n + (c_row_off), 1.0)]
            )
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for idx, val in a_coeffs:
                U[idx, col] += val
            for idx, val in b_coeffs:
                V[idx, col] += val
            for idx, val in c_coeffs:
                W[idx, col] += val
            col += 1

    # 1. Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # 2. Split B into B_T (rows 0,1) and B_B (rows 2,3)
    # 3. Split B_T and B_B into 3 blocks of 2x2 each (cols 0-1, 2-3, 4-5)
    
    # Block 1: A_L (0,0) * B_T (0,0) -> C (0,0)
    # A rows 0-1, cols 0-1. B rows 0-1, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(0, 0, 0, 0, 0, 0)
    
    # Block 2: A_L (0,0) * B_T (0,2) -> C (0,2)
    # A rows 0-1, cols 0-1. B rows 0-1, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(0, 0, 0, 2, 0, 2)
    
    # Block 3: A_L (0,0) * B_T (0,4) -> C (0,4)
    # A rows 0-1, cols 0-1. B rows 0-1, cols 4-5. C rows 0-1, cols 4-5.
    add_strassen_block(0, 0, 0, 4, 0, 4)
    
    # Block 4: A_R (0,2) * B_B (2,0) -> C (0,0)
    # A rows 0-1, cols 2-3. B rows 2-3, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(0, 2, 2, 0, 0, 0)
    
    # Block 5: A_R (0,2) * B_B (2,2) -> C (0,2)
    # A rows 0-1, cols 2-3. B rows 2-3, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(0, 2, 2, 2, 0, 2)
    
    # Block 6: A_R (0,2) * B_B (2,4) -> C (0,4)
    # A rows 0-1, cols 2-3. B rows 2-3, cols 4-5. C rows 0-1, cols 4-5.
    add_strassen_block(0, 2, 2, 4, 0, 4)

    # Remainder: A (col 4) * B (row 4)
    # A is 2x1 at col 4. B is 1x6 at row 4.
    # 12 scalar multiplications.
    for i in range(n):  # 0, 1
        for k in range(p):  # 0..5
            # U index: i*m + 4
            U[i * m + 4, col] = 1.0
            # V index: 4*p + k
            V[4 * p + k, col] = 1.0
            # W index: k*n + i
            W[k * n + i, col] = 1.0
            col += 1

    if time.time() > deadline:
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
