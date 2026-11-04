import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 65
    V = [[0] * rank for _ in range(m * p)] # 20 x 65
    W = [[0] * rank for _ in range(p * n)] # 20 x 65
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    
    # Helper to get global indices for a 2x2 block in a matrix
    # Matrix A is 4x4, flattened row-major.
    # Block (br, bc) of size 2x2.
    def get_A_indices(br, bc):
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [r*4 + c for r in rows for c in cols]

    # Matrix B is 4x5. We use cols 0-3 for the 4x4 part.
    # Block (br, bc) of size 2x2.
    def get_B_indices(br, bc):
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [r*5 + c for r in rows for c in cols]

    # Matrix C is 4x5. We use cols 0-3 for the 4x4 part.
    # W index is k*n + i. k is col, i is row.
    # Block (br, bc) of C (4x4 part).
    # br corresponds to row block of C?
    # C = A B.
    # C block (br, bc) is rows br*2..br*2+1, cols bc*2..bc*2+1.
    def get_C_indices(br, bc):
        rows = [br*2, br*2+1]
        cols = [bc*2, bc*2+1]
        return [k*4 + i for k in cols for i in rows]

    # Define the 7 Strassen terms for 2x2
    # Each term is (u_vec, v_vec, w_vec) for 2x2 indices
    # u_vec corresponds to A 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    # v_vec corresponds to B 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    # w_vec corresponds to C 2x2 indices [0,1,2,3] -> [00, 01, 10, 11]
    
    # Standard Strassen 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
    # where coeffs are for the 4 elements of the 2x2 matrix.
    # Order: 00, 01, 10, 11
    
    strassen_2x2 = [
        # (U, V, W)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Define the 7 block products for 4x4
    # M1 = (A11 + A22)(B11 + B22) -> (A00+A11)(B00+B11)
    # M2 = (A21 + A22)B11 -> (A10+A11)B00
    # M3 = A11(B12 - B22) -> A00(B01-B11)
    # M4 = A22(B21 - B11) -> A11(B10-B00)
    # M5 = (A11 + A12)B22 -> (A00+A01)B11
    # M6 = (A21 - A11)(B11 + B12) -> (A10-A00)(B00+B01)
    # M7 = (A12 - A22)(B21 + B22) -> (A01-A11)(B10+B11)
    
    # For each M_k, we specify:
    # A_blocks: list of (block, coeff)
    # B_blocks: list of (block, coeff)
    # C_blocks: list of (block, coeff)
    # Note: C is the sum of these terms.
    
    block_defs = [
        # M1
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        # M2
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        # M3
        {
            'A': [((0,0), 1)],
            'B': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        # M4
        {
            'A': [((1,1), 1)],
            'B': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        # M5
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        # M6
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        # M7
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    col = 0
    for b_def in block_defs:
        # For each of the 7 Strassen terms in the 2x2 sub-problem
        for u_2x2, v_2x2, w_2x2 in strassen_2x2:
            # Construct U vector (length 16)
            u_vec = [0] * 16
            for (br, bc), coeff in b_def['A']:
                a_indices = get_A_indices(br, bc)
                for idx, a_idx in enumerate(a_indices):
                    u_vec[a_idx] += coeff * u_2x2[idx]
            
            # Construct V vector (length 20)
            v_vec = [0] * 20
            for (br, bc), coeff in b_def['B']:
                b_indices = get_B_indices(br, bc)
                for idx, b_idx in enumerate(b_indices):
                    v_vec[b_idx] += coeff * v_2x2[idx]
            
            # Construct W vector (length 20)
            w_vec = [0] * 20
            for (br, bc), coeff in b_def['C']:
                c_indices = get_C_indices(br, bc)
                for idx, c_idx in enumerate(c_indices):
                    w_vec[c_idx] += coeff * w_2x2[idx]
            
            # Fill matrices
            for i in range(16):
                U[i][col] = u_vec[i]
            for j in range(20):
                V[j][col] = v_vec[j]
            for k in range(20):
                W[k][col] = w_vec[k]
            
            col += 1
            
    # --- Part 2: 4x4x1 Schoolbook (Rank 16) ---
    # A is 4x4, B'' is 4x1 (col 4 of B)
    # C'' is 4x1 (col 4 of C)
    # C[i, 4] = sum_j A[i, j] B''[j, 4]
    
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1 # B'' index is j*5 + 4
            W[4*n + i][col] = 1 # C'' index is 4*4 + i
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
