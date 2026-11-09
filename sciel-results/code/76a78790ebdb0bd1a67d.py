import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # U shape (8, R), V shape (24, R), W shape (12, R)
    
    # We use rank 42 decomposition
    R = 42
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add a Strassen term for a 2x2 by 2x2 block
    # A is 2x2, B is 2x2
    # A indices in U: base_i, base_j
    # B indices in V: base_j, base_k
    # C indices in W: base_k, base_i (note W is k*n+i)
    
    def add_strassen(U, V, W, col_offset, a_base_i, a_base_j, b_base_j, b_base_k):
        # a_base_i: row offset in U for i (0 or 2)
        # a_base_j: row offset in U for j (0 or 2)
        # b_base_j: row offset in V for j (0 or 4)
        # b_base_k: row offset in V for k (0 or 2)
        
        # U rows for A:
        # A00: U[a_base_i + 0 * 2 + 0] -> U[a_base_i]
        # A01: U[a_base_i + 0 * 2 + 1] -> U[a_base_i + 1]
        # A10: U[a_base_i + 1 * 2 + 0] -> U[a_base_i + 2]
        # A11: U[a_base_i + 1 * 2 + 1] -> U[a_base_i + 3]
        
        # V rows for B:
        # B00: V[b_base_j + 0 * 6 + 0] -> V[b_base_j]
        # B01: V[b_base_j + 0 * 6 + 1] -> V[b_base_j + 1]
        # B10: V[b_base_j + 1 * 6 + 0] -> V[b_base_j + 6]
        # B11: V[b_base_j + 1 * 6 + 1] -> V[b_base_j + 7]
        
        # W rows for C:
        # C00: W[0 * 2 + 0] -> W[b_base_k + 0 * 2 + 0]
        # C01: W[1 * 2 + 0] -> W[b_base_k + 1 * 2 + 0]
        # C10: W[0 * 2 + 1] -> W[b_base_k + 0 * 2 + 1]
        # C11: W[1 * 2 + 1] -> W[b_base_k + 1 * 2 + 1]
        
        # Strassen coefficients
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
        
        terms = [
            # (M_idx, U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: list of (row, coeff)
            # V_coeffs: list of (row, coeff)
            # W_coeffs: list of (row, coeff)
            
            # M1
            (0, [(a_base_i, 1), (a_base_i+3, 1)], [(b_base_j, 1), (b_base_j+7, 1)], [(b_base_k, 1), (b_base_k+2, 1)]),
            # M2
            (1, [(a_base_i+2, 1), (a_base_i+3, 1)], [(b_base_j, 1)], [(b_base_k+1, 1)]),
            # M3
            (2, [(a_base_i, 1)], [(b_base_j+1, 1), (b_base_j+7, -1)], [(b_base_k+1, 1)]),
            # M4
            (3, [(a_base_i+3, 1)], [(b_base_j+6, 1), (b_base_j, -1)], [(b_base_k, 1), (b_base_k+1, 1)]),
            # M5
            (4, [(a_base_i, 1), (a_base_i+1, 1)], [(b_base_j+7, 1)], [(b_base_k, -1)]),
            # M6
            (5, [(a_base_i+2, 1), (a_base_i, -1)], [(b_base_j, 1), (b_base_j+1, 1)], [(b_base_k+2, 1)]),
            # M7
            (6, [(a_base_i+1, 1), (a_base_i+3, -1)], [(b_base_j+6, 1), (b_base_j+7, 1)], [(b_base_k, 1)]),
        ]
        
        for t in terms:
            idx = col_offset + t[0]
            for row, coeff in t[1]:
                U[row][idx] += coeff
            for row, coeff in t[2]:
                V[row][idx] += coeff
            for row, coeff in t[3]:
                W[row][idx] += coeff

    # Block 1: A1 (rows 0-3 of U), B1 (rows 0-11 of V)
    # B1 is 2x6. Split into 3 2x2 blocks.
    # B11: j=0,1, k=0,1. V rows: 0,1,6,7.
    # B12: j=0,1, k=2,3. V rows: 2,3,8,9.
    # B13: j=0,1, k=4,5. V rows: 4,5,10,11.
    
    # A1: i=0,1, j=0,1. U rows: 0,1,2,3.
    # a_base_i = 0, a_base_j = 0.
    
    # B11: b_base_j = 0, b_base_k = 0.
    add_strassen(U, V, W, 0, 0, 0, 0, 0)
    
    # B12: b_base_j = 0, b_base_k = 2.
    # V rows for k=2,3 are 2,3 and 8,9.
    # b_base_k is offset in k.
    # V index = j*6 + k.
    # For B12, j=0,1. k=2,3.
    # V rows: 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9.
    # In add_strassen, b_base_k is the base k offset.
    # V row for B00 is b_base_j + 0*6 + b_base_k.
    # So for B12, b_base_j=0, b_base_k=2.
    add_strassen(U, V, W, 7, 0, 0, 0, 2)
    
    # B13: b_base_j = 0, b_base_k = 4.
    add_strassen(U, V, W, 14, 0, 0, 0, 4)
    
    # Block 2: A2 (rows 4-7 of U), B2 (rows 12-23 of V)
    # A2: i=0,1, j=2,3.
    # U index = i*4 + j.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    # So U rows: 2,3,6,7.
    # a_base_i = 2? No, the mapping in add_strassen assumes contiguous j.
    # A20 (i=0,j=2) is U[2].
    # A21 (i=0,j=3) is U[3].
    # A22 (i=1,j=2) is U[6].
    # A23 (i=1,j=3) is U[7].
    # The helper assumes A00 is U[a_base_i], A01 is U[a_base_i+1], A10 is U[a_base_i+2], A11 is U[a_base_i+3].
    # Here A00 is U[2], A01 is U[3]. OK.
    # A10 is U[6], A11 is U[7]. NOT OK. U[2+2]=U[4] is A02 (i=0,j=2? No i=1,j=0).
    # The U indexing is i*m + j.
    # For A2 (j=2,3), the rows are not contiguous in the same way if we just shift base.
    # We need a more flexible helper or manual mapping.
    
    # Let's modify add_strassen to take explicit row indices.
    
    # Re-implementing logic to be safe.
    
    # Clear U, V, W and restart with explicit indices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def add_strassen_explicit(U, V, W, col_offset, u_rows, v_rows, w_rows):
        # u_rows: [A00, A01, A10, A11]
        # v_rows: [B00, B01, B10, B11]
        # w_rows: [C00, C01, C10, C11]
        
        terms = [
            # M1
            (0, [(u_rows[0], 1), (u_rows[3], 1)], [(v_rows[0], 1), (v_rows[3], 1)], [(w_rows[0], 1), (w_rows[2], 1)]),
            # M2
            (1, [(u_rows[2], 1), (u_rows[3], 1)], [(v_rows[0], 1)], [(w_rows[1], 1)]),
            # M3
            (2, [(u_rows[0], 1)], [(v_rows[1], 1), (v_rows[3], -1)], [(w_rows[1], 1)]),
            # M4
            (3, [(u_rows[3], 1)], [(v_rows[2], 1), (v_rows[0], -1)], [(w_rows[0], 1), (w_rows[1], 1)]),
            # M5
            (4, [(u_rows[0], 1), (u_rows[1], 1)], [(v_rows[3], 1)], [(w_rows[0], -1)]),
            # M6
            (5, [(u_rows[2], 1), (u_rows[0], -1)], [(v_rows[0], 1), (v_rows[1], 1)], [(w_rows[3], 1)]),
            # M7
            (6, [(u_rows[1], 1), (u_rows[3], -1)], [(v_rows[2], 1), (v_rows[3], 1)], [(w_rows[0], 1)]),
        ]
        
        for t in terms:
            idx = col_offset + t[0]
            for row, coeff in t[1]:
                U[row][idx] += coeff
            for row, coeff in t[2]:
                V[row][idx] += coeff
            for row, coeff in t[3]:
                W[row][idx] += coeff

    # Block 1
    # A1: i=0,1; j=0,1.
    # U rows:
    # A00 (i=0,j=0): 0*4+0 = 0
    # A01 (i=0,j=1): 0*4+1 = 1
    # A10 (i=1,j=0): 1*4+0 = 4
    # A11 (i=1,j=1): 1*4+1 = 5
    u_rows_1 = [0, 1, 4, 5]
    
    # B1: j=0,1; k=0..5.
    # B11 (k=0,1):
    # B00 (j=0,k=0): 0*6+0 = 0
    # B01 (j=0,k=1): 0*6+1 = 1
    # B10 (j=1,k=0): 1*6+0 = 6
    # B11 (j=1,k=1): 1*6+1 = 7
    v_rows_11 = [0, 1, 6, 7]
    
    # C11 (k=0,1):
    # C00 (i=0,k=0): 0*2+0 = 0
    # C01 (i=0,k=1): 1*2+0 = 2
    # C10 (i=1,k=0): 0*2+1 = 1
    # C11 (i=1,k=1): 1*2+1 = 3
    w_rows_11 = [0, 2, 1, 3]
    
    add_strassen_explicit(U, V, W, 0, u_rows_1, v_rows_11, w_rows_11)
    
    # B12 (k=2,3):
    # B00 (j=0,k=2): 0*6+2 = 2
    # B01 (j=0,k=3): 0*6+3 = 3
    # B10 (j=1,k=2): 1*6+2 = 8
    # B11 (j=1,k=3): 1*6+3 = 9
    v_rows_12 = [2, 3, 8, 9]
    
    # C12 (k=2,3):
    # C00 (i=0,k=2): 2*2+0 = 4
    # C01 (i=0,k=3): 3*2+0 = 6
    # C10 (i=1,k=2): 2*2+1 = 5
    # C11 (i=1,k=3): 3*2+1 = 7
    w_rows_12 = [4, 6, 5, 7]
    
    add_strassen_explicit(U, V, W, 7, u_rows_1, v_rows_12, w_rows_12)
    
    # B13 (k=4,5):
    # B00 (j=0,k=4): 0*6+4 = 4
    # B01 (j=0,k=5): 0*6+5 = 5
    # B10 (j=1,k=4): 1*6+4 = 10
    # B11 (j=1,k=5): 1*6+5 = 11
    v_rows_13 = [4, 5, 10, 11]
    
    # C13 (k=4,5):
    # C00 (i=0,k=4): 4*2+0 = 8
    # C01 (i=0,k=5): 5*2+0 = 10
    # C10 (i=1,k=4): 4*2+1 = 9
    # C11 (i=1,k=5): 5*2+1 = 11
    w_rows_13 = [8, 10, 9, 11]
    
    add_strassen_explicit(U, V, W, 14, u_rows_1, v_rows_13, w_rows_13)
    
    # Block 2
    # A2: i=0,1; j=2,3.
    # U rows:
    # A00 (i=0,j=2): 0*4+2 = 2
    # A01 (i=0,j=3): 0*4+3 = 3
    # A10 (i=1,j=2): 1*4+2 = 6
    # A11 (i=1,j=3): 1*4+3 = 7
    u_rows_2 = [2, 3, 6, 7]
    
    # B2: j=2,3; k=0..5.
    # B21 (k=0,1):
    # B00 (j=2,k=0): 2*6+0 = 12
    # B01 (j=2,k=1): 2*6+1 = 13
    # B10 (j=3,k=0): 3*6+0 = 18
    # B11 (j=3,k=1): 3*6+1 = 19
    v_rows_21 = [12, 13, 18, 19]
    
    # C21 (k=0,1):
    # Same W rows as C11 because C is shared?
    # No, C = A1B1 + A2B2.
    # The W matrix accumulates contributions.
    # So we use the same w_rows for C11.
    w_rows_21 = [0, 2, 1, 3]
    
    add_strassen_explicit(U, V, W, 21, u_rows_2, v_rows_21, w_rows_21)
    
    # B22 (k=2,3):
    # B00 (j=2,k=2): 2*6+2 = 14
    # B01 (j=2,k=3): 2*6+3 = 15
    # B10 (j=3,k=2): 3*6+2 = 20
    # B11 (j=3,k=3): 3*6+3 = 21
    v_rows_22 = [14, 15, 20, 21]
    
    # C22 (k=2,3):
    w_rows_22 = [4, 6, 5, 7]
    
    add_strassen_explicit(U, V, W, 28, u_rows_2, v_rows_22, w_rows_22)
    
    # B23 (k=4,5):
    # B00 (j=2,k=4): 2*6+4 = 16
    # B01 (j=2,k=5): 2*6+5 = 17
    # B10 (j=3,k=4): 3*6+4 = 22
    # B11 (j=3,k=5): 3*6+5 = 23
    v_rows_23 = [16, 17, 22, 23]
    
    # C23 (k=4,5):
    w_rows_23 = [8, 10, 9, 11]
    
    add_strassen_explicit(U, V, W, 35, u_rows_2, v_rows_23, w_rows_23)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
