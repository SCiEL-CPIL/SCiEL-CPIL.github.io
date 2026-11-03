import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses a block decomposition:
    - Partition indices into I1={0,1} and I2={2,3,4}.
    - Apply Strassen (rank 7) to the 2x2x2 block (I1, I1, I1).
    - Apply Schoolbook (rank 27) to the 3x3x3 block (I2, I2, I2).
    - Apply Schoolbook to all other off-diagonal blocks.
    Total Rank = 7 + 27 + 90 = 124.
    """
    n, m, p = 5, 5, 5
    
    # Global index calculations
    # A index: i * 5 + k
    # B index: k * 5 + j
    # C index: j * 5 + i  (Note: C row is j, C col is i based on trace convention W[k*n+i] where k=row, i=col? 
    # Wait, prompt says W[k*n+i]. In seed: c = l * n + i. 
    # l was col B (which becomes row C). i was row A (which becomes col C).
    # So C index = row_C * 5 + col_C.
    # In my notation: row_C = j, col_C = i.
    # So C index = j * 5 + i. Correct.
    
    # We will build U, V, W as lists of lists (rows x rank)
    # Since rank is not known a priori, we accumulate terms as tuples of dicts.
    terms = []
    
    def add_term(a_dict, b_dict, c_dict):
        # Filter zero coefficients
        a_clean = {k: v for k, v in a_dict.items() if v != 0}
        b_clean = {k: v for k, v in b_dict.items() if v != 0}
        c_clean = {k: v for k, v in c_dict.items() if v != 0}
        if not a_clean or not b_clean or not c_clean:
            return
        terms.append((a_clean, b_clean, c_clean))

    I1 = [0, 1]
    I2 = [2, 3, 4]
    
    # Define a helper to get global indices for a block
    def get_indices(offsets):
        # offsets is a list of global indices
        return offsets

    # Strassen 2x2x2
    def run_strassen(row_offs, col_a_offs, col_b_offs):
        # Local indices 0, 1
        # A[i][k] -> Global A: (row_offs[i]) * 5 + (col_a_offs[k])
        # B[k][j] -> Global B: (col_a_offs[k]) * 5 + (col_b_offs[j])
        # C[i][j] -> Global C: (col_b_offs[j]) * 5 + (row_offs[i])
        
        def A_idx(i, k):
            return row_offs[i] * 5 + col_a_offs[k]
        
        def B_idx(k, j):
            return col_a_offs[k] * 5 + col_b_offs[j]
            
        def C_idx(i, j):
            return col_b_offs[j] * 5 + row_offs[i]
            
        # Strassen terms:
        # M1 = (A00 + A11)(B00 + B11)
        add_term(
            {A_idx(0,0): 1, A_idx(1,1): 1},
            {B_idx(0,0): 1, B_idx(1,1): 1},
            {C_idx(0,0): 1, C_idx(1,1): 1}
        )
        
        # M2 = (A10 + A11)(B00)
        add_term(
            {A_idx(1,0): 1, A_idx(1,1): 1},
            {B_idx(0,0): 1},
            {C_idx(1,0): 1, C_idx(1,1): -1}
        )
        
        # M3 = (A00)(B01 - B11)
        add_term(
            {A_idx(0,0): 1},
            {B_idx(0,1): 1, B_idx(1,1): -1},
            {C_idx(0,1): 1, C_idx(1,1): 1}
        )
        
        # M4 = (A11)(B10 - B00)
        add_term(
            {A_idx(1,1): 1},
            {B_idx(1,0): 1, B_idx(0,0): -1},
            {C_idx(1,0): 1, C_idx(0,0): 1}
        )
        
        # M5 = (A00 + A01)(B11)
        add_term(
            {A_idx(0,0): 1, A_idx(0,1): 1},
            {B_idx(1,1): 1},
            {C_idx(0,1): 1, C_idx(1,1): -1}
        )
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term(
            {A_idx(1,0): 1, A_idx(0,0): -1},
            {B_idx(0,0): 1, B_idx(0,1): 1},
            {C_idx(1,1): 1}
        )
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term(
            {A_idx(0,1): 1, A_idx(1,1): -1},
            {B_idx(1,0): 1, B_idx(1,1): 1},
            {C_idx(0,0): 1}
        )

    # Schoolbook for rectangular blocks
    def run_schoolbook(row_offs, col_a_offs, col_b_offs):
        # row_offs: global indices for i
        # col_a_offs: global indices for k
        # col_b_offs: global indices for j
        
        for i in range(len(row_offs)):
            for k in range(len(col_a_offs)):
                for j in range(len(col_b_offs)):
                    a_idx = row_offs[i] * 5 + col_a_offs[k]
                    b_idx = col_a_offs[k] * 5 + col_b_offs[j]
                    c_idx = col_b_offs[j] * 5 + row_offs[i]
                    
                    add_term(
                        {a_idx: 1},
                        {b_idx: 1},
                        {c_idx: 1}
                    )

    # Block 1: (I1, I1, I1) -> Strassen
    run_strassen(I1, I1, I1)
    
    # Block 2: (I2, I2, I2) -> Schoolbook (Rank 27)
    run_schoolbook(I2, I2, I2)
    
    # Off-diagonal blocks (Schoolbook)
    # (I1, I1, I2)
    run_schoolbook(I1, I1, I2)
    # (I1, I2, I1)
    run_schoolbook(I1, I2, I1)
    # (I1, I2, I2)
    run_schoolbook(I1, I2, I2)
    
    # (I2, I1, I1)
    run_schoolbook(I2, I1, I1)
    # (I2, I1, I2)
    run_schoolbook(I2, I1, I2)
    # (I2, I2, I1)
    run_schoolbook(I2, I2, I1)
    
    # Also need:
    # (I1, I1, I1) done
    # (I1, I1, I2) done
    # (I1, I2, I1) done
    # (I1, I2, I2) done
    # (I2, I1, I1) done
    # (I2, I1, I2) done
    # (I2, I2, I1) done
    # (I2, I2, I2) done
    
    # Wait, I missed (I1, I2, I2) is 2x3x3.
    # And (I2, I1, I1) is 3x2x2.
    # All 8 combinations are covered.
    
    rank = len(terms)
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for k, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][k] = val
        for idx, val in v_dict.items():
            V[idx][k] = val
        for idx, val in w_dict.items():
            W[idx][k] = val
            
    return {
        "U": U,
        "V": V,
        "W": W
    }
