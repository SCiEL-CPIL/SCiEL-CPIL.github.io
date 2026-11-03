import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen for 2x2 blocks and schoolbook for 2x1 block
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_row_offset, b_col_offset, c_col_offset):
        # a_row_offset: 0 or 2 (for j=0,1 or j=2,3)
        # b_col_offset: 0, 2, or 4 (for k=0,1 or 2,3 or 4,5)
        # c_col_offset: same as b_col_offset
        
        # Local indices:
        # A: i in {0,1}, j in {0,1} relative to a_row_offset
        # B: j in {0,1} relative to a_row_offset, k in {0,1} relative to b_col_offset
        # C: i in {0,1}, k in {0,1} relative to c_col_offset
        
        # We need to add 7 terms
        
        # M1
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11 (with signs)
        
        # We must add 7 separate rank-1 terms to the global matrices.
        
        # Term 1: M1
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        # But rank-1 tensors are scalar products.
        # We need to decompose the bilinear form into rank-1 tensors.
        # M1 = (A00+A11)(B00+B11)
        # This is not a single rank-1 tensor in the U, V, W space if W has multiple components.
        # Wait, the decomposition is $\sum u_t \otimes v_t \otimes w_t$.
        # For Strassen, we define 7 intermediate products $M_t$.
        # Each $M_t$ is a scalar.
        # Then $C$ is a linear combination of $M_t$.
        # So we add 7 columns to U, V, W.
        # For column t:
        # U is the linear form on A for $M_t$.
        # V is the linear form on B for $M_t$.
        # W is the vector in C space (coefficients for $C_{i,k}$).
        
        # So for each of the 7 M's, we add one column.
        
        # M1
        U[a_row_offset + 0][col] = 1 # A00
        U[a_row_offset + 1][col] = 1 # A11
        V[a_row_offset * p + b_col_offset + 0][col] = 1 # B00
        V[a_row_offset * p + b_col_offset + 1][col] = 1 # B11
        W[(c_col_offset + 0) * n + 0][col] = 1 # C00
        W[(c_col_offset + 1) * n + 1][col] = 1 # C11
        col += 1
        
        # M2
        U[a_row_offset + 1][col] = 1 # A10
        U[a_row_offset + 1][col] += 1 # A11 (so 2? No, (A10+A11))
        # Wait, U is a linear form.
        # M2 = (A10+A11) B00
        # U: A10 (1), A11 (1)
        # V: B00 (1)
        # W: C10 (1), C11 (-1)
        U[a_row_offset + 1][col] = 1 # A10
        U[a_row_offset + 1][col] = 1 # A11 (overwrite? No, distinct indices)
        # A indices: i=1, j=0 and i=1, j=1.
        # Flat index for A: i*m + j.
        # a_row_offset is j offset.
        # A10: i=1, j=a_row_offset.
        # A11: i=1, j=a_row_offset+1.
        
        # Let's rewrite carefully.
        
        # Clear previous partial writes for this column?
        # Better to write a helper that takes lists of (idx, val)
        
        pass

    # Redo with a cleaner helper
    
    def add_term_helper(u_pairs, v_pairs, w_pairs):
        nonlocal col
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1

    # Block 1: j=0,1
    # B cols 0,1
    # C cols 0,1
    
    # M1
    add_term_helper(
        [(0*5+0, 1), (1*5+0, 1), (0*5+1, 1), (1*5+1, 1)], # A00, A10? No.
        # A indices: i*m+j
        # A00: 0*5+0=0
        # A11: 1*5+1=6
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        # C indices: k*n+i
        # C00: 0*2+0=0
        # C11: 1*2+1=3
        [(0, 1), (6, 1)], # U: A00, A11
        [(0*6+0, 1), (1*6+1, 1)], # V: B00, B11
        [(0*2+0, 1), (1*2+1, 1)] # W: C00, C11
    )
    
    # M2
    # M2 = (A10+A11)B00
    # U: A10 (1), A11 (1)
    # V: B00 (1)
    # W: C10 (1), C11 (-1)
    add_term_helper(
        [(1*5+0, 1), (1*5+1, 1)], # A10, A11
        [(0*6+0, 1)], # B00
        [(0*2+1, 1), (1*2+1, -1)] # C10, C11
    )
    
    # M3
    # M3 = A00(B10-B11)
    # U: A00 (1)
    # V: B10 (1), B11 (-1)
    # W: C01 (1), C11 (1)
    add_term_helper(
        [(0*5+0, 1)], # A00
        [(1*6+0, 1), (1*6+1, -1)], # B10, B11
        [(0*2+1, 1), (1*2+1, 1)] # C01, C11
    )
    
    # M4
    # M4 = A11(B01-B00)
    # U: A11 (1)
    # V: B01 (1), B00 (-1)
    # W: C00 (1), C10 (1)
    add_term_helper(
        [(1*5+1, 1)], # A11
        [(0*6+1, 1), (0*6+0, -1)], # B01, B00
        [(0*2+0, 1), (0*2+1, 1)] # C00, C10
    )
    
    # M5
    # M5 = (A00+A10)B11
    # U: A00 (1), A10 (1)
    # V: B11 (1)
    # W: C00 (-1), C01 (1)
    add_term_helper(
        [(0*5+0, 1), (1*5+0, 1)], # A00, A10
        [(1*6+1, 1)], # B11
        [(0*2+0, -1), (0*2+1, 1)] # C00, C01
    )
    
    # M6
    # M6 = (A10-A00)(B00+B10)
    # U: A10 (1), A00 (-1)
    # V: B00 (1), B10 (1)
    # W: C11 (1)
    add_term_helper(
        [(1*5+0, 1), (0*5+0, -1)], # A10, A00
        [(0*6+0, 1), (1*6+0, 1)], # B00, B10
        [(1*2+1, 1)] # C11
    )
    
    # M7
    # M7 = (A01-A11)(B11+B01)
    # U: A01 (1), A11 (-1)
    # V: B11 (1), B01 (1)
    # W: C00 (1)
    add_term_helper(
        [(0*5+1, 1), (1*5+1, -1)], # A01, A11
        [(1*6+1, 1), (0*6+1, 1)], # B11, B01
        [(0*2+0, 1)] # C00
    )
    
    # Block 2: j=2,3
    # B cols 2,3
    # C cols 2,3
    
    # Shift indices
    j_off = 2
    k_off = 2
    
    # M1
    add_term_helper(
        [(0*5+j_off, 1), (1*5+j_off+1, 1)], # A02, A13
        [(j_off*6+k_off, 1), ((j_off+1)*6+k_off+1, 1)], # B22, B33
        [(k_off*2+0, 1), ((k_off+1)*2+1, 1)] # C02, C13
    )
    
    # M2
    add_term_helper(
        [(1*5+j_off, 1), (1*5+j_off+1, 1)], # A12, A13
        [(j_off*6+k_off, 1)], # B22
        [(k_off*2+1, 1), ((k_off+1)*2+1, -1)] # C12, C13
    )
    
    # M3
    add_term_helper(
        [(0*5+j_off, 1)], # A02
        [((j_off+1)*6+k_off, 1), ((j_off+1)*6+k_off+1, -1)], # B32, B33
        [(k_off*2+1, 1), ((k_off+1)*2+1, 1)] # C03, C13
    )
    
    # M4
    add_term_helper(
        [(1*5+j_off+1, 1)], # A13
        [(j_off*6+k_off+1, 1), (j_off*6+k_off, -1)], # B23, B22
        [(k_off*2+0, 1), (k_off*2+1, 1)] # C02, C12
    )
    
    # M5
    add_term_helper(
        [(0*5+j_off, 1), (1*5+j_off, 1)], # A02, A12
        [((j_off+1)*6+k_off+1, 1)], # B33
        [(k_off*2+0, -1), (k_off*2+1, 1)] # C02, C03
    )
    
    # M6
    add_term_helper(
        [(1*5+j_off, 1), (0*5+j_off, -1)], # A12, A02
        [(j_off*6+k_off, 1), ((j_off+1)*6+k_off, 1)], # B22, B32
        [((k_off+1)*2+1, 1)] # C13
    )
    
    # M7
    add_term_helper(
        [(0*5+j_off+1, 1), (1*5+j_off+1, -1)], # A03, A13
        [((j_off+1)*6+k_off+1, 1), (j_off*6+k_off+1, 1)], # B33, B23
        [(k_off*2+0, 1)] # C02
    )
    
    # Block 3: j=4
    # B cols 4,5
    # C cols 4,5
    # Schoolbook: 2x1x6 -> 12 terms
    
    j_idx = 4
    
    for k in range(6):
        # C0k = A04 B4k
        add_term_helper(
            [(0*5+j_idx, 1)], # A04
            [(j_idx*6+k, 1)], # B4k
            [(k*2+0, 1)] # C0k
        )
        
        # C1k = A14 B4k
        add_term_helper(
            [(1*5+j_idx, 1)], # A14
            [(j_idx*6+k, 1)], # B4k
            [(k*2+1, 1)] # C1k
        )
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
