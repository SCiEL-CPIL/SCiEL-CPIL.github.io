import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    # Split A into two 2x2 blocks (A1, A2)
    # Split B into four 2x2x4 blocks? No, B is 4x8.
    # Split B into 4 blocks of 2x4:
    # B = [[B11, B12], [B21, B22]] where each is 2x4.
    # A = [A1, A2] where each is 2x2.
    # C = A1 @ B11 + A1 @ B12 + A2 @ B21 + A2 @ B22
    # Each term is 2x2 by 2x4. Rank 14.
    # Total rank 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen decomposition for 2x2 by 2x2
    # Inputs: u_offset, v_offset, w_offset, col_offset
    # u_offset: index in U for i*m+j
    # v_offset: index in V for j*p+k
    # w_offset: index in W for k*n+i
    def add_strassen_2x2(u_base, v_base, w_base, col_base):
        # Strassen 7 multiplications for 2x2 by 2x2
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # We need to map these to the tensor factors.
        # U indices for A: i*m+j
        # V indices for B: j*p+k
        # W indices for C: k*n+i
        
        # This is complex to map directly to Strassen's M's because Strassen combines A's and B's first.
        # The tensor decomposition requires U, V, W to be linear forms.
        # Strassen works by defining 7 linear combinations of A and B, multiplying them, and combining results.
        # This corresponds to a rank 7 decomposition.
        
        # For 2x2 by 2x4, we have two 2x2 multiplications.
        # Let's just use schoolbook for 2x2 by 2x4 to be safe? Rank 8.
        # 4 * 8 = 32.
        # 32 is better than 56? No, 32 is the lower bound.
        # If I use schoolbook for 2x2 by 2x4, rank is 8.
        # Total rank 4 * 8 = 32.
        
        # Let's verify if 2x2 by 2x4 can be done in rank 8.
        # A is 2x2. B is 2x4.
        # C1 = A11 B11 + A21 B21
        # C2 = A11 B12 + A21 B22
        # C3 = A12 B11 + A22 B21
        # C4 = A12 B12 + A22 B22
        # This is 4 dot products of length 2.
        # Each dot product is rank 2?
        # Sum of 4 rank 2 tensors = rank 8.
        # So yes, rank 8 is possible for 2x2 by 2x4.
        
        # So total rank 32 is possible!
        
        pass

    # Implement Rank 32 (Schoolbook for sub-blocks)
    # A = [A1, A2] (2x2 each)
    # B = [[B11, B12], [B21, B22]] (2x4 each)
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22
    
    # Term 1: A1 B11
    # A1 indices in A: (0,0), (0,1), (1,0), (1,1) -> U indices 0,1,4,5
    # B11 indices in B: (0,0..3), (1,0..3) -> V indices 0..3, 8..11
    # C indices: k=0..3, i=0,1 -> W indices 0,1, 2,3, 4,5, 6,7
    
    # We will iterate over the 4 sub-problems.
    # Sub-problem 1: A1 * B11
    # A1 is rows 0,1 of A. Cols 0,1.
    # B11 is rows 0,1 of B. Cols 0,3.
    
    col = 0
    for a_block in range(2): # 0 for A1, 1 for A2
        for b_block in range(2): # 0 for B11/B21, 1 for B12/B22
            # Determine offsets
            # A_block a_block: rows 0,1. cols 2*a_block, 2*a_block+1
            # B_block b_block: 
            # If b_block=0 (Left half of B): rows 0,1 (if a_block=0) or 2,3 (if a_block=1)?
            # No, B is split vertically into B_top and B_bot?
            # B = [[B11, B12], [B21, B22]]
            # B11 is rows 0,1 cols 0,1? No, B is 4x8.
            # B11 is 2x4. Rows 0,1. Cols 0,1,2,3.
            # B12 is 2x4. Rows 0,1. Cols 4,5,6,7.
            # B21 is 2x4. Rows 2,3. Cols 0,1,2,3.
            # B22 is 2x4. Rows 2,3. Cols 4,5,6,7.
            
            # A1 is rows 0,1. Cols 0,1.
            # A2 is rows 0,1. Cols 2,3.
            
            # Product A1 * B11:
            # A1 (2x2) * B11 (2x4) -> 2x4.
            # This contributes to C rows 0,1 (from A) and Cols 0,1,2,3 (from B11).
            
            # Product A1 * B12:
            # A1 (2x2) * B12 (2x4) -> 2x4.
            # Contributes to C rows 0,1 and Cols 4,5,6,7.
            
            # Product A2 * B21:
            # A2 (2x2) * B21 (2x4) -> 2x4.
            # Contributes to C rows 0,1 and Cols 0,1,2,3.
            
            # Product A2 * B22:
            # A2 (2x2) * B22 (2x4) -> 2x4.
            # Contributes to C rows 0,1 and Cols 4,5,6,7.
            
            # We need to sum A1*B11 and A2*B21 for the left half of C.
            # And A1*B12 and A2*B22 for the right half.
            
            # Let's just decompose each 2x2 by 2x4 multiplication in rank 8.
            # And add them up.
            
            # A_block index: 0 (cols 0,1), 1 (cols 2,3)
            # B_block index: 
            # 0: rows 0,1, cols 0..3 (B11)
            # 1: rows 0,1, cols 4..7 (B12)
            # 2: rows 2,3, cols 0..3 (B21)
            # 3: rows 2,3, cols 4..7 (B22)
            
            # Mapping:
            # If a_block=0 (A1), it multiplies B11 (b_block=0) and B12 (b_block=1).
            # If a_block=1 (A2), it multiplies B21 (b_block=2) and B22 (b_block=3).
            
            b_idx = a_block * 2 + b_block # 0,1 for A1; 2,3 for A2
            
            # A indices
            a_row_start = 0
            a_col_start = a_block * 2
            
            # B indices
            b_row_start = b_block * 2 # 0,0,2,2
            b_col_start = (b_block % 2) * 4 # 0,4,0,4
            
            # C indices
            c_row_start = 0 # Always 0,1
            c_col_start = (b_block % 2) * 4 # 0,4,0,4
            
            # Decompose 2x2 by 2x4 in rank 8
            # A (2x2) * B (2x4)
            # A indices: i in 0,1; j in 0,1
            # B indices: j in 0,1; k in 0,1,2,3
            # C indices: i in 0,1; k in 0,1,2,3
            
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        # U index: (a_row_start + i) * m + (a_col_start + j)
                        u_idx = (a_row_start + i) * m + (a_col_start + j)
                        # V index: (b_row_start + j) * p + (b_col_start + k)
                        v_idx = (b_row_start + j) * p + (b_col_start + k)
                        # W index: (c_col_start + k) * n + (c_row_start + i)
                        w_idx = (c_col_start + k) * n + (c_row_start + i)
                        
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
                        
    return {"U": U, "V": V, "W": W}
