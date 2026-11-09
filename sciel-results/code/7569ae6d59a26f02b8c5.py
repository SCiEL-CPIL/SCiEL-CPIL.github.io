"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x4 * 4x5.
    Strategy:
    1. Use Strassen's rank-7 algorithm for the four 2x2x2 sub-blocks (multiplying 2x2 by 2x2).
       This handles columns 0-3 of B.
    2. Use schoolbook (rank 8) for the remaining 2x4x1 slice (column 4 of B).
       Total rank = 4 * 7 + 2 * 4 * 1 = 28 + 8 = 36.
    
    This is better than the schoolbook rank of 40.
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # We partition the middle dimension m=4 into two blocks of size 2:
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    # And the output dimension p=5 into:
    # Part 1: k in {0, 1, 2, 3} (handled by two 2x2x2 Strassen blocks? No, Strassen is 2x2x2)
    # Wait, Strassen multiplies 2x2 * 2x2 -> 2x2.
    # So for A (2x4) and B (4x5):
    # We can split A into A1 (2x2) and A2 (2x2).
    # We can split B into B1 (2x5) and B2 (2x5).
    # C = A1*B1 + A2*B2.
    
    # To use Strassen, we need B1 and B2 to be 2x2. But they are 2x5.
    # We can split B1 into B1a (2x2), B1b (2x2), B1c (2x1).
    # Similarly B2 into B2a (2x2), B2b (2x2), B2c (2x1).
    
    # So:
    # C = A1*B1a + A1*B1b + A1*B1c + A2*B2a + A2*B2b + A2*B2c
    # The 2x2x2 multiplications (A1*B1a, A1*B1b, A2*B2a, A2*B2b) can each use Strassen (rank 7).
    # The 2x2x1 multiplications (A1*B1c, A2*B2c) use schoolbook (rank 2*2*1 = 4 each? No, rank is n*m*p = 2*2*1=4).
    # Wait, for 2x2*2x1, schoolbook rank is 4.
    
    # Total Rank = 4 * 7 (Strassen) + 2 * 4 (Schoolbook for 2x2x1) = 28 + 8 = 36.
    
    # Let's construct the matrices U, V, W.
    # Indices:
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(U, V, W, i_offset, j_offset, k_offset, start_col):
        """
        Adds the Strassen decomposition for a 2x2x2 block.
        i_offset: row index offset in A (0 or 2? No, A is 2x4, so i is 0..1. The block is always 2 rows high.)
        Actually, A is 2x4. The "block" refers to the columns of A and rows of B.
        A_block is A[:, j_offset:j_offset+2]. Shape 2x2.
        B_block is B[j_offset:j_offset+2, k_offset:k_offset+2]. Shape 2x2.
        
        U indices: i in 0..1, j in j_offset..j_offset+1.
        V indices: j in j_offset..j_offset+1, k in k_offset..k_offset+1.
        W indices: k in k_offset..k_offset+1, i in 0..1.
        """
        # Map local indices (0,1) to global indices
        # U[i, j] -> U[i*m + j]
        # V[j, k] -> V[j*p + k]
        # W[k, i] -> W[k*n + i]
        
        # We will fill 7 columns starting at start_col.
        # Each column r corresponds to one of the 7 products M1..M7.
        # U, V, W entries are coefficients.
        
        # Product 1: (A00+A11) * (B00+B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C01 (1), C10 (-1), C11 (1)
        
        # Let's define the contributions to U, V, W for each of the 7 terms.
        # The term is U[i,j] * V[j,k] * W[k,i].
        
        # Term 1: M1
        # U[i, j]:
        #   i=0, j=0 (A00): 1
        #   i=1, j=1 (A11): 1
        # V[j, k]:
        #   j=0, k=0 (B00): 1
        #   j=1, k=1 (B11): 1
        # W[k, i]:
        #   k=0, i=0 (C00): 1
        #   k=1, i=0 (C01): 1
        #   k=0, i=1 (C10): -1
        #   k=1, i=1 (C11): 1
        
        # We need to map these local indices to global.
        # Local i: 0, 1 -> Global i: i_offset + i? 
        # Wait, A is 2x4. The "block" is defined by j_offset.
        # The rows of A are always 0 and 1. So i is 0 and 1.
        # The columns of A are j_offset and j_offset+1.
        # The rows of B are j_offset and j_offset+1.
        # The columns of B are k_offset and k_offset+1.
        # The rows of C (output) are 0 and 1.
        # The columns of C are k_offset and k_offset+1.
        
        # So:
        # U index: i * m + j. i is 0 or 1. j is j_offset or j_offset+1.
        # V index: j * p + k. j is j_offset or j_offset+1. k is k_offset or k_offset+1.
        # W index: k * n + i. k is k_offset or k_offset+1. i is 0 or 1.
        
        # Let's write a helper to set values.
        def set_val(mat, idx, col_idx, val):
            mat[idx][col_idx] += val
            
        # Term 1
        set_val(U, 0*m + j_offset, start_col + 0, 1)   # A00
        set_val(U, 1*m + (j_offset+1), start_col + 0, 1) # A11
        set_val(V, j_offset*p + k_offset, start_col + 0, 1)   # B00
        set_val(V, (j_offset+1)*p + (k_offset+1), start_col + 0, 1) # B11
        set_val(W, k_offset*n + 0, start_col + 0, 1)   # C00
        set_val(W, (k_offset+1)*n + 0, start_col + 0, 1) # C01
        set_val(W, k_offset*n + 1, start_col + 0, -1)   # C10
        set_val(W, (k_offset+1)*n + 1, start_col + 0, 1) # C11
        
        # Term 2: (A10+A11) * B00
        # U: A10 (1), A11 (1)
        # V: B00 (1)
        # W: C10 (1), C11 (-1)
        set_val(U, 1*m + j_offset, start_col + 1, 1)   # A10
        set_val(U, 1*m + (j_offset+1), start_col + 1, 1) # A11
        set_val(V, j_offset*p + k_offset, start_col + 1, 1)   # B00
        set_val(W, k_offset*n + 1, start_col + 1, 1)   # C10
        set_val(W, (k_offset+1)*n + 1, start_col + 1, -1) # C11
        
        # Term 3: A00 * (B01-B11)
        # U: A00 (1)
        # V: B01 (1), B11 (-1)
        # W: C01 (1), C11 (1)
        set_val(U, 0*m + j_offset, start_col + 2, 1)   # A00
        set_val(V, j_offset*p + (k_offset+1), start_col + 2, 1)   # B01
        set_val(V, (j_offset+1)*p + (k_offset+1), start_col + 2, -1) # B11
        set_val(W, (k_offset+1)*n + 0, start_col + 2, 1) # C01
        set_val(W, (k_offset+1)*n + 1, start_col + 2, 1) # C11
        
        # Term 4: A11 * (B10-B00)
        # U: A11 (1)
        # V: B10 (1), B00 (-1)
        # W: C00 (1), C10 (1)
        set_val(U, 1*m + (j_offset+1), start_col + 3, 1) # A11
        set_val(V, (j_offset+1)*p + k_offset, start_col + 3, 1)   # B10
        set_val(V, j_offset*p + k_offset, start_col + 3, -1)   # B00
        set_val(W, k_offset*n + 0, start_col + 3, 1)   # C00
        set_val(W, k_offset*n + 1, start_col + 3, 1)   # C10
        
        # Term 5: (A00+A01) * B11
        # U: A00 (1), A01 (1)
        # V: B11 (1)
        # W: C00 (-1), C01 (1)
        set_val(U, 0*m + j_offset, start_col + 4, 1)   # A00
        set_val(U, 0*m + (j_offset+1), start_col + 4, 1) # A01
        set_val(V, (j_offset+1)*p + (k_offset+1), start_col + 4, 1) # B11
        set_val(W, k_offset*n + 0, start_col + 4, -1)   # C00
        set_val(W, (k_offset+1)*n + 0, start_col + 4, 1) # C01
        
        # Term 6: (A10-A00) * (B00+B01)
        # U: A10 (1), A00 (-1)
        # V: B00 (1), B01 (1)
        # W: C11 (1)
        set_val(U, 1*m + j_offset, start_col + 5, 1)   # A10
        set_val(U, 0*m + j_offset, start_col + 5, -1)   # A00
        set_val(V, j_offset*p + k_offset, start_col + 5, 1)   # B00
        set_val(V, j_offset*p + (k_offset+1), start_col + 5, 1)   # B01
        set_val(W, (k_offset+1)*n + 1, start_col + 5, 1) # C11
        
        # Term 7: (A01-A11) * (B10+B11)
        # U: A01 (1), A11 (-1)
        # V: B10 (1), B11 (1)
        # W: C00 (1)
        set_val(U, 0*m + (j_offset+1), start_col + 6, 1) # A01
        set_val(U, 1*m + (j_offset+1), start_col + 6, -1) # A11
        set_val(V, (j_offset+1)*p + k_offset, start_col + 6, 1)   # B10
        set_val(V, (j_offset+1)*p + (k_offset+1), start_col + 6, 1) # B11
        set_val(W, k_offset*n + 0, start_col + 6, 1)   # C00
        
        return start_col + 7

    # Block 1: j_offset=0, k_offset=0
    col = add_strassen_block(U, V, W, 0, 0, 0, col)
    # Block 2: j_offset=0, k_offset=2
    col = add_strassen_block(U, V, W, 0, 0, 2, col)
    # Block 3: j_offset=2, k_offset=0
    col = add_strassen_block(U, V, W, 0, 2, 0, col)
    # Block 4: j_offset=2, k_offset=2
    col = add_strassen_block(U, V, W, 0, 2, 2, col)
    
    # Now handle the remaining column k=4.
    # This involves A (2x4) * B_col (4x1).
    # We can split this into A1 (2x2) * B1_col (2x1) and A2 (2x2) * B2_col (2x1).
    # Schoolbook for 2x2 * 2x1 has rank 4.
    # Total 8 columns.
    
    # A1 is A[:, 0:2], B1_col is B[0:2, 4]
    # A2 is A[:, 2:4], B2_col is B[2:4, 4]
    
    # Schoolbook for 2x2 * 2x1:
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    # Rank 4 terms:
    # 1. A00 * B00 -> C0
    # 2. A01 * B10 -> C0
    # 3. A10 * B00 -> C1
    # 4. A11 * B10 -> C1
    
    def add_schoolbook_2x2x1(U, V, W, j_offset, k, start_col):
        """
        Adds schoolbook decomposition for A[:, j_offset:j_offset+2] * B[j_offset:j_offset+2, k].
        """
        # Term 1: A00 * B00
        # U: A00 (1) -> i=0, j=j_offset
        # V: B00 (1) -> j=j_offset, k=k
        # W: C0 (1) -> k=k, i=0
        set_val(U, 0*m + j_offset, start_col + 0, 1)
        set_val(V, j_offset*p + k, start_col + 0, 1)
        set_val(W, k*n + 0, start_col + 0, 1)
        
        # Term 2: A01 * B10
        # U: A01 (1) -> i=0, j=j_offset+1
        # V: B10 (1) -> j=j_offset+1, k=k
        # W: C0 (1) -> k=k, i=0
        set_val(U, 0*m + (j_offset+1), start_col + 1, 1)
        set_val(V, (j_offset+1)*p + k, start_col + 1, 1)
        set_val(W, k*n + 0, start_col + 1, 1)
        
        # Term 3: A10 * B00
        # U: A10 (1) -> i=1, j=j_offset
        # V: B00 (1) -> j=j_offset, k=k
        # W: C1 (1) -> k=k, i=1
        set_val(U, 1*m + j_offset, start_col + 2, 1)
        set_val(V, j_offset*p + k, start_col + 2, 1)
        set_val(W, k*n + 1, start_col + 2, 1)
        
        # Term 4: A11 * B10
        # U: A11 (1) -> i=1, j=j_offset+1
        # V: B10 (1) -> j=j_offset+1, k=k
        # W: C1 (1) -> k=k, i=1
        set_val(U, 1*m + (j_offset+1), start_col + 3, 1)
        set_val(V, (j_offset+1)*p + k, start_col + 3, 1)
        set_val(W, k*n + 1, start_col + 3, 1)
        
        return start_col + 4
    
    # Add schoolbook for j_offset=0, k=4
    col = add_schoolbook_2x2x1(U, V, W, 0, 4, col)
    # Add schoolbook for j_offset=2, k=4
    col = add_schoolbook_2x2x1(U, V, W, 2, 4, col)
    
    # col should be 36 now.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
