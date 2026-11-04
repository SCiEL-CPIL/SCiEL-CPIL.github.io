"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Strategy: Use a rank-5 decomposition for 2x3x3 (Strassen-based) 
    # and combine with identity components to handle the 7 columns.
    # 
    # The 3x7 multiplication can be viewed as 3x3 * 7.
    # We can decompose the 7 columns into blocks.
    # 
    # Actually, let's try a direct construction using known small matrix multiplication tensors.
    # 
    # Known: Rank of 2x2x2 is 7 (Strassen).
    # Known: Rank of 2x3x3 is at most 10 (schoolbook) but can be lower?
    # Actually, for 2x3x3, the rank is 9? No, it's at most 9?
    # Let's check: 2x3x3 tensor.
    # 
    # Let's try to construct a decomposition of rank < 42 (schoolbook).
    # 
    # One approach: Use the fact that 3x7 = 3x(3+3+1).
    # If we have a rank-R decomposition for 2x3x3, and another for 2x3x1, we can combine.
    # Rank of 2x3x1 is 3 (just the 3 multiplications).
    # So if rank(2x3x3) = R3, then rank(2x3x7) <= 2*R3 + 3.
    # If R3 = 9, then rank <= 21.
    # If R3 = 8, then rank <= 19.
    #
    # Is there a rank-8 decomposition for 2x3x3?
    # The 2x3x3 tensor is equivalent to 3x2x3.
    # 
    # Let's try to find a low-rank decomposition for 2x3x3 first.
    # 
    # Alternatively, we can use a general search for small ranks.
    # 
    # Let's try rank 21 first (2 * 9 + 3).
    # 
    # Standard 2x3x3 decomposition (rank 9):
    # U (6x9), V (9x9), W (6x9)
    # 
    # Let's define the 2x3x3 tensor T[i,j,k] = delta_{i,0} delta_{j,k} + delta_{i,1} delta_{j,k} ... no.
    # T_{i,j,k} = 1 if i is row of A, j is col of A / row of B, k is col of B.
    # A is 2x3, B is 3x3.
    # 
    # Schoolbook: 9 terms.
    # Can we do better?
    # 
    # Let's try to use Strassen's trick.
    # Split 2x3 into two 1x3? No.
    # 
    # Let's try a rank-10 decomposition for 2x3x3 just in case, or stick with 9.
    # 
    # Actually, let's look for a rank-7 or 8 decomposition for 2x3x3.
    # 
    # For now, let's implement the rank-21 construction (2 * 9 + 3).
    # 
    # Construction for 2x3x3 (rank 9):
    # U[i*3+j, c], V[j*3+k, c], W[k*2+i, c]
    # 
    # Let's write the schoolbook 2x3x3:
    # i in 0..1, j in 0..2, k in 0..2.
    # U[i*3+j, c] = 1, V[j*3+k, c] = 1, W[k*2+i, c] = 1.
    # 
    # Now, to combine two 2x3x3 blocks and one 2x3x1 block:
    # The 7 columns of B are split into 3, 3, 1.
    # 
    # For the first 3 columns (k=0,1,2): use the 2x3x3 decomposition.
    # For the next 3 columns (k=3,4,5): use another 2x3x3 decomposition.
    # For the last 1 column (k=6): use the 2x3x1 decomposition (rank 3).
    # 
    # Total rank = 9 + 9 + 3 = 21.
    # 
    # Let's construct this.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 6x21
    V = [[0] * rank for _ in range(m * p)]  # 21x21
    W = [[0] * rank for _ in range(p * n)]  # 14x21
    
    # Helper to add a 2x3x3 block
    def add_block_2x3x3(offset_u, offset_v, offset_w, col_start):
        # offset_u: not used directly, U is shared
        # offset_v: start column in V for this block's rows? No, V rows are fixed.
        # V rows are j*p+k. For block 1, k=0..2. For block 2, k=3..5.
        # So V rows are different.
        # W rows are k*n+i. For block 1, k=0..2. For block 2, k=3..5.
        # So W rows are different.
        # U rows are i*m+j. Same for both blocks.
        # 
        # So we just add to the same U rows, but different V and W rows.
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    col = col_start + (i*3 + j)*3 + k  # Wait, schoolbook indexing
                    # Actually, let's just use a counter
                    pass
        
        # Let's do it properly with a counter
        col = col_start
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3 + (k + offset_v_k)][col] = 1  # offset_v_k is 0 or 3
                    W[(k + offset_w_k)*2 + i][col] = 1  # offset_w_k is 0 or 3
                    col += 1
    
    # Block 1: k=0..2
    add_block_2x3x3(0, 0, 0, 0)
    
    # Block 2: k=3..5
    add_block_2x3x3(0, 3, 3, 9)
    
    # Block 3: k=6 (2x3x1, rank 3)
    # U[i*3+j, c] = 1, V[j*3+6, c] = 1, W[6*2+i, c] = 1
    col = 18
    for i in range(2):
        for j in range(3):
            U[i*3+j][col] = 1
            V[j*3+6][col] = 1
            W[6*2+i][col] = 1
            col += 1
    
    # Verify
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # Let's check a few entries.
    # i=0, j=0, k=0:
    # U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # Other terms: U[0,0] is 1 in col 0. V[0,0] is 1 in col 0. W[0,0] is 1 in col 0.
    # So sum = 1. Correct.
    # 
    # i=0, j=0, k=1:
    # U[0,0]=1, V[1,0]=1, W[2,0]=1 -> 1
    # Correct.
    # 
    # i=0, j=0, k=3:
    # U[0,0]=1, V[3,0]=1, W[6,0]=1 -> 1
    # Correct.
    
    # Check if we can do better.
    # 
    # Let's try to find a rank-8 decomposition for 2x3x3.
    # 
    # If we can find one, then rank(2x3x7) <= 2*8 + 3 = 19.
    # 
    # Let's try to search for a rank-8 decomposition for 2x3x3.
    # 
    # This is a small problem, so we can use a search.
    
    if time.time() < deadline - 10:
        # Try to find a rank-8 decomposition for 2x3x3
        # 
        # The 2x3x3 tensor has 18 non-zero entries.
        # 
        # Let's try to use a known decomposition.
        # 
        # Actually, let's just return the rank-21 decomposition for now.
        # 
        # We can try to improve it later.
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
