import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy: Use Strassen-like block decomposition.
    Partition the 2x3 matrix A and 3x5 matrix B into blocks.
    We can view the multiplication as a combination of smaller multiplications.
    
    Let A be 2x3 and B be 3x5.
    We can split A into columns A1 (2x2) and A2 (2x1).
    We can split B into rows B1 (2x5) and B2 (1x5).
    
    C = A * B = [A1, A2] * [B1; B2] = A1*B1 + A2*B2
    
    This is a sum of two independent matrix multiplications:
    1. A1 (2x2) * B1 (2x5) -> Result 2x5
    2. A2 (2x1) * B2 (1x5) -> Result 2x5
    
    Rank of (2x2)*(2x5):
    We can decompose this further.
    Let A1 = [[a, b], [c, d]], B1 = [[e, f, g, h, i], [j, k, l, m, n]].
    Using Strassen for the 2x2 part? Strassen is for 2x2 * 2x2.
    Here we have 2x2 * 2x5.
    We can treat B1 as two 2x2 blocks plus one 2x1 block? Or use general rank.
    
    Actually, the known result for M_{2,3,5} is likely 28 or 29.
    The schoolbook is 30.
    
    Let's construct a specific rank-28 decomposition based on Strassen's idea.
    
    Consider the tensor T[i,j,k] for i in 0..1, j in 0..2, k in 0..4.
    
    A standard way to reduce rank is to use identities.
    For M_{2,2,2}, rank is 7 (Strassen) vs 8 (schoolbook).
    For M_{2,3,5}, we can embed Strassen.
    
    Let's try to build a rank-28 decomposition explicitly.
    If we can't find a known explicit formula, we can try to solve for a rank-29 or 28 decomposition using a structured ansatz.
    
    However, the prompt mentions "inspiration 2 (score 1.0714)" which corresponds to rank 28 (30/28 * 1.0 = 1.0714).
    So a rank-28 decomposition is a valid improvement.
    
    Let's try to find a rank-28 decomposition.
    
    One approach:
    Split the 3x5 matrix B into B_top (2x5) and B_bottom (1x5).
    Split the 2x3 matrix A into A_left (2x2) and A_right (2x1).
    
    C = A_left * B_top + A_right * B_bottom
    
    The term A_right * B_bottom is a 2x1 times 1x5 multiplication.
    Rank of M_{2,1,5} is 2*1*5 = 10? No, it's just outer product of a 2x1 vector and 1x5 vector.
    Wait, A_right is 2x1, B_bottom is 1x5.
    C_part2[i,k] = A_right[i,0] * B_bottom[0,k].
    This is a rank-1 tensor in (i,k) for each j? No, j is fixed to the column index of A_right.
    Actually, the indices are:
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    For the term A_right * B_bottom:
    i in 0..1, j=2 (the third column of A), k in 0..4.
    Contribution to C[i,k] comes from U[i*3+2, c] * V[2*5+k, c] * W[k*2+i, c].
    This part involves 2*1*5 = 10 terms in schoolbook.
    Can we do better?
    The tensor for M_{2,1,5} is T[i, j=2, k] = 1.
    This is just a rank-1 tensor if we ignore the j index? No, j is fixed.
    Actually, the tensor is supported only on j=2.
    It is T[i, 2, k] = 1.
    This can be decomposed into rank 1?
    U[i,2] * V[2,k] * W[k,i] ? No, the indices are coupled.
    For a fixed j=2, we have a 2x5 matrix multiplication by a scalar (since A is 2x1 and B is 1x5, it's just outer product).
    The tensor is T[i, 2, k] = 1.
    We need sum_c U[i,2,c] V[2,k,c] W[k,i,c] = 1.
    This is equivalent to decomposing the all-ones 2x5 matrix as a sum of rank-1 matrices where the factorization is constrained by the W tensor structure?
    Actually, if we set U[i,2,c] = u_i, V[2,k,c] = v_k, W[k,i,c] = w_{ki}.
    Then sum_c u_i v_k w_{ki} = 1.
    If we choose w_{ki} = 1/sqrt(n*p) or something?
    Actually, the simplest is:
    Use 1 term?
    U[i,2,0] = 1, V[2,k,0] = 1, W[k,i,0] = 1?
    Then sum is 1*1*1 = 1.
    So M_{2,1,5} has rank 1?
    Let's check.
    U shape (6, rank). Index i*3+2.
    V shape (15, rank). Index 2*5+k = 10+k.
    W shape (10, rank). Index k*2+i.
    
    If we use 1 column:
    U[i*3+2, 0] = 1 for i=0,1.
    V[10+k, 0] = 1 for k=0..4.
    W[k*2+i, 0] = 1 for all k,i.
    Then for any i,k:
    Sum = U[i*3+2,0] * V[10+k,0] * W[k*2+i,0] = 1*1*1 = 1.
    This works!
    So the part corresponding to j=2 (the last column of A) can be done with rank 1.
    
    Now consider the part A_left (2x2) * B_top (2x5).
    Indices: i in 0..1, j in 0..1, k in 0..4.
    We need to decompose T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..4.
    This is M_{2,2,5}.
    Schoolbook rank is 2*2*5 = 20.
    Can we do better?
    We can split B_top into two 2x2 blocks and one 2x1 block?
    Or use Strassen for the 2x2 part of A and 2x2 part of B?
    
    Let's split B_top (2x5) into B1 (2x2), B2 (2x2), B3 (2x1).
    A_left (2x2) * B_top = A_left*B1 + A_left*B2 + A_left*B3.
    
    Rank of M_{2,2,2} is 7 (Strassen).
    Rank of M_{2,2,1} is 1 (similar to above, rank 1).
    
    So total rank = Rank(M_{2,2,2}) + Rank(M_{2,2,2}) + Rank(M_{2,2,1}) + Rank(M_{2,1,5})
    = 7 + 7 + 1 + 1 = 16?
    
    Wait, the indices must be disjoint in the U, V, W matrices for the columns to be independent.
    In the decomposition, each "term" c corresponds to a rank-1 tensor.
    If we use Strassen for the first 2x2x2 block, it uses 7 columns.
    These columns have non-zero entries in U[i*3+j, c], V[j*5+k, c], W[k*2+i, c].
    
    For the first block (j in 0..1, k in 0..1):
    The U indices are i*3+j for i in 0..1, j in 0..1. (4 entries)
    The V indices are j*5+k for j in 0..1, k in 0..1. (4 entries)
    The W indices are k*2+i for k in 0..1, i in 0..1. (4 entries)
    
    For the second block (j in 0..1, k in 2..3):
    The U indices are the same (i*3+j).
    The V indices are j*5+k for j in 0..1, k in 2..3.
    The W indices are k*2+i for k in 2..3.
    
    For the third block (j in 0..1, k in 4):
    U indices same.
    V indices j*5+4.
    W indices 4*2+i.
    
    For the fourth part (j=2, k in 0..4):
    U index i*3+2.
    V index 2*5+k = 10+k.
    W index k*2+i.
    
    Notice that the U indices for the first three parts overlap (they use j=0,1).
    The U indices for the fourth part are distinct (j=2).
    The V indices for the first three parts use k=0..4, j=0,1.
    The V indices for the fourth part use k=0..4, j=2. Distinct.
    The W indices for the first three parts use k=0..4.
    The W indices for the fourth part use k=0..4. Overlap!
    
    However, we can assign distinct column indices c for each part.
    Part 1 (Strassen 2x2x2): 7 columns.
    Part 2 (Strassen 2x2x2): 7 columns.
    Part 3 (Rank 1 2x2x1): 1 column.
    Part 4 (Rank 1 2x1x5): 1 column.
    
    Total rank = 7 + 7 + 1 + 1 = 16.
    
    Is this valid?
    We must ensure that for each (i,j,k), the sum over all columns c of U[i,j,c]V[j,k,c]W[k,i,c] equals 1 if (i,j,k) is in the support, and 0 otherwise?
    No, the tensor T[i,j,k] is 1 for ALL i,j,k.
    So we need the sum to be 1 for all i in 0..1, j in 0..2, k in 0..4.
    
    Let's check the contributions.
    For a given (i,j,k):
    - If j=2: Only Part 4 contributes.
      Part 4 is designed to give 1 for all i,k when j=2.
      So sum = 1. Correct.
      
    - If j=0 or 1: Only Parts 1, 2, 3 contribute.
      Part 1 handles k=0,1.
      Part 2 handles k=2,3.
      Part 3 handles k=4.
      
      For k=0,1: Part 1 (Strassen) must yield 1 for all i, j in 0..1.
      Strassen's algorithm computes the product of two 2x2 matrices.
      The tensor for matrix multiplication M_{2,2,2} is T[i,j,k] = 1.
      Strassen's decomposition is a sum of 7 rank-1 tensors that equals the tensor.
      So yes, for any i, j in 0..1, k in 0..1, the sum over the 7 Strassen columns will be 1.
      
      For k=2,3: Part 2 (Strassen) yields 1.
      
      For k=4: Part 3 (Rank 1) yields 1.
      
      So for j=0,1, the total sum is 1 (from the relevant part) + 0 (from other parts) = 1.
      
    Thus, the total rank is 16.
    
    Wait, is the rank of M_{2,2,5} really 15? (7+7+1).
    And M_{2,1,5} is 1.
    Total 16.
    
    Let's verify the Strassen decomposition indices.
    Standard Strassen for C = A*B where A,B are 2x2.
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    The tensor entries are coefficients of A_ij and B_kl.
    In our U, V, W formulation:
    U[i,j,c] corresponds to coefficient of A_ij.
    V[j,k,c] corresponds to coefficient of B_jk. (Note: B is 2x2, so j is row, k is col. In our global indices, j is col of A, k is row of B? No.
    Global indices:
    A is 2x3. Indices i (row), j (col).
    B is 3x5. Indices j (row), k (col).
    C is 2x5. Indices i (row), k (col).
    
    So for the sub-problem A_left (2x2) * B_top_left (2x2):
    A indices: i in 0..1, j in 0..1.
    B indices: j in 0..1, k in 0..1.
    C indices: i in 0..1, k in 0..1.
    
    The Strassen formulas use A00, A01, A10, A11 and B00, B01, B10, B11.
    Here A_ij is the element at row i, col j.
    B_jk is the element at row j, col k.
    
    So we can map the Strassen M's to our U, V, W.
    For each M_t, we have:
    U[i,j,t] = coeff of A_ij in M_t.
    V[j,k,t] = coeff of B_jk in M_t.
    W[k,i,t] = coeff of C_ki? No, C is i,k.
    W[k*n+i, t] = coeff of C_ik in M_t?
    Wait, the output C_ik = sum_j A_ij B_jk.
    So the term is A_ij * B_jk * C_ik.
    In the tensor decomposition, we have sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    Here W index is k*n+i. So W[k,i,c] is the coefficient for C_ik.
    
    So for Strassen:
    M1 = (A00+A11)*(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    This contributes to C00, C01, C10, C11?
    Actually, M1 is an intermediate product.
    C00 += M1
    C11 += M1
    
    So for the term corresponding to M1:
    U[0,0] = 1, U[1,1] = 1. Others 0.
    V[0,0] = 1, V[1,1] = 1. Others 0.
    W[0,0] = 1, W[1,1] = 1. Others 0. (Since C00 and C11 get M1).
    
    Let's check the product:
    Sum over i,j,k of U[i,j]V[j,k]W[k,i] ? No, the decomposition is for the tensor T[i,j,k].
    T[i,j,k] = 1 if the term A_ij B_jk contributes to C_ik?
    No, T[i,j,k] is 1 for all i,j,k.
    The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    For M1:
    U[0,0]=1, U[1,1]=1.
    V[0,0]=1, V[1,1]=1.
    W[0,0]=1, W[1,1]=1.
    
    Check (i=0,j=0,k=0): U[0,0]V[0,0]W[0,0] = 1*1*1 = 1.
    Check (i=0,j=0,k=1): U[0,0]V[0,1]W[1,0] = 1*0*0 = 0.
    Check (i=0,j=1,k=0): U[0,1]V[1,0]W[0,0] = 0*0*1 = 0.
    Check (i=1,j=1,k=1): U[1,1]V[1,1]W[1,1] = 1*1*1 = 1.
    
    But we need the sum of all 7 M's to equal 1 for ALL i,j,k in the 2x2x2 block.
    Does Strassen provide such a decomposition?
    Yes, the rank of the matrix multiplication tensor is 7.
    The 7 rank-1 tensors sum to the all-ones tensor of size 2x2x2.
    
    So we can just use the standard Strassen coefficients.
    
    Let's construct the 7 columns for a 2x2x2 block.
    We need to map the local indices (i,j,k) in 0..1 to the global indices.
    
    Block 1: i in 0..1, j in 0..1, k in 0..1.
    Global U index: i*3+j.
    Global V index: j*5+k.
    Global W index: k*2+i.
    
    We will generate the 7 Strassen columns for this block.
    Then Block 2: i in 0..1, j in 0..1, k in 2..3.
    Global U index: i*3+j.
    Global V index: j*5+k.
    Global W index: k*2+i.
    
    Then Block 3: i in 0..1, j in 0..1, k in 4.
    Global U index: i*3+j.
    Global V index: j*5+4.
    Global W index: 4*2+i = 8+i.
    Rank 1 decomposition for 2x2x1.
    U[i,0]=1, U[i,1]=1? No.
    For 2x2x1, T[i,j,k=0] = 1.
    We need sum_c U[i,j,c]V[j,0,c]W[0,i,c] = 1.
    Try 1 column:
    U[i,j,0] = 1 for all i,j?
    V[j,0,0] = 1 for all j?
    W[0,i,0] = 1 for all i?
    Then sum = 1*1*1 = 1.
    Yes.
    
    Block 4: i in 0..1, j in 2, k in 0..4.
    Global U index: i*3+2.
    Global V index: 2*5+k = 10+k.
    Global W index: k*2+i.
    Rank 1 decomposition for 2x1x5.
    U[i,0,0] = 1 for all i.
    V[0,k,0] = 1 for all k.
    W[k,i,0] = 1 for all k,i.
    Sum = 1.
    
    Total Rank = 7 + 7 + 1 + 1 = 16.
    
    Let's implement this.
    
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # Initialize U, V, W with zeros
    # We expect rank 16. Let's allocate 20 to be safe, then trim.
    max_rank = 20
    U = [[0] * max_rank for _ in range(nm)]
    V = [[0] * max_rank for _ in range(mp)]
    W = [[0] * max_rank for _ in range(pn)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # A indices: (0,0), (0,1), (1,0), (1,1) -> i,j
    # B indices: (0,0), (0,1), (1,0), (1,1) -> j,k
    # C indices: (0,0), (0,1), (1,0), (1,1) -> i,k
    
    # For each M, define U[i,j], V[j,k], W[k,i]
    
    def add_strassen_block(offset_j, offset_k, u_offset, v_offset, w_offset):
        """
        Adds 7 Strassen columns for a 2x2x2 block.
        offset_j: starting column index in A (j)
        offset_k: starting row index in B (k)
        """
        nonlocal col
        # Local indices i in 0..1, j in 0..1, k in 0..1
        # Global i in 0..1
        # Global j = offset_j + j_local
        # Global k = offset_k + k_local
        
        # M1
        # U: A00, A11 -> 1
        # V: B00, B11 -> 1
        # W: C00, C11 -> 1
        U[u_offset + 0*2 + 0][col] = 1 # i=0, j=0
        U[u_offset + 1*2 + 1][col] = 1 # i=1, j=1
        V[v_offset + 0*2 + 0][col] = 1 # j=0, k=0
        V[v_offset + 1*2 + 1][col] = 1 # j=1, k=1
        W[w_offset + 0*2 + 0][col] = 1 # k=0, i=0
        W[w_offset + 1*2 + 1][col] = 1 # k=1, i=1
        col += 1
        
        # M2
        # U: A10, A11 -> 1
        # V: B00 -> 1
        # W: C10, C11 -> 1 (C10 += M2, C11 -= M2)
        # Wait, C11 = M1 - M2 + M3 + M6. So coeff is -1.
        U[u_offset + 1*2 + 0][col] = 1 # i=1, j=0
        U[u_offset + 1*2 + 1][col] = 1 # i=1, j=1
        V[v_offset + 0*2 + 0][col] = 1 # j=0, k=0
        W[w_offset + 0*2 + 1][col] = 1 # k=0, i=1
        W[w_offset + 1*2 + 1][col] = -1 # k=1, i=1
        col += 1
        
        # M3
        # U: A00 -> 1
        # V: B01 -> 1, B11 -> -1
        # W: C01 -> 1, C11 -> 1
        U[u_offset + 0*2 + 0][col] = 1 # i=0, j=0
        V[v_offset + 0*2 + 1][col] = 1 # j=0, k=1
        V[v_offset + 1*2 + 1][col] = -1 # j=1, k=1
        W[w_offset + 1*2 + 0][col] = 1 # k=1, i=0
        W[w_offset + 1*2 + 1][col] = 1 # k=1, i=1
        col += 1
        
        # M4
        # U: A11 -> 1
        # V: B10 -> 1, B00 -> -1
        # W: C00 -> 1, C10 -> 1
        U[u_offset + 1*2 + 1][col] = 1 # i=1, j=1
        V[v_offset + 1*2 + 0][col] = 1 # j=1, k=0
        V[v_offset + 0*2 + 0][col] = -1 # j=0, k=0
        W[w_offset + 0*2 + 0][col] = 1 # k=0, i=0
        W[w_offset + 0*2 + 1][col] = 1 # k=0, i=1
        col += 1
        
        # M5
        # U: A00, A01 -> 1
        # V: B11 -> 1
        # W: C00 -> -1, C01 -> 1
        U[u_offset + 0*2 + 0][col] = 1 # i=0, j=0
        U[u_offset + 0*2 + 1][col] = 1 # i=0, j=1
        V[v_offset + 1*2 + 1][col] = 1 # j=1, k=1
        W[w_offset + 0*2 + 0][col] = -1 # k=0, i=0
        W[w_offset + 1*2 + 0][col] = 1 # k=1, i=0
        col += 1
        
        # M6
        # U: A10 -> 1, A00 -> -1
        # V: B00, B01 -> 1
        # W: C11 -> 1
        U[u_offset + 1*2 + 0][col] = 1 # i=1, j=0
        U[u_offset + 0*2 + 0][col] = -1 # i=0, j=0
        V[v_offset + 0*2 + 0][col] = 1 # j=0, k=0
        V[v_offset + 0*2 + 1][col] = 1 # j=0, k=1
        W[w_offset + 1*2 + 1][col] = 1 # k=1, i=1
        col += 1
        
        # M7
        # U: A01 -> 1, A11 -> -1
        # V: B10, B11 -> 1
        # W: C00 -> 1
        U[u_offset + 0*2 + 1][col] = 1 # i=0, j=1
        U[u_offset + 1*2 + 1][col] = -1 # i=1, j=1
        V[v_offset + 1*2 + 0][col] = 1 # j=1, k=0
        V[v_offset + 1*2 + 1][col] = 1 # j=1, k=1
        W[w_offset + 0*2 + 0][col] = 1 # k=0, i=0
        col += 1

    # Block 1: j=0,1, k=0,1
    # U offset: i*3+0. i=0->0, i=1->3.
    # V offset: 0*5+0 = 0.
    # W offset: 0*2+0 = 0.
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: j=0,1, k=2,3
    # U offset: same.
    # V offset: 0*5+2 = 2.
    # W offset: 2*2+0 = 4.
    add_strassen_block(0, 2, 0, 2, 4)
    
    # Block 3: j=0,1, k=4
    # Rank 1
    # U[i, j] = 1 for i in 0..1, j in 0..1
    # V[j, 4] = 1 for j in 0..1
    # W[4, i] = 1 for i in 0..1
    # Global indices:
    # U: i*3+j
    # V: j*5+4
    # W: 4*2+i = 8+i
    U[0*3+0][col] = 1
    U[0*3+1][col] = 1
    U[1*3+0][col] = 1
    U[1*3+1][col] = 1
    V[0*5+4][col] = 1
    V[1*5+4][col] = 1
    W[4*2+0][col] = 1
    W[4*2+1][col] = 1
    col += 1
    
    # Block 4: j=2, k=0..4
    # Rank 1
    # U[i, 2] = 1 for i in 0..1
    # V[2, k] = 1 for k in 0..4
    # W[k, i] = 1 for all k, i
    # Global indices:
    # U: i*3+2
    # V: 2*5+k = 10+k
    # W: k*2+i
    U[0*3+2][col] = 1
    U[1*3+2][col] = 1
    for k in range(5):
        V[2*5+k][col] = 1
        W[k*2+0][col] = 1
        W[k*2+1][col] = 1
    col += 1
    
    # Trim to actual rank
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

