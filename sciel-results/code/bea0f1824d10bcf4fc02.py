import numpy as np
from itertools import product
import time

# EVOLVE-BLOCK-START
DIMENSIONS = [2, 4, 5]
DEADLINE_SECONDS = 145  # Safety margin before 150s kill

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    return np.allclose(D, T, atol=tol)

def _build_block_strassen_2x4x5():
    """
    Build a rank-36 decomposition using Block-Strassen structure.
    
    For 2x4x5: split the 4x4 middle into 2x2 blocks.
    We can view this as multiplying two 2x2 blocks of the A matrix
    with 2x2 blocks of the B matrix, where each "element" is a 2x5 block.
    
    Using Strassen's 7-term algorithm for 2x2x2, applied to the block structure:
    A = [[A00, A01], [A10, A11]] where each Aij is 1x4 (row of A)
    B = [[B00, B01, B02, B03, B04], [B10, B11, B12, B13, B14]] where each Bij is 1x1
    
    Actually, let's think more carefully:
    A is 2x4, B is 4x5.
    Split A into left (2x2) and right (2x2) halves: A = [A_L | A_R]
    Split B into top (2x5) and bottom (2x5) halves: B = [B_T; B_B]
    
    Product C = A_L * B_T + A_R * B_B
    
    We can use Strassen on the 2x2x2 block multiplication if we view:
    M1 = A_L (2x2), M2 = B_T (2x5)... but B_T is 2x5, not 2x2.
    
    Let me reconsider. The standard approach:
    View the 4x4 middle dimension as 2x2 blocks.
    A is 2x4 = 2x(2*2), so A has 2 columns of 2x2 blocks? No.
    
    Actually, let's use a different decomposition.
    Think of it as: we have n=2 rows in A, m=4 columns in A = 4 rows in B, p=5 cols in B.
    
    Split the 4 middle dimension into two groups of 2:
    Group 0: indices 0,1
    Group 1: indices 2,3
    
    A = [A_0 | A_1] where A_0 is 2x2, A_1 is 2x2
    B = [B_0; B_1] where B_0 is 2x5, B_1 is 2x5
    
    C = A_0 * B_0 + A_1 * B_1
    
    Now, A_0 * B_0 is a 2x2 times 2x5 multiplication.
    We can further decompose this using Strassen-like methods.
    
    For 2x2 * 2x5:
    Split A_0 = [[a, b], [c, d]]
    Split B_0 = [[e, f, g, h, i], [j, k, l, m, n]]
    
    Standard: 4 multiplications: ae, af, ag, ah, ai, cj, ck, cl, cm, cn, de, df, dg, dh, di, je...
    Wait, that's 4*5 = 20 scalar mults for the 2x2*2x5.
    
    For both halves: 40 total, which is schoolbook.
    
    To use Strassen, we need square blocks. Let me try a different approach.
    
    Actually, the known result is that 2x4x5 has rank 36. Let me construct it directly
    using the known Strassen-based decomposition.
    
    The idea: pad to 2x4x4 (which has a known Strassen-based rank), then extend to 2x4x5.
    
    For 2x4x4: This can be seen as 2x(4x4). The 4x4 matrix multiplication can be done
    with Strassen's algorithm (rank 7 for 2x2x2, so 4x4x4 has rank <= 4*7=28 using recursion,
    but actually 4x4x4 has known rank 26 or so).
    
    Hmm, this is getting complex. Let me try a more direct approach.
    
    Alternative: Use the fact that the tensor for 2x4x5 can be decomposed by
    treating it as a sum of two 2x2x5 sub-tensors plus cross terms.
    
    Actually, let me just implement the known rank-36 construction directly.
    
    The construction uses the following idea:
    - Split the 4-dimensional middle space into two 2-dimensional subspaces
    - Use Strassen's 7-term algorithm on the 2x2x2 "core"
    - Handle the 5th dimension separately
    
    For the 2x2x2 core (using Strassen):
    M1 = (A11 - A00) * (B00 + B11)
    M2 = (A00 + A11) * B10
    M3 = A10 * (B01 - B11)
    M4 = (A11 - A10) * (B00 + B10)
    M5 = A00 * (B11 - B00)
    M6 = (A01 - A11) * (B10 + B11)
    M7 = (A00 + A01) * B01
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M7
    C10 = M2 + M5
    C11 = M1 - M2 + M3 + M6
    
    For 2x4x5, we need to extend this. Let me think of A as 2x4 and B as 4x5.
    
    Split A into A_left (2x2) and A_right (2x2).
    Split B into B_top (2x5) and B_bottom (2x5).
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 * 2x5):
    We can use a 7-term Strassen-like method if we pad B_top to 2x2... but that doesn't work directly.
    
    Let me try yet another approach. The known rank-36 for 2x4x5 comes from:
    rank(2x4x5) <= rank(2x2x2) * 2 + rank(2x2x1) * 5 or similar.
    
    Actually, I recall that the standard construction gives:
    For 2xm x mxp, if m is even, we can use Strassen on the m/2 x m/2 blocks.
    
    For 2x4x5: m=4, so m/2=2.
    We have two 2x2 blocks in A and two 2x5 blocks in B.
    
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x5
    
    C = A1*B1 + A2*B2
    
    Now, A1*B1 is 2x2 * 2x5. We can write this as:
    A1 = [[a,b],[c,d]], B1 = rows r1, r2 of B1 (each 1x5)
    A1*B1 = [[a*r1 + b*r2], [c*r1 + d*r2]]
    
    This requires 4 multiplications of 1x5 vectors (i.e., 4*5=20 scalar mults).
    Similarly for A2*B2: another 20. Total 40.
    
    To reduce, we need to share computations. Strassen helps when both matrices are square.
    
    Let me try padding: treat B1 as 2x2 by only using first 2 columns, and handle columns 3-5 separately.
    
    B1 = [B1_2 | B1_3] where B1_2 is 2x2, B1_3 is 2x3
    A1*B1 = A1*B1_2 + A1*B1_3
    
    A1*B1_2 can be done with 7 Strassen terms (each term is a 1x1 * 1x1 = scalar, so 7 scalar mults)
    Wait no, each Strassen term involves multiplying 2x2 sub-blocks... 
    
    I'm overcomplicating this. Let me just hard-code the known rank-36 decomposition
    based on the literature.
    
    The 2x4x5 multiplication tensor has rank 36. The construction is:
    - Use the 7-term Strassen algorithm for the 2x2x2 part
    - Extend to handle the 4x5 structure
    
    Let me define it explicitly. The indices are:
    A[i,j] for i in {0,1}, j in {0,1,2,3}
    B[j,k] for j in {0,1,2,3}, k in {0,1,2,3,4}
    C[i,k] for i in {0,1}, k in {0,1,2,3,4}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2,3}:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group 1: j in {0,1}, Group 2: j in {2,3}
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    For group 1, define A1[i,:] = [A[i,0], A[i,1]] (2x2) and B1[:,k] = [B[0,k], B[1,k]] (2x5)
    For group 2, define A2[i,:] = [A[i,2], A[i,3]] (2x2) and B2[:,k] = [B[2,k], B[3,k]] (2x5)
    
    C = A1*B1 + A2*B2
    
    Now, for A1*B1 where A1 is 2x2 and B1 is 2x5:
    We can use a "Strassen-like" decomposition. Pad B1 to 2x2 by considering only first 2 columns,
    then handle the rest.
    
    B1 = [B1a | B1b] where B1a is 2x2 (cols 0,1) and B1b is 2x3 (cols 2,3,4)
    
    A1*B1 = A1*B1a + A1*B1b
    
    A1*B1a: 2x2 * 2x2, use Strassen (7 terms)
    A1*B1b: 2x2 * 2x3, use 4 terms (standard) -> 4 terms
    
    Similarly for A2*B2: 7 + 4 = 11 terms
    
    Total: 7 + 4 + 7 + 4 = 22 terms? That seems too low. Let me check.
    
    Wait, each "term" in the 2x2*2x5 case involves multiplying a 1x1 (scalar from A) with a 1x5 (row from B),
    which is 5 scalar multiplications. So:
    
    A1*B1a: 7 Strassen terms, each is scalar * scalar (since we're doing 2x2*2x2 at the block level,
    but each block element is a scalar here since n=2, p=5 means each "block" is 1x1 for A and 1x1 for B... 
    
    I think I'm confusing myself. Let me restart with a cleaner formulation.
    
    The multiplication tensor T for 2x4x5 has entries T[a,b,c] where:
    a = i*4 + j (i in 0..1, j in 0..3)
    b = j*5 + k (j in 0..3, k in 0..4)  
    c = k*2 + i (k in 0..4, i in 0..1)
    
    T[a,b,c] = 1 when a=i*4+j, b=j*5+k, c=k*2+i
    
    Schoolbook: 2*4*5 = 40 terms, one for each (i,j,k).
    
    To get rank 36, we need to find 36 rank-1 tensors whose sum equals T.
    
    The known construction uses the following idea:
    - The 4x5 part can be decomposed using Strassen on 2x2 blocks
    - Specifically, split the 4 into 2+2
    
    Let me define the Strassen terms for the 2x2x2 core:
    For indices (i, j, k) where i in {0,1}, j in {0,1}, k in {0,1}:
    These form a 2x2x2 tensor.
    
    The 4x5 structure: j in {0,1,2,3}, k in {0,1,2,3,4}
    
    Split j: j0 in {0,1}, j1 in {2,3}
    
    For j0 part: (i, j0, k) where i in {0,1}, j0 in {0,1}, k in {0,1,2,3,4}
    This is a 2x2x5 tensor.
    
    For j1 part: (i, j1, k) where i in {0,1}, j1 in {2,3}, k in {0,1,2,3,4}
    Also 2x2x5.
    
    Each 2x2x5 tensor can be decomposed. The 2x2x5 tensor has rank at most 2*5=10 (schoolbook).
    But we can do better.
    
    For 2x2x5: A is 2x2, B is 2x5, C is 2x5.
    C[i,k] = sum_{j=0,1} A[i,j] * B[j,k]
    
    This is equivalent to: for each k, C[:,k] = A * B[:,k]
    So we're doing 5 independent 2x2 * 1x1 multiplications... no wait.
    
    A is 2x2, B[:,k] is a 2x1 vector, so A * B[:,k] is a 2x1 vector.
    Each such multiplication requires 4 scalar mults (schoolbook) or 7 (Strassen, but Strassen is for 2x2*2x2).
    
    For 2x2 * 2x1: it's just 4 scalar multiplications, no Strassen benefit.
    
    So 2x2x5 has rank 2*2*5 = 20 schoolbook, and I don't see an easy reduction.
    
    Hmm, but the known result says 2x4x5 has rank 36, which is 40 - 4.
    
    Let me look at this differently. Maybe the reduction comes from the fact that
    we can share some Strassen terms between the two 2x2x5 blocks.
    
    Actually, I think the correct approach is:
    
    View the 4x4 middle as a 2x2 block matrix where each block is 2x2.
    A (2x4) can be seen as having 2 blocks of 2x2? No, A is 2x4, so it's 1x2 blocks of 2x2.
    
    I think I need to just implement the known rank-36 construction from the literature.
    Let me define it based on the Strassen algorithm applied to the 2x2x2 sub-problems.
    
    Here's my best guess at the construction:
    
    The 2x4x5 tensor can be written as a sum of:
    - 7 terms from Strassen on the "left" 2x2x2 part (j in {0,1}, k in {0,1})
    - 7 terms from Strassen on the "right" 2x2x2 part (j in {2,3}, k in {0,1})  
    - 10 terms for the remaining k in {2,3,4} parts
    
    7 + 7 + 10 = 24? That's too low.
    
    Let me try: 7 + 7 + 4 + 4 + 4 = 26? Still not 36.
    
    I think the correct decomposition is:
    - For each of the 5 values of k:
      - If k in {0,1}: use 7 Strassen terms for the 2x2 (in j) part
      - If k in {2,3,4}: use 4 schoolbook terms
    
    5 * 7 = 35? Close to 36.
    
    Or: 2 * 7 + 3 * 4 = 14 + 12 = 26. No.
    
    Let me try: the reduction of 4 comes from using Strassen on two 2x2x2 sub-problems
    instead of 4x4 schoolbook terms.
    
    2x2x2 has rank 7 (Strassen) vs 8 (schoolbook), saving 1 per sub-problem.
    If we have 4 such sub-problems (2 in j-dim, 2 in k-dim), we save 4 total: 40 - 4 = 36.
    
    Yes! That's it.
    
    The 4x5 structure: j in {0,1,2,3}, k in {0,1,2,3,4}
    Split j into {0,1} and {2,3}
    Split k into {0,1} and {2,3,4}
    
    Four sub-problems:
    1. j in {0,1}, k in {0,1}: 2x2x2 -> rank 7 (Strassen)
    2. j in {2,3}, k in {0,1}: 2x2x2 -> rank 7 (Strassen)
    3. j in {0,1}, k in {2,3,4}: 2x2x3 -> rank 6 (schoolbook: 2*2*3=6)
    4. j in {2,3}, k in {2,3,4}: 2x2x3 -> rank 6 (schoolbook: 2*2*3=6)
    
    Total: 7 + 7 + 6 + 6 = 26. That's way less than 36.
    
    Wait, that can't be right. The rank of 2x4x5 is at least 36 according to known results.
    Let me double-check.
    
    Actually, I may have the dimensions wrong. Let me re-read the task.
    "multiplying a 2 by 4 matrix by a 4 by 5 matrix"
    So A is 2x4, B is 4x5, C is 2x5.
    
    The tensor is T[i,j,k] where i indexes rows of A (0,1), j indexes cols of A/rows of B (0,1,2,3),
    k indexes cols of B (0,1,2,3,4).
    
    T[i,j,k] contributes to C[i,k] = sum_j A[i,j]*B[j,k].
    
    The tensor space is (n*m) x (m*p) x (p*n) = 8 x 20 x 10.
    
    Schoolbook rank: 2*4*5 = 40.
    
    The known rank of 2x4x5 matrix multiplication: I believe it's 36.
    
    The construction I described (26) is clearly wrong because it's too optimistic.
    The issue is that the "sub-problems" I defined aren't independent in the way I thought.
    
    Let me reconsider. The Strassen algorithm works for 2x2x2 because the 8 schoolbook terms
    can be replaced by 7. For larger problems, you can apply Strassen recursively or in blocks.
    
    For 2x4x5:
    - The 4 in the middle can be split into 2+2
    - The 5 can be split into 2+3
    
    When we apply Strassen to the 2x2x2 blocks, we save 1 term per 2x2x2 block.
    
    Number of 2x2x2 blocks: (4/2) * (5/2) = 2 * 2 = 4 blocks (with the last k-block being 2x2, and we have one extra column).
    
    Actually: j has 4 elements -> 2 blocks of 2. k has 5 elements -> 2 blocks of 2 + 1 leftover.
    
    So: 2 * 2 = 4 full 2x2x2 blocks, plus 2 * 1 = 2 blocks of 2x2x1.
    
    Each 2x2x2 block: rank 7 (Strassen) instead of 8 (schoolbook), saving 1.
    Each 2x2x1 block: rank 4 (schoolbook, no Strassen benefit since one dimension is 1).
    
    Total: 4*7 + 2*4 = 28 + 8 = 36. Yes!
    
    So the construction is:
    - 4 Strassen 2x2x2 decompositions (for j-blocks {0,1},{2,3} and k-blocks {0,1},{2,3})
    - 2 schoolbook 2x2x1 decompositions (for j-blocks {0,1},{2,3} and k-block {4})
    
    Total rank: 4*7 + 2*4 = 36.
    
    Now let me implement this.
    
    For each (j_block, k_block) pair, we have a 2x2x2 or 2x2x1 sub-problem.
    
    j_block 0: j in {0, 1}
    j_block 1: j in {2, 3}
    
    k_block 0: k in {0, 1}
    k_block 1: k in {2, 3}
    k_block 2: k in {4}
    
    For (j_block, k_block) = (0,0): j in {0,1}, k in {0,1}
    This is a 2x2x2 sub-problem. A_sub[i, j_local] = A[i, j_block*2 + j_local]
    B_sub[j_local, k_local] = B[j_block*2 + j_local, k_block*2 + k_local]
    C_sub[i, k_local] = sum_{j_local} A_sub[i,j_local] * B_sub[j_local,k_local]
    
    Apply Strassen to this 2x2x2.
    
    Similarly for other blocks.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Strassen 2x2x2 terms
    # M1 = (A11 - A00) * (B00 + B11)
    # M2 = (A00 + A11) * B10
    # M3 = A10 * (B01 - B11)
    # M4 = (A11 - A10) * (B00 + B10)
    # M5 = A00 * (B11 - B00)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 + A01) * B01
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each "multiplication" M corresponds to a rank-1 tensor.
    # M = (linear combo of A) * (linear combo of B)
    # This gives a rank-1 tensor in the A-B-C space.
    
    # Let's define the Strassen terms for a 2x2x2 sub-problem with local indices.
    # The sub-problem has A_sub (2x2), B_sub (2x2), C_sub (2x2).
    # Global indices: A[i, j_global], B[j_global, k_global], C[i, k_global]
    
    def strassen_2x2x2(j_base, k_base, U, V, W, col_offset):
        """Add 7 Strassen terms for the 2x2x2 sub-problem at j_base, k_base."""
        # Local indices: i in {0,1}, jl in {0,1}, kl in {0,1}
        # Global: j = j_base + jl, k = k_base + kl
        
        # M1 = (A[1, j_base+1] - A[0, j_base+1]) * (B[j_base, k_base] + B[j_base+1, k_base+1])
        # Wait, I need to be careful with the Strassen formula.
        
        # Standard Strassen for C = A*B where A,B,C are 2x2:
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        #
        # M1 = (A11 - A00) * (B00 + B11)
        # M2 = (A00 + A11) * B10
        # M3 = A10 * (B01 - B11)
        # M4 = (A11 - A10) * (B00 + B10)
        # M5 = A00 * (B11 - B00)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A00 + A01) * B01
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # In our context, A[i, j] where i is row of A (global i in {0,1}), j is column of A.
        # The sub-problem A_sub has rows i in {0,1} and columns jl in {0,1}.
        # So A_sub[0,0] = A[0, j_base+0], A_sub[0,1] = A[0, j_base+1]
        # A_sub[1,0] = A[1, j_base+0], A_sub[1,1] = A[1, j_base+1]
        
        # B_sub[jl, kl] where jl in {0,1}, kl in {0,1}
        # B_sub[0,0] = B[j_base+0, k_base+0], B_sub[0,1] = B[j_base+0, k_base+1]
        # B_sub[1,0] = B[j_base+1, k_base+0], B_sub[1,1] = B[j_base+1, k_base+1]
        
        # C_sub[i, kl] where i in {0,1}, kl in {0,1}
        # C_sub[0,0] = C[0, k_base+0], C_sub[0,1] = C[0, k_base+1]
        # C_sub[1,0] = C[1, k_base+0], C_sub[1,1] = C[1, k_base+1]
        
        terms = []
        
        # M1 = (A11 - A00) * (B00 + B11)
        # A11 = A[1, j_base+1], A00 = A[0, j_base+0]
        # B00 = B[j_base+0, k_base+0], B11 = B[j_base+1, k_base+1]
        # Coefficient for A[1, j_base+1]: +1
        # Coefficient for A[0, j_base+0]: -1
        # Coefficient for B[j_base+0, k_base+0]: +1
        # Coefficient for B[j_base+1, k_base+1]: +1
        # C contributions: C00 += M1, C11 += M1
        # C[0, k_base+0] += M1, C[1, k_base+1] += M1
        terms.append({
            'U': {1*(n) + (j_base+1): 1, 0*n + (j_base+0): -1},  # A indices: i*m + j
            'V': {(j_base+0)*p + (k_base+0): 1, (j_base+1)*p + (k_base+1): 1},  # B indices: j*p + k
            'W': {0*p + 0: 1, 1*p + 1: 1},  # C indices: k*n + i
        })
        
        # M2 = (A00 + A11) * B10
        # A00 = A[0, j_base+0], A11 = A[1, j_base+1]
        # B10 = B[j_base+1, k_base+0]
        # C10 += M2
        terms.append({
            'U': {0*n + (j_base+0): 1, 1*n + (j_base+1): 1},
            'V': {(j_base+1)*p + (k_base+0): 1},
            'W': {0*p + 1: 1},  # C[1, k_base+0]
        })
        
        # M3 = A10 * (B01 - B11)
        # A10 = A[1, j_base+0]
        # B01 = B[j_base+0, k_base+1], B11 = B[j_base+1, k_base+1]
        # C01 += M3
        terms.append({
            'U': {1*n + (j_base+0): 1},
            'V': {(j_base+0)*p + (k_base+1): 1, (j_base+1)*p + (k_base+1): -1},
            'W': {1*p + 0: 1},  # C[0, k_base+1]
        })
        
        # M4 = (A11 - A10) * (B00 + B10)
        # A11 = A[1, j_base+1], A10 = A[1, j_base+0]
        # B00 = B[j_base+0, k_base+0], B10 = B[j_base+1, k_base+0]
        # C00 += M4
        terms.append({
            'U': {1*n + (j_base+1): 1, 1*n + (j_base+0): -1},
            'V': {(j_base+0)*p + (k_base+0): 1, (j_base+1)*p + (k_base+0): 1},
            'W': {0*p + 0: 1},  # C[0, k_base+0]
        })
        
        # M5 = A00 * (B11 - B00)
        # A00 = A[0, j_base+0]
        # B11 = B[j_base+1, k_base+1], B00 = B[j_base+0, k_base+0]
        # C10 += M5
        terms.append({
            'U': {0*n + (j_base+0): 1},
            'V': {(j_base+1)*p + (k_base+1): 1, (j_base+0)*p + (k_base+0): -1},
            'W': {0*p + 1: 1},  # C[1, k_base+0]
        })
        
        # M6 = (A01 - A11) * (B10 + B11)
        # A01 = A[0, j_base+1], A11 = A[1, j_base+1]
        # B10 = B[j_base+1, k_base+0], B11 = B[j_base+1, k_base+1]
        # C11 += M6
        terms.append({
            'U': {0*n + (j_base+1): 1, 1*n + (j_base+1): -1},
            'V': {(j_base+1)*p + (k_base+0): 1, (j_base+1)*p + (k_base+1): 1},
            'W': {1*p + 1: 1},  # C[1, k_base+1]
        })
        
        # M7 = (A00 + A01) * B01
        # A00 = A[0, j_base+0], A01 = A[0, j_base+1]
        # B01 = B[j_base+0, k_base+1]
        # C01 += M7
        terms.append({
            'U': {0*n + (j_base+0): 1, 0*n + (j_base+1): 1},
            'V': {(j_base+0)*p + (k_base+1): 1},
            'W': {1*p + 0: 1},  # C[0, k_base+1]
        })
        
        for term in terms:
            for a_idx, coeff in term['U'].items():
                U[a_idx][col_offset] += coeff
            for b_idx, coeff in term['V'].items():
                V[b_idx][col_offset] += coeff
            for c_idx, coeff in term['W'].items():
                W[c_idx][col_offset] += coeff
            col_offset += 1
        
        return col_offset
    
    def schoolbook_2x2x1(j_base, k_global, U, V, W, col_offset):
        """Add 4 schoolbook terms for the 2x2x1 sub-problem at j_base, k_global."""
        # A_sub[i, jl] for i in {0,1}, jl in {0,1}
        # B_sub[jl, 0] = B[j_base+jl, k_global]
        # C_sub[i, 0] = C[i, k_global]
        
        for i in range(2):
            for jl in range(2):
                j_global = j_base + jl
                U[i*m + j_global][col_offset] = 1
                V[j_global*p + k_global][col_offset] = 1
                W[k_global*n + i][col_offset] = 1
                col_offset += 1
        
        return col_offset
    
    # Build the decomposition
    total_rank = 4*7 + 2*4  # 36
    U = [[0] * total_rank for _ in range(n*m)]
    V = [[0] * total_rank for _ in range(m*p)]
    W = [[0] * total_rank for _ in range(p*n)]
    
    col = 0
    # Four 2x2x2 Strassen blocks
    for j_block in range(2):
        for k_block in range(2):
            col = strassen_2x2x2(j_block*2, k_block*2, U, V, W, col)
    
    # Two 2x2x1 schoolbook blocks for k=4
    for j_block in range(2):
        col = schoolbook_2x2x1(j_block*2, 4, U, V, W, col)
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    
    # Try the Block-Strassen construction (rank 36)
    try:
        result = _build_block_strassen_2x4x5()
        if _verify(result['U'], result['V'], result['W']):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 40)
    result = _build_schoolbook()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    raise ValueError("No valid decomposition found")
# EVOLVE-BLOCK-END
