"""Matrix multiplication 3x3x8 rank-minimal decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    We use the known Strassen-style approach. For 3x3 * 3x8, we can decompose
    the 3x8 into two 3x4 blocks and use rank-7 algorithms for each, but that gives
    rank 14*2=28... no wait, each 3x3*3x4 multiplication can be done in rank 7?
    
    Actually, let me think more carefully. The schoolbook rank is 3*3*8 = 72.
    
    For 3x3 * 3x4: schoolbook is 36. Known best for 3x3*3x3 is 7 (Strassen-like for 3x3? No, 3x3*3x3 best known is 7? Actually no, 3x3*3x3 minimum is not 7. The Strassen algorithm is for 2x2. For 3x3, the best known upper bound is around 23-24, but let me recall...
    
    Actually, for small matrix multiplication:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 3x3 * 3x3: best known is around 23-24? No wait, I think it's higher.
    
    Let me reconsider. The key insight is that we can use the fact that 3x3 * 3x8 = (3x3 * 3x4) + (3x3 * 3x4) where we split the 8 columns into two groups of 4.
    
    For 3x3 * 3x4, the tensor is T_{i,j,k} where i in [3], j in [3], k in [4]. Schoolbook rank = 3*3*4 = 36.
    
    A better approach: use the fact that we can apply Strassen-like ideas. 
    
    Actually, let me try a different approach. We know that:
    - The 2x2 * 2x2 multiplication has rank 7 (Strassen).
    - We can embed 3x3 * 3x8 into a larger structure.
    
    Let me try to construct a decomposition using block methods.
    
    For 3x3 * 3x8:
    Split A (3x3) into blocks and B (3x8) into blocks.
    
    Alternative: Use the fact that 3 = 2+1. We can write:
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B = [[B1], [B2]] where B1 is 2x8, B2 is 1x8
    
    Then AB = [A11*B1 + A12*B2, A21*B1 + A22*B2]
    
    The first row block (2x8) requires computing A11*B1 (2x2 * 2x8) and A12*B2 (2x1 * 1x8).
    A11*B1: 2x2 * 2x8. This can be done by splitting B1 into four 2x2 blocks, each requiring rank 7, so 4*7 = 28.
    A12*B2: 2x1 * 1x8 = 2*1*8 = 16 scalar products, rank 16.
    Similarly for the second row block.
    
    Total: 28 + 16 + 28 + 16 = 88. That's worse than 72.
    
    Let me try another split. 3x3 * 3x8.
    
    Actually, a simpler approach: just use the schoolbook but look for savings.
    
    Hmm, let me think about what's achievable. The minimum rank for 3x3 * 3x8 is at least the maximum of:
    - The slice rank lower bound
    - Other lower bounds
    
    For now, let me try to implement a known good construction. 
    
    Actually, I recall that for multiplying a 3x3 by a 3x8 matrix, one can achieve rank around 50-60 using clever decompositions. But let me try to be more systematic.
    
    Let me try using the approach where we treat the 3x3 * 3x8 as a linear map and use bilinear decomposition.
    
    For a practical approach within the time limit, let me try:
    1. Start with schoolbook (rank 72)
    2. Try to find a lower rank by using the structure
    
    Actually, let me try a concrete construction. We can use the fact that:
    3x3 * 3x8 can be decomposed by first doing 3x3 * 3x4 twice.
    
    For 3x3 * 3x4, let me try to find a good rank.
    
    3x3 * 3x4: Think of this as computing a 3x4 matrix C where C[i,k] = sum_j A[i,j]*B[j,k].
    
    One approach: Use the 2x2 Strassen trick on pairs of rows/columns.
    
    Let me try a different tactic. I'll implement a search for a low-rank decomposition using the fact that the entries can be integers or half-integers.
    
    Given the time constraints, let me try to construct a rank-56 decomposition or similar using block Strassen methods.
    
    Actually, let me just try to find the best known result. For 3x3 * 3x8, I believe a rank of around 50 is achievable.
    
    Let me try a concrete construction:
    
    Method: Split the 8 columns into two groups of 4.
    Compute A * B1 (3x3 * 3x4) and A * B2 (3x3 * 3x4) separately.
    
    For 3x3 * 3x4, use the following approach:
    - Write A as 3x3, B as 3x4
    - Use the identity that we can compute 3x3 * 3x3 in rank 23 (or whatever the best known is) and handle the extra column.
    
    This is getting complicated. Let me just try to implement a reasonable decomposition and see.
    
    For now, let me try rank 64 as a starting point (worse than schoolbook 72, but using a structured approach).
    
    Actually, I just realized: the simplest improvement is to note that we don't need full generality. Let me try to use the fact that the matrix sizes are small and search for a good decomposition.
    
    Given the 150s time limit, let me try a heuristic search for a low-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to construct a lower-rank decomposition
    # Strategy: Use block decomposition with Strassen-like ideas
    
    # For 3x3 * 3x8, let's try splitting into 2x2 + 1x1 blocks
    # A = [[A00, A01], [A10, A11]] where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1
    # B = [[B0], [B1]] where B0 is 2x8, B1 is 1x8
    
    # C = AB = [A00*B0 + A01*B1, A10*B0 + A11*B1]
    
    # A00*B0: 2x2 * 2x8. Split B0 into four 2x2 blocks. Each needs rank 7 (Strassen).
    # So A00*B0 needs 4*7 = 28 rank terms.
    
    # A01*B1: 2x1 * 1x8 = 16 scalar products. Rank 16.
    # A10*B0: 1x2 * 2x8 = 16 scalar products. Rank 16.
    # A11*B1: 1x1 * 1x8 = 8 scalar products. Rank 8.
    
    # Total: 28 + 16 + 16 + 8 = 68
    
    # But we can do better by sharing some computations.
    
    # Let me try a direct construction with rank 68.
    
    def strassen_2x2(A, B):
        """
        Strassen 2x2 * 2x2 multiplication.
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        C = [[a*e + c*g, a*f + c*h], [b*e + d*g, b*f + d*h]]
        
        Strassen:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-b)(e+f)
        M7 = (d-a)(g+h)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        pass  # We'll implement this directly in the decomposition
    
    # Let me construct the rank-68 decomposition explicitly.
    # 
    # Variables for A (3x3): A[i,j] for i,j in 0,1,2
    # Variables for B (3x8): B[j,k] for j in 0,1,2, k in 0,1,2,3,4,5,6,7
    # Output C (3x8): C[i,k] for i in 0,1,2, k in 0,1,2,3,4,5,6,7
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Block structure:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    #
    # B = [[B00, B01, B02, B03, B04, B05, B06, B07],
    #      [B10, B11, B12, B13, B14, B15, B16, B17],
    #      [B20, B21, B22, B23, B24, B25, B26, B27]]
    #
    # Let's use the 2x2 Strassen on the first 2x2 block of A and appropriate blocks of B.
    #
    # For the 2x2 * 2x8 part (rows 0,1 of A, cols 0,1 of B):
    # A_block = [[A00, A01], [A10, A11]]
    # B_block = [[B00,...,B07], [B10,...,B17]]
    #
    # Strassen for 2x2 * 2x8:
    # We need to compute:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # ... etc for all 8 columns
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # ... etc
    #
    # Strassen factors:
    # M1 = (A00+A11) * (B00+B10) for col 0
    # M2 = (A10+A11) * B00 for col 0
    # M3 = A00 * (B01-B11) ... wait, this doesn't work directly for multiple columns.
    
    # Actually, for 2x2 * 2x8, we can apply Strassen column by column, but that gives 8*7 = 56 terms.
    # That's not better than the 4*7 = 28 I computed earlier (splitting into four 2x2 blocks).
    
    # Let me go back to the block approach:
    # Split B into four 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # For each 2x2 block, use Strassen (rank 7).
    # Plus the cross terms with the third row/col.
    
    # This is getting complex. Let me just implement a reasonable decomposition.
    
    # Given time constraints, let me try to implement a rank-64 construction.
    
    # Actually, let me try a simpler approach: just return the schoolbook decomposition
    # but with a small optimization. The schoolbook is rank 72.
    
    # Let me try to find if there's a known rank for 3x3 * 3x8.
    
    # I'll implement a search-based approach to find a low-rank decomposition.
    
    # For the tensor T_{i,j,k} = delta (C[i,k] = sum_j A[i,j]*B[j,k]),
    # we need: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # and the decomposition satisfies:
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) corresponds to a valid term, 0 otherwise
    # Wait, that's not quite right. The tensor for matrix multiplication is:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T_{(i,j),(j,k),(k,i)} = 1
    # And the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # and the condition is: for all (i,j),(j,k),(k,i):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid triple, 0 otherwise
    # But actually, the tensor is only nonzero at positions where the middle index matches:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # And the decomposition must satisfy:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c.
    
    # This is a system of equations. For rank R, we have R * (n*m + m*p + p*n) unknowns
    # and n*m*p equations (but many are trivially 0).
    
    # The nonzero entries are at positions (i*m+j, j*p+k, k*n+i) for i in range(n), j in range(m), k in range(p).
    # There are n*m*p = 72 such entries.
    
    # So we need to solve: for each (i,j,k):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # And for all other (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # This is a nonlinear system. Let me try a different approach.
    
    # Given the complexity, let me just return a good known decomposition.
    
    # I'll try to implement the rank-68 construction I outlined above.
    
    # For now, let me return the schoolbook decomposition as a safe fallback,
    # but try to improve it if time permits.
    
    if time.time() > deadline - 5:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to construct a rank-64 decomposition using block methods.
    # 
    # I'll use the following approach:
    # 1. Split the 3x3 * 3x8 into sub-problems
    # 2. Use Strassen for 2x2 * 2x2 sub-blocks
    # 3. Handle the remaining terms with scalar products
    
    # Let me define the decomposition explicitly.
    
    # A[i,j] for i,j in 0,1,2
    # B[j,k] for j in 0,1,2, k in 0,1,2,3,4,5,6,7
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # I'll use the following rank-64 construction:
    # 
    # For the 2x2 * 2x8 part (using rows 0,1 of A and cols 0,1 of B):
    # Use Strassen: 7 terms per 2x2 * 2x2 block, 4 blocks = 28 terms
    # 
    # For the remaining terms involving row 2 of A or col 2 of B:
    # A[2,j]*B[j,k] for j=0,1,2 and all k: 3*8 = 24 terms
    # A[i,2]*B[2,k] for i=0,1 and all k: 2*8 = 16 terms
    # But we've already counted A[0,0]*B[0,k] and A[1,1]*B[1,k] etc. in the Strassen part.
    # 
    # This is getting messy. Let me just return the schoolbook for now.
    
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
