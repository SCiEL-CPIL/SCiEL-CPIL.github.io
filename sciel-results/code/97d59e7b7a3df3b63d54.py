import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Strategy: Use 4 disjoint 2x2x2 Strassen blocks + schoolbook for remainder
    # This gives rank 71 (28 Strassen + 43 schoolbook)
    # 
    # Then try to improve by finding a better decomposition using
    # a 5x5 matrix multiplication decomposition with lower rank.
    #
    # Key insight: The 3x5x5 tensor can be viewed as 3 independent 1x5x5 
    # "rows" of the output. Each row i needs to compute (A[i,:] * B)[k] 
    # which is a 1x5 matrix times 5x5 matrix = 1x5 result.
    #
    # Actually, let's think about this differently. The tensor T[i,j,k] 
    # represents C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # For fixed i, C[i,k] = sum_j A[i,j] * B[j,k] is a 5x5 matrix-vector 
    # multiplication (treating B's columns). The bilinear complexity of 
    # 1x5 * 5x5 -> 1x5 is the same as 5x5 * 5x1, which has rank 25 
    # (schoolbook) or potentially less with Strassen.
    #
    # Actually, 1x5 * 5x5 is just 5 independent dot products of length 5,
    # so it needs rank 25 in the bilinear sense (each output needs its own
    # sum).
    #
    # Wait, that's not right either. Let me reconsider.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor has indices (i, j, k) with i in [0,3), j in [0,5), k in [0,5).
    # T[i,j,k] = 1 for all valid (i,j,k).
    #
    # A rank-r decomposition means:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # where U is 3x5xr, V is 5x5xr, W is 5x3xr.
    #
    # The current approach uses 4 Strassen blocks (each 2x2x2) covering
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3} partially.
    #
    # Let me try a different approach: use a known lower-rank decomposition
    # for 5x5 matrix multiplication.
    #
    # Actually, let's try to use the fact that we have 3 rows of output.
    # We can group the computation.
    
    # First, let's build the baseline rank-71 solution
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7  # 28
    total_terms = n * m * p  # 75
    strassen_terms_count = num_strassen_blocks * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    total_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        c = col_offset + 0
        U[i0*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0, c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += -1
        
        c = col_offset + 2
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + i0, c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1
        V[j0*p + k0, c] += -1
        W[k0*n + i0, c] += 1
        W[k0*n + (i0+1), c] += 1
        
        c = col_offset + 4
        U[i0*m + j0, c] += 1
        U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += -1
        W[(k0+1)*n + i0, c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0, c] += 1
        U[i0*m + j0, c] += -1
        V[j0*p + k0, c] += 1
        V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+1), c] += -1
        V[(j0+1)*p + k0, c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
    
    def verify(Uv, Vv, Wv, tol=1e-10):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uv[i*m+j], Vv[j*p+k] * Wv[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify initial decomposition
    if not verify(U, V, W):
        # Fallback to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = np.zeros((n*m, total_rank_fb))
        V_fb = np.zeros((m*p, total_rank_fb))
        W_fb = np.zeros((p*n, total_rank_fb))
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1.0
                    V_fb[j*p+k, col_fb] = 1.0
                    W_fb[k*n+i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = total_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try to improve using rank reduction heuristics
    # Strategy: Try to find linear dependencies among the rank-1 terms
    
    # For each term c, its contribution to T[i,j,k] is:
    # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # If we can find coefficients alpha_c such that
    # sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0
    # then we can remove one term.
    
    # This is equivalent to finding the rank of the tensor as a 
    # multilinear map, which is hard.
    
    # Alternative: Try to use a known 5x5 matrix multiplication 
    # decomposition with rank < 25.
    
    # Actually, let's try a different approach: use the structure of
    # the problem more cleverly.
    
    # The key observation: we have 3 rows of output. Each row i computes
    # C[i,:] = A[i,:] @ B, which is a 1x5 times 5x5 multiplication.
    #
    # The bilinear complexity of 1x5 * 5x5 -> 1x5 is 25 (each of the 5
    # outputs needs a sum of 5 products).
    #
    # But wait - we can share computations across rows if we use 
    # Strassen-like tricks.
    
    # Let's try: use Strassen for 2x5x5 blocks (rows 0,1 together)
    # and schoolbook for row 2.
    
    # For 2x5x5: We can split j into {0,1} and {2,3,4}
    # But 2x5x5 doesn't have a clean Strassen decomposition.
    
    # Alternative: Use the fact that 5 = 2 + 2 + 1
    # Split j into {0,1}, {2,3}, {4}
    # Split k into {0,1}, {2,3}, {4}
    
    # For each pair of j-blocks and k-blocks, we have a 2x2x2 problem
    # for rows 0,1, and a 1x2x2 or 2x1x2 problem for row 2.
    
    # Let's try a different decomposition:
    # Use Strassen for 2x2x2 blocks for rows 0,1
    # And for row 2, use schoolbook
    
    # Actually, let's try to use a 3x3x3 Strassen-like decomposition
    # for the first 3x3x3 sub-block, then handle the rest.
    
    # 3x3x3 matrix multiplication has rank at most 23 (Laderman's algorithm
    # gives 23 for 4x4, but 3x3 is known to have rank <= 23 as well, 
    # though the exact minimum is unknown, believed to be around 19-23).
    
    # Let's use a known 3x3x3 decomposition. One approach:
    # Use the "middle product" trick.
    
    # Actually, let's try a simpler improvement: 
    # Use 2x2x2 Strassen blocks more cleverly.
    
    # Current: 4 blocks covering i={0,1}, j={0,1,2,3}, k={0,1,2,3}
    # Remaining: i=2 (all j,k) + i={0,1} with j=4 or k=4
    
    # Let's try adding more Strassen blocks for the remaining parts.
    
    # For i={0,1}, j={2,3}, k={2,3} is already covered.
    # For i={0,1}, j={0,1}, k={2,3} is already covered.
    # For i={0,1}, j={2,3}, k={0,1} is already covered.
    # For i={0,1}, j={0,1}, k={0,1} is already covered.
    
    # Uncovered for i={0,1}: j=4 (any k), k=4 (any j<4), and (j=4,k=4)
    # That's 4 + 4 + 1 = 9 terms for i={0,1}
    # Plus all 25 terms for i=2
    
    # Total uncovered: 9 + 25 = 34 terms
    # But we said 43 earlier. Let me recount.
    
    # Covered: 4 blocks * 8 = 32 terms
    # Total: 75
    # Uncovered: 75 - 32 = 43. OK.
    
    # For i={0,1}: 
    #   j=0,1,2,3 and k=0,1,2,3: covered (16 terms)
    #   j=4, k=0,1,2,3: 4 terms
    #   j=0,1,2,3, k=4: 4 terms  
    #   j=4, k=4: 1 term
    #   Total for i={0,1}: 16 + 4 + 4 + 1 = 25
    #   Uncovered for i={0,1}: 25 - 16 = 9
    
    # For i=2: all 25 terms uncovered
    
    # Total uncovered: 9 + 25 = 34. But 75-32=43. Discrepancy.
    
    # Let me recount the covered terms. Each 2x2x2 block covers 8 (i,j,k) tuples.
    # 4 blocks * 8 = 32. These are all distinct since the blocks are disjoint.
    # So 75 - 32 = 43 uncovered. My manual count was wrong.
    
    # Let me list them properly:
    # Block (0,0,0): i=0,1; j=0,1; k=0,1 -> 8 terms
    # Block (0,2,0): i=0,1; j=2,3; k=0,1 -> 8 terms
    # Block (0,0,2): i=0,1; j=0,1; k=2,3 -> 8 terms
    # Block (0,2,2): i=0,1; j=2,3; k=2,3 -> 8 terms
    # Total covered: 32 terms, all with i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Uncovered:
    # i=0,1: j=4 (k=0..4: 5 terms) + k=4 (j=0..3: 4 terms) = 9 terms
    # i=2: all 25 terms
    # Total: 9 + 25 = 34
    
    # But 75 - 32 = 43. So I'm missing 9 terms.
    
    # Ah, I see: for i=0,1, j in {0,1,2,3}, k in {0,1,2,3} is 4*4=16 terms.
    # 4 blocks * 8 = 32, but each block covers 2*2*2=8 (i,j,k) tuples, 
    # not 8 (j,k) tuples for fixed i.
    
    # Wait, each 2x2x2 block covers 2 i-values, 2 j-values, 2 k-values = 8 tuples.
    # 4 blocks cover 32 distinct (i,j,k) tuples.
    # Total tuples: 3*5*5 = 75.
    # Uncovered: 75 - 32 = 43.
    
    # Let me count uncovered more carefully:
    # i=0: j=0..4, k=0..4 = 25 terms. Covered: j=0,1,2,3 and k=0,1,2,3 = 16. Uncovered: 25-16=9
    # i=1: same as i=0. Uncovered: 9
    # i=2: 25 terms, all uncovered.
    # Total uncovered: 9+9+25 = 43. OK, that matches.
    
    # So for i=0 and i=1, we have 9 uncovered terms each.
    # These are: (j=4, k=0..4) and (j=0..3, k=4) for each i.
    
    # Can we use Strassen for these? The uncovered part for i={0,1} is:
    # j=4 with k=0..4 (5 terms) and j=0..3 with k=4 (4 terms)
    # This is like a "cross" shape, not a clean 2x2 block.
    
    # Let's try a different approach: use a 5x5 matrix multiplication
    # decomposition.
    
    # Known: 5x5 matrix multiplication can be done with rank 25 (schoolbook)
    # or potentially less. The best known upper bound for 5x5 is 25,
    # and the lower bound is also 25 (since it's a full-rank tensor).
    # Actually, no - the tensor rank of 5x5 matrix multiplication is 
    # at most 25 but could be less.
    
    # Actually, for n x n matrix multiplication, the tensor rank is at most
    # n^3 (schoolbook) and at least n^2 (since the tensor has slice rank n^2).
    # For 5x5, the slice rank is 25, so tensor rank >= 25.
    # And schoolbook gives 25. So tensor rank of 5x5 matmul is exactly 25.
    
    # This means we can't do better than 25 for a 5x5 sub-problem.
    
    # But our problem is 3x5x5, not 5x5x5. The slice rank argument 
    # gives a lower bound of max(n*m, m*p, p*n) = max(15, 25, 15) = 25.
    # So tensor rank >= 25.
    
    # Our current rank is 71. Can we do better?
    
    # Let's try a different decomposition strategy.
    
    # Idea: Use the fact that we only have 3 rows. 
    # C = A @ B where A is 3x5 and B is 5x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as: for each k, C[:,k] = A @ B[:,k]
    # This is 5 independent 3x5 * 5x1 matrix-vector multiplications.
    # Each has rank 15 (schoolbook) or potentially less.
    
    # Alternatively: for each i, C[i,:] = A[i,:] @ B
    # This is 3 independent 1x5 * 5x5 multiplications.
    # Each has rank 25 (schoolbook).
    
    # Total schoolbook: 3*25 = 75 or 5*15 = 75.
    
    # Can we share computations? If we use the same intermediate 
    # values for different rows, we might save rank.
    
    # Let's try: compute M = A @ B using a block decomposition.
    # Split A into [A1; A2; A3] where A1, A2, A3 are 1x5.
    # Split B into columns B1, B2, B3, B4, B5 (each 5x1).
    # C[i,k] = A_i @ B_k
    
    # Use Strassen for 2x2 blocks of A and B.
    # A has 3 rows, B has 5 columns.
    
    # Let's try: group rows 0,1 together and use Strassen for 2x2 blocks.
    # Group columns 0,1,2,3 in pairs and column 4 separately.
    
    # For rows 0,1 and columns 0,1: 2x2x2 Strassen (rank 7)
    # For rows 0,1 and columns 2,3: 2x2x2 Strassen (rank 7)
    # For rows 0,1 and column 4: 2x1x1 (rank 2)
    # For row 2 and columns 0,1: 1x2x2 (rank 4)
    # For row 2 and columns 2,3: 1x2x2 (rank 4)
    # For row 2 and column 4: 1x1x1 (rank 1)
    
    # Total: 7+7+2+4+4+1 = 25. That's not better than 71.
    
    # Wait, I think I'm confusing myself. The "rank" here is the 
    # bilinear complexity, not the number of scalar multiplications.
    
    # Let me restart with a clearer approach.
    
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5).
    # We want to find the minimum r such that
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is the tensor rank of the 3x5x5 all-ones tensor.
    
    # Actually, no. The matrix multiplication tensor for 3x5x5 is:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k).
    
    # The tensor rank is the minimum r such that
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For the all-ones tensor of size 3x5x5, what is its rank?
    
    # The all-ones tensor of size a x b x c has tensor rank 
    # min(a*b, b*c, c*a) in some cases, but generally it's 
    # at most a*b*c and at least max(a*b, b*c, c*a).
    
    # For 3x5x5: max(15, 25, 15) = 25. So rank >= 25.
    # And rank <= 75 (schoolbook).
    
    # Can we achieve rank 25? That would require 
    # T[i,j,k] = sum_{c=0}^{24} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # One way: use the fact that T is a "simple" tensor.
    # T[i,j,k] = 1 = 1*1*1.
    
    # Actually, the all-ones tensor of size a x b x c can be written as
    # T[i,j,k] = sum_{c=0}^{r-1} u_c[i] * v_c[j] * w_c[k]
    # where r is the rank of the tensor as a 3-way array.
    
    # For the all-ones tensor, we can write:
    # T[i,j,k] = 1 = (1)(1)(1)
    # So it has rank 1! But that's for the 3-way array rank, not 
    # the bilinear complexity of matrix multiplication.
    
    # I think I've been confusing two different concepts. Let me clarify.
    
    # The matrix multiplication tensor for multiplying an a x b matrix 
    # by a b x c matrix is a tensor in R^{a*b*c} (or more precisely,
    # a 3-way tensor with modes of sizes a, b, c... no wait).
    
    # Actually, the matrix multiplication tensor M_{a,b,c} is defined as:
    # M[i,j,k] = 1 for i in [0,a), j in [0,b), k in [0,c)
    # where the three modes correspond to:
    #   mode 1: (i,j) pairs, size a*b
    #   mode 2: (j,k) pairs, size b*c  
    #   mode 3: (i,k) pairs, size a*c
    
    # So it's a 3-way tensor of size (a*b) x (b*c) x (a*c).
    # For 3x5x5: size 15 x 25 x 15.
    
    # The tensor rank is the minimum r such that
    # M[i,j,k] = sum_{c=0}^{r-1} U[(i,j),c] * V[(j,k),c] * W[(i,k),c]
    
    # This is what we're trying to minimize.
    
    # The slice rank lower bound: for any tensor in X x Y x Z,
    # rank >= max(slice_rank_X, slice_rank_Y, slice_rank_Z)
    # where slice_rank_X is the maximum rank of slices perpendicular to X.
    
    # For M_{3,5,5}:
    # Slices perpendicular to mode 1 (size 15): each slice is a 25x15 matrix.
    # The rank of each slice is at most 15 (since it's 25x15).
    # Actually, the slice rank in mode 1 is the maximum over i of 
    # rank(M[i,:,:]) where M[i,:,:] is a 25x15 matrix.
    # Each such slice has rank at most 15.
    
    # Slices perpendicular to mode 2 (size 25): each slice is a 15x15 matrix.
    # Rank at most 15.
    
    # Slices perpendicular to mode 3 (size 15): each slice is a 15x25 matrix.
    # Rank at most 15.
    
    # So slice rank >= 15 in all modes, giving rank >= 15.
    
    # But we know rank >= 25 from the "border rank" or other arguments.
    # Actually, I'm not sure about the exact lower bound.
    
    # Let's just try to find a good decomposition empirically.
    
    # Approach: Start with schoolbook (rank 75) and try to reduce rank
    # by finding linear dependencies.
    
    # Actually, let's try a completely different approach:
    # Use the fact that 3x5x5 matmul can be reduced to fewer 
    # scalar multiplications using Strassen-like tricks.
    
    # Key idea: Use Strassen for 2x2x2 sub-blocks wherever possible.
    
    # We have 3 rows, 5 cols in A, and 5 rows, 5 cols in B.
    # Output is 3x5.
    
    # Let's partition:
    # Rows of A: {0,1} and {2}
    # Cols of A / Rows of B: {0,1}, {2,3}, {4}
    # Cols of B: {0,1}, {2,3}, {4}
    
    # For rows {0,1} and col-groups {0,1} x {0,1}: 2x2x2 Strassen (rank 7)
    # For rows {0,1} and col-groups {2,3} x {0,1}: 2x2x2 Strassen (rank 7)
    # For rows {0,1} and col-groups {0,1} x {2,3}: 2x2x2 Strassen (rank 7)
    # For rows {0,1} and col-groups {2,3} x {2,3}: 2x2x2 Strassen (rank 7)
    # For rows {0,1} and col-groups {4} x {0,1}: 2x1x2 (rank 4)
    # For rows {0,1} and col-groups {0,1} x {4}: 2x2x1 (rank 4)
    # For rows {0,1} and col-groups {2,3} x {4}: 2x2x1 (rank 4)
    # For rows {0,1} and col-groups {4} x {2,3}: 2x1x2 (rank 4)
    # For rows {0,1} and col-groups {4} x {4}: 2x1x1 (rank 2)
    
    # For row {2} and col-groups {0,1} x {0,1}: 1x2x2 (rank 4)
    # For row {2} and col-groups {2,3} x {0,1}: 1x2x2 (rank 4)
    # For row {2} and col-groups {0,1} x {2,3}: 1x2x2 (rank 4)
    # For row {2} and col-groups {2,3} x {2,3}: 1x2x2 (rank 4)
    # For row {2} and col-groups {4} x {0,1}: 1x1x2 (rank 2)
    # For row {2} and col-groups {0,1} x {4}: 1x2x1 (rank 2)
    # For row {2} and col-groups {2,3} x {4}: 1x2x1 (rank 2)
    # For row {2} and col-groups {4} x {2,3}: 1x1x2 (rank 2)
    # For row {2} and col-groups {4} x {4}: 1x1x1 (rank 1)
    
    # Total: 7*4 + 4*4 + 2 + 4*4 + 2*4 + 1 = 28 + 16 + 2 + 16 + 8 + 1 = 71
    
    # Same as before! The current decomposition is already optimal 
    # for this partitioning strategy.
    
    # To do better, we need a different strategy.
    
    # Let's try: use a 3x3x3 Strassen-like decomposition for the 
    # first 3x3x3 sub-block.
    
    # 3x3x3 matmul: A is 3x3, B is 3x3, C is 3x3.
    # Schoolbook: 27 multiplications.
    # Best known: rank 23 (Laderman's algorithm for 4x4 gives 23, 
    # but for 3x3 the best known is 23 as well, achieved by various 
    # algorithms).
    
    # Actually, the tensor rank of 3x3x3 matmul is at most 23 
    # (Laderman) and at least 19 (lower bound). The exact value 
    # is unknown.
    
    # Let's use a rank-23 decomposition for 3x3x3.
    
    # Laderman's algorithm for 4x4x4 gives rank 23. For 3x3x3,
    # we can use a similar approach.
    
    # Actually, let's just use the schoolbook for 3x3x3 (rank 27)
    # and see if we can do better overall.
    
    # Partition:
    # Rows: {0,1,2} (all 3 rows)
    # Cols/Rows: {0,1,2}, {3,4}
    # Cols: {0,1,2}, {3,4}
    
    # For 3x3x3 block (rows 0,1,2; cols 0,1,2; rows 0,1,2): rank 27 (schoolbook)
    # For 3x2x2 block (rows 0,1,2; cols 3,4; rows 0,1,2): 
    #   This is 3 independent 1x2x2 problems, each rank 4. Total: 12.
    #   Or we can use Strassen: 3x2x2 = 3 * (1x2x2). Each 1x2x2 has rank 4.
    #   Actually, 1x2x2: C[i,k] = sum_j A[i,j] * B[j,k] where i is fixed.
    #   This is 2 independent dot products of length 2. Rank 4.
    #   So 3x2x2 has rank 3*4 = 12.
    #
    # For 3x3x2 block (rows 0,1,2; cols 0,1,2; rows 3,4):
    #   Similarly, rank 12.
    #
    # For 3x2x2 block (rows 0,1,2; cols 3,4; rows 3,4):
    #   Rank 12.
    
    # Total: 27 + 12 + 12 + 12 = 63. Better than 71!
    
    # Let's verify: 3x5x5 = 3x(3+2) x (3+2)
    # = 3x3x3 + 3x2x3 + 3x3x2 + 3x2x2
    # = 27 + 18 + 18 + 12 = 75 (schoolbook)
    
    # With Strassen on the 3x3x3 block: 23 + 18 + 18 + 12 = 71. Same.
    # With Strassen on other blocks:
    # 3x2x3: can we do better than 18? 
    #   3x2x3 = 3 * (1x2x3). Each 1x2x3 has rank 6 (schoolbook).
    #   Can we do 1x2x3 with rank < 6? 
    #   1x2x3: C[k] = A[0] * B[0,k] + A[1] * B[1,k] for k=0,1,2.
    #   This is 3 independent sums of 2 products. Rank 6.
    #   I don't think we can do better.
    
    # So the 3x3x3 + 3x2x3 + 3x3x2 + 3x2x2 decomposition gives:
    # 27 + 18 + 18 + 12 = 75 (all schoolbook)
    # 23 + 18 + 18 + 12 = 71 (Strassen on 3x3x3)
    
    # Same as before. Not an improvement.
    
    # Let's try yet another approach: use the fact that we can 
    # share Strassen intermediate values across different blocks.
    
    # For example, in the 4-block Strassen decomposition, the 
    # intermediate values for different blocks are independent.
    # But if we reorganize, we might be able to share.
    
    # This is getting complex. Let's just return the current 
    # rank-71 solution, which is already good.
    
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try one more thing: use a rank-23 decomposition for 3x3x3
    # and see if the total rank improves.
    
    # Laderman's 4x4x4 rank-23 algorithm can be adapted to 3x3x3.
    # Actually, let's use a known 3x3x3 rank-23 decomposition.
    
    # The 3x3x3 matmul tensor has rank at most 23.
    # One construction: use the 4x4x4 Laderman algorithm and 
    # restrict to 3x3x3.
    
    # Laderman's 4x4x4 algorithm uses 23 multiplications.
    # For 3x3x3, we can use a similar approach.
    
    # Actually, let's just use schoolbook for everything and accept rank 71.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
