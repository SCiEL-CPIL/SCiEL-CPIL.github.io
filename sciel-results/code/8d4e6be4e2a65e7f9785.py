"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: use a block decomposition where we split the inner dimension (j) into
    # two parts: first 2 and last 3. Then:
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
    # 
    # The first product is 3x2 * 2x5, the second is 3x3 * 3x5.
    # For a 3x2 * 2x5 product, the tensor rank is at most 3*2*5 = 30, but we can do better.
    # Actually, for 3x2 * 2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a sum of two "outer product" type terms. The tensor rank of this specific
    # sub-problem is at most 15 (one per (i,k) pair if we use schoolbook on the inner dim).
    # But actually, for fixed i, C[i,k] depends linearly on A[i,0], A[i,1] and B[0,k], B[1,k].
    # The tensor for the 3x2*2x5 part has rank at most 2*3*5 = 30 with schoolbook, but
    # we can use the fact that it's essentially two separate 3x1*1x5 products added together,
    # giving rank 3*5*2 = 30.
    #
    # Wait, let me think more carefully. For the 3x2 * 2x5 multiplication:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # This is exactly the matrix multiplication tensor M_{3,2,5}.
    # The rank of M_{3,2,5} is at most 3*2*5 = 30 (schoolbook).
    # But can we do better? For M_{1,m,p}, the rank is m*p (optimal). For M_{n,1,p}, rank is n*p.
    # For M_{2,2,2}, rank is 7 (Strassen). 
    # For M_{3,2,5}, a good upper bound would be to note that we can treat it as
    # computing 3 vectors of length 2 times 5 rows of a 2x5 matrix.
    # Actually, R(3,2,5) <= R(3,2,2) * something? No.
    #
    # Let me just use schoolbook for each block and see if the total is less than 75.
    # Block 1: 3x2 * 2x5 -> rank 3*2*5 = 30
    # Block 2: 3x3 * 3x5 -> rank 3*3*5 = 45
    # Total: 75. Same as full schoolbook. No improvement.
    #
    # Alternative: use Strassen-like tricks on the 3x3 * 3x5 block.
    # For 3x3 * 3x3, rank is at most 19 (or 16?). Actually R(3,3,3) <= 19.
    # But we have 3x3 * 3x5. We can write this as 3x3 * 3x3 + 3x3 * 3x2 (splitting the 5 columns into 3+2).
    # R(3,3,3) <= 19, R(3,3,2) <= 18 (schoolbook). Total for the 3x3*3x5 part: 37.
    # And R(3,2,5) <= 30 (schoolbook). Total: 67. Better than 75!
    #
    # But can we do better on R(3,2,5)? 
    # R(3,2,5): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For each i, this is a bilinear form. The tensor is a sum over i of terms.
    # Actually, R(3,2,5) = R(3,2,5). Let me think of it as: for each (i,k), we need 
    # A[i,0]*B[0,k] + A[i,1]*B[1,k]. This is 15 independent outputs, each requiring 
    # 2 products. The tensor rank is at most 30. Can we reduce?
    # Using Winograd-type: for 1x2 * 2x1, rank is 3 (Strassen for 1x1? No, that's trivial).
    # For 1x2 * 2x1: C = A0*B0 + A1*B1. Rank is 3 (using (A0+A1)(B0+B1) - A0B0 - A1B1... 
    # wait, that's 3 terms but one is a sum). Actually rank of 1x2*2x1 is 3? 
    # No, it's just 2 products, so rank 2. You can't do better than 2 for 1x2*2x1.
    # So R(3,2,5) >= 15*2 = 30? No, because the A and B entries are shared across k.
    # For fixed i, A[i,0] and A[i,1] are shared across all k. So:
    # C[i,0], C[i,1], ..., C[i,4] all use A[i,0] and A[i,1].
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a linear map from (A[i,0], A[i,1]) to a 5-dim vector, with coefficients 
    # depending on B[0,k], B[1,k].
    # The tensor for fixed i is essentially a 2x5 matrix in the "B" part.
    # The rank of the full tensor M_{3,2,5} is the sum over i of the rank of the 
    # i-th slice? Not quite, because A[i,j] is specific to i.
    # Actually, R(3,2,5) = 3 * R(1,2,5) = 3 * 10 = 30? Since the i-th slice 
    # involves A[i,0], A[i,1] which don't appear in other slices.
    # R(1,2,5) = 2*5 = 10 (schoolbook, and optimal since each output C[k] = A0*B0k + A1*B1k 
    # requires at least 2 products... actually is R(1,2,5) = 10 or can it be less?).
    # For 1x2 * 2x5: C[k] = A0*B0k + A1*B1k for k=0..4.
    # The tensor is a 2x5x1 tensor (or 5x2x1). Each C[k] is a separate output.
    # The A part has 2 entries, B part has 10 entries, C part has 5 entries.
    # R(1,2,5) is the tensor rank of a 2x10x5 tensor where T[a,b,k] = 1 if a and b 
    # are paired (a=0,b=0..4 for B[0,k]; a=1,b=5..9 for B[1,k]) and C index is k.
    # Actually T[a, (j,k), c] = 1 if a=j and c=k. So it's the identity in a sense.
    # R(1,2,5) = 10 (each of the 10 products A[j]*B[j,k] for j=0,1 and k=0..4 is needed).
    # So R(3,2,5) = 30.
    #
    # Now for R(3,3,5): split 5 into 3+2.
    # R(3,3,5) <= R(3,3,3) + R(3,3,2).
    # R(3,3,3) <= 19 (known upper bound).
    # R(3,3,2): C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k] for i=0..2, k=0,1.
    # This is 3x3 * 3x2. R(3,3,2) <= 18 (schoolbook).
    # Can we do better? R(3,3,2) = R(2,3,3) by symmetry. 
    # Actually, R(3,3,2): for each k (2 values), we compute a 3x3 matrix product summed...
    # No, C[i,k] = sum_j A[i,j]B[j,k]. For k=0: C[i,0] = A[i,0]B[0,0]+A[i,1]B[1,0]+A[i,2]B[2,0].
    # This is like computing 2 separate 3x3*3x1 products. R(3,3,1) = 9. So R(3,3,2) <= 18.
    # Is R(3,3,1) = 9? For 3x3*3x1: C[i] = sum_j A[i,j]B[j]. Yes, 9 products, rank 9.
    # So R(3,3,2) = 18.
    # Total: R(3,3,5) <= 19 + 18 = 37.
    # Grand total: R(3,2,5) + R(3,3,5) <= 30 + 37 = 67.
    #
    # Let me now construct this decomposition explicitly.
    #
    # Part 1: 3x2 * 2x5 (columns 0,1 of A; rows 0,1 of B)
    # Schoolbook: 30 terms
    # For i in 0..2, j in 0..1, k in 0..4:
    #   U[i*2+j] * V[j*5+k] * W[k*3+i]
    #
    # Part 2: 3x3 * 3x5 (columns 2,3,4 of A; rows 2,3,4 of B)
    # Split into 3x3*3x3 + 3x3*3x2
    # 
    # Sub-part 2a: 3x3 * 3x3 (cols 2,3,4 of A; rows 2,3,4 of B; cols 0,1,2 of C)
    # Use rank-19 decomposition for 3x3*3x3.
    # 
    # Sub-part 2b: 3x3 * 3x2 (cols 2,3,4 of A; rows 2,3,4 of B; cols 3,4 of C)
    # Schoolbook: 18 terms
    #
    # Let me implement the rank-19 decomposition for 3x3*3x3.
    # A known rank-19 decomposition (from Bini, Capovani, Lotti, Romani, or similar):
    #
    # The 3x3 matrix multiplication can be done with 19 multiplications.
    # Here's one such decomposition (using 0-indexed, C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]):
    #
    # I'll use a known construction. Let me define the 19 terms for 3x3*3x3.
    # One standard rank-19 decomposition:
    #
    # Terms 0-8: schoolbook for the diagonal and some off-diagonal
    # Terms 9-18: Strassen-like combinations
    #
    # Actually, let me use a simpler approach: since I need to be exact, let me
    # just use schoolbook for the 3x3*3x3 part (rank 27) and see if the total is 
    # still better than 75.
    # Total would be: 30 (for 3x2*2x5) + 27 (for 3x3*3x3) + 18 (for 3x3*3x2) = 75.
    # Same as before. No improvement.
    #
    # I need to use the rank-19 for 3x3*3x3 to get 30 + 19 + 18 = 67.
    #
    # Let me find a concrete rank-19 decomposition for 3x3*3x3.
    #
    # Here's a known rank-19 decomposition (from the literature):
    # We compute 19 products M_r = (linear combo of A)*(linear combo of B) for r=1..19,
    # then C[i,j] = linear combo of M_r.
    #
    # Let me use the following (one of the known constructions):
    #
    # A is 3x3, B is 3x3, C = A*B.
    # A[i,k] for i,k in 0,1,2. B[k,j] for k,j in 0,1,2.
    #
    # I'll use a decomposition based on the idea of "tensor rank 19 for 3x3x3".
    # One such decomposition is given by:
    #
    # Let's define the 19 rank-1 tensors. Each is specified by:
    # - u_r[i,k] for A indices
    # - v_r[k,j] for B indices  
    # - w_r[i,j] for C indices
    # such that sum_r u_r[i,k] * v_r[k,j] * w_r[i,j] = delta (the multiplication tensor).
    #
    # Actually, the convention is: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T[i,k,j] = 1 (it's the identity tensor in the middle index).
    # We want T = sum_r U_r * V_r * W_r where (U_r)[i,k] (U_r)[k,j] (W_r)[i,j].
    #
    # Here is a rank-19 decomposition (I'll construct it):
    #
    # Actually, for simplicity and correctness, let me just use the schoolbook for 
    # everything and accept rank 75. The task asks for rank-minimal, but if I can't 
    # guarantee correctness of a non-trivial decomposition, rank 75 is safe.
    #
    # However, let me try one more thing: use the known result that R(3,3,3) <= 19
    # and implement it carefully.
    #
    # From Bini et al. (1978), a rank-19 decomposition for 3x3x3:
    #
    # I'll use the following explicit construction. The 19 terms are:
    #
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C = A*B
    #
    # 19 products:
    # M1 = a00*b00
    # M2 = a00*b11
    # M3 = a00*b22
    # M4 = a10*b00
    # M5 = a10*b11
    # M6 = a10*b22
    # M7 = a20*b00
    # M8 = a20*b11
    # M9 = a20*b22
    # M10 = (a01+a11+a21)*(b00+b01+b02)  -- wait, this doesn't seem right.
    #
    # Let me look up a specific rank-19 construction. 
    # 
    # Actually, I recall that one common rank-19 construction uses:
    # - 9 "diagonal" terms (schoolbook for the diagonal positions)
    # - 10 "off-diagonal" terms using Strassen-like identities
    #
    # This is getting complicated. Let me just use rank 75 schoolbook to ensure correctness.
    #
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
