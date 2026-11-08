import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We use the tensor identity M_{4,4,6} ≅ M_{2,2,3} ⊗ M_{2,2,2}
    # Rank of M_{2,2,2} is 7 (Strassen)
    # We need a rank-6 decomposition of M_{2,2,3}
    # Actually, let's think about this more carefully.
    # M_{4,4,6} can be seen as:
    #   A is 4x4, B is 4x6, C is 4x6
    #   Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    #   Split B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x3? No...
    #   Actually B is 4x6. Split into 2x2 blocks of 2x3:
    #   B = [[B00, B01], [B10, B11]] where each B_ij is 2x3
    #   C is 4x6, split into 2x2 blocks of 2x3:
    #   C = [[C00, C01], [C10, C11]] where each C_ij is 2x3
    #   
    # Then C_ij = sum_k A_ik * B_kj where A_ik is 2x2 and B_kj is 2x3
    # So C_ij is a 2x3 matrix computed as (2x2) * (2x3)
    # 
    # The tensor M_{4,4,6} is the Kronecker product of M_{2,2,2} (for the block indices)
    # and M_{2,2,3} (for the inner dimensions).
    # 
    # Rank(M_{a,b,c} ⊗ M_{d,e,f}) = Rank(M_{a,b,c}) * Rank(M_{d,e,f})
    # 
    # So we need Rank(M_{2,2,3}). The known minimum rank for 2x2 by 2x3 is 6? 
    # Actually, let me reconsider. The standard approach:
    # 
    # Alternative: M_{4,4,6} can be decomposed using:
    # - Top level: Strassen on the 2x2 block structure (rank 7)
    # - Each term involves multiplying a 2x2 by a 2x3 matrix
    # - Rank of M_{2,2,3}: We can use schoolbook (4*3=12) or find better.
    #   Actually, for 2x2 * 2x3, the minimum rank is known to be 6? Let me check.
    #   No, I think it's not well-known to be 6. Let me use a different approach.
    #
    # Let me reconsider the Kronecker product approach:
    # M_{4,4,6} ≅ M_{2,2,2} ⊗ M_{2,2,3}
    # If we can find a rank-6 decomposition of M_{2,2,3}, then total rank = 7 * 6 = 42.
    # But is rank(M_{2,2,3}) = 6? 
    #
    # Actually, let's think of it differently. 
    # A 2x2 * 2x3 multiplication: 
    #   C = A * B where A is 2x2, B is 2x3, C is 2x3
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # We can group the columns of B. For each column j of B (j=0,1,2),
    # we compute A * B[:,j] which is a 2x1 vector.
    # Each such multiplication has rank 2 (schoolbook: 2 mults).
    # Total: 3 * 2 = 6 multiplications.
    # 
    # Wait, but that's just schoolbook with grouping. Is there a better way?
    # Actually, for 2x2 * 2x3, I believe the minimum rank is indeed 6.
    # Let me verify: the tensor has dimension 4*2*3 = 24. 
    # The rank is at least max(4, 2*3, 4*3) = 12? No, that's not right.
    # The rank is at least the maximum of the dimensions of the three modes divided by... 
    # Actually, rank >= max(n, m, p) is not a valid lower bound.
    # rank >= n*m/p, m*p/n, n*p/m? 
    # For 2x2x3: rank >= 2*2/3 ≈ 1.33, 2*3/2 = 3, 2*3/2 = 3. So rank >= 3.
    # But we know schoolbook gives 6. Can we do better?
    # 
    # Let me just use the rank-6 schoolbook-like decomposition for 2x2*2x3:
    # For each of the 3 columns of B, do 2 multiplications (one for each row of the result).
    # Actually, that's 3 columns * 2 rows = 6 terms, each being a product of one A entry and one B entry.
    # This is just schoolbook: 2*2*3 = 12 terms? No wait.
    #
    # Let me be more careful. Schoolbook for 2x2 * 2x3:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # There are 2*2*3 = 12 individual products, but they combine.
    # The rank of the tensor is the minimum number of rank-1 tensors that sum to it.
    # Schoolbook uses 2*2*3 = 12 rank-1 terms? No, schoolbook uses m = 2 multiplications for each output entry,
    # but the multiplications are shared. Actually, in the standard schoolbook, we compute:
    # t0 = A[0,0]*B[0,0], t1 = A[0,1]*B[1,0], etc. That's 12 terms.
    # But we can do better by sharing.
    #
    # Actually, for 2x2 * 2x3, the minimum rank is known to be 6. Here's a construction:
    # Think of B as three 2x1 vectors. For each vector, multiply by A (2x2) to get a 2x1 result.
    # Multiplying 2x2 by 2x1 has rank 2 (just two products: A[0,0]*B[0] + A[0,1]*B[1] for row 0,
    # and A[1,0]*B[0] + A[1,1]*B[1] for row 1). Wait, that's still 4 products per column if done naively.
    #
    # Hmm, let me reconsider. The rank of M_{2,2,2} is 7. The rank of M_{2,2,1} is 2.
    # M_{2,2,3} can be seen as three copies of M_{2,2,1} for each column, but the A matrix is shared.
    # So the tensor is not a simple Kronecker product in that direction.
    #
    # Let me just use a straightforward approach:
    # M_{4,4,6} = M_{2,2,2} (block) ⊗ M_{2,2,3} (inner)
    # For M_{2,2,3}, I'll use a rank-6 decomposition based on:
    #   For each of the 3 output columns, compute 2x2 * 2x1 using rank-2 (schoolbook for 2x2*2x1).
    #   But wait, 2x2 * 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #   This requires 4 products, not 2. The rank of M_{2,2,1} is 2? Let me check.
    #   M_{2,2,1}: A is 2x2, B is 2x1, C is 2x1.
    #   C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #   C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #   This can be done with 4 products (schoolbook). Can we do it with fewer?
    #   The rank of M_{2,2,1} is 2? I think so, because it's essentially two independent dot products
    #   that share the B vector. Actually no, the rank is 2 if we can express it as sum of 2 rank-1 tensors.
    #   Rank-1 tensor for M_{2,2,1}: U[i,k] * V[k,j] * W[i,j] where i in {0,1}, k in {0,1}, j in {0}.
    #   Since j is only 0, W[i,0] is just a scalar. So a rank-1 term is:
    #   sum_{i,k} U[i,k] V[k] W[i] * A[i,k] B[k] -> C[i]
    #   This is equivalent to: C[i] = W[i] * sum_k U[i,k] V[k] A[i,k] B[k]
    #   Hmm, this is getting complicated. Let me just use schoolbook for the inner part.
    #
    # Revised plan:
    # M_{4,4,6} using Kronecker:
    #   Top: Strassen 2x2 (rank 7)
    #   Inner: schoolbook 2x2 * 2x3 (rank 12)
    #   Total: 7 * 12 = 84. That's worse than 77.
    #
    # Let me try:
    #   Top: Strassen 2x2 (rank 7)  
    #   Inner: rank-6 for 2x2*2x3
    #   Total: 7 * 6 = 42
    #
    # I need to find a rank-6 decomposition of 2x2 * 2x3.
    # 
    # Here's a rank-6 decomposition for 2x2 * 2x3:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B = [[c00, c01, c02], [c10, c11, c12]]
    #
    # We can write:
    # t0 = a00 * b00, t1 = a01 * b10, t2 = a00 * b01, t3 = a01 * b11,
    # t4 = a00 * b02, t5 = a01 * b12, t6 = a10 * b00, t7 = a11 * b10,
    # t8 = a10 * b01, t9 = a11 * b11, t10 = a10 * b02, t11 = a11 * b12
    #
    # That's 12 terms. To get rank 6, we need to combine them.
    # 
    # Actually, I recall that the rank of 2x2 * 2x3 is 6. Let me construct it:
    # Group by the row of A:
    # Row 0 of C: [a00*b00 + a01*b10, a00*b01 + a01*b11, a00*b02 + a01*b12]
    #   = a00 * [b00, b01, b02] + a01 * [b10, b11, b12]
    # Row 1 of C: [a10*b00 + a11*b10, a10*b01 + a11*b11, a10*b02 + a11*b12]
    #   = a10 * [b00, b01, b02] + a11 * [b10, b11, b12]
    #
    # So we have 4 "products" of scalars with vectors:
    # a00 * B_row0, a01 * B_row1, a10 * B_row0, a11 * B_row1
    # But each of these is a vector of length 3, so in the tensor sense,
    # a00 * B_row0 contributes to C[0,0], C[0,1], C[0,2] simultaneously.
    # In the tensor decomposition, each rank-1 term contributes to all of C[0,:] at once.
    # So the 4 terms above give us:
    # Term 1: a00 * [b00, b01, b02] -> contributes to C[0,0], C[0,1], C[0,2]
    # Term 2: a01 * [b10, b11, b12] -> contributes to C[0,0], C[0,1], C[0,2]
    # Term 3: a10 * [b00, b01, b02] -> contributes to C[1,0], C[1,1], C[1,2]
    # Term 4: a11 * [b10, b11, b12] -> contributes to C[1,0], C[1,1], C[1,2]
    #
    # But wait, in a rank-1 tensor term for M_{2,2,3}, we have:
    # U[i,k] * V[k,j] * W[i,j]
    # where i in {0,1} (row of C), k in {0,1} (col of A / row of B), j in {0,1,2} (col of C)
    # A rank-1 term fixes a specific pattern. The 4 terms above are not rank-1 in the tensor sense
    # because they involve different j values.
    #
    # Let me think again. A rank-1 tensor for M_{2,2,3} is:
    # sum_{i,k,j} U[i,k] V[k,j] W[i,j] A[i,k] B[k,j]
    # This means for a fixed (i0, k0, j0), the term is U[i0,k0] V[k0,j0] W[i0,j0] A[i0,k0] B[k0,j0]
    # contributing to C[i0, j0].
    #
    # So each rank-1 term contributes to exactly one entry of C. Therefore, to cover all 6 entries of C,
    # we need at least 6 rank-1 terms. And schoolbook uses 12, but we can do it in 6 by being clever?
    # No, each rank-1 term contributes to one C[i,j] entry. So we need at least 6 terms (one per C entry).
    # And 6 is achievable if for each C[i,j], we can express it as a single product. But C[i,j] = sum_k A[i,k]B[k,j],
    # which is a sum of 2 products, not a single product. So we need at least 2 terms per C entry, giving 12 total.
    # Unless we can share terms across different C entries.
    #
    # Actually, I think the rank of M_{2,2,3} is 12 (schoolbook) and cannot be improved.
    # Let me just use the Kronecker product with schoolbook inner:
    # Rank = 7 * 12 = 84. That's worse than 77.
    #
    # Let me try a different decomposition. What if we use:
    # M_{4,4,6} = M_{2,2,2} ⊗ M_{2,2,3} but with a better M_{2,2,3}?
    # Or what if we don't use Strassen at the top level?
    #
    # Alternative: Use schoolbook at top level (16 terms) with some optimization at inner level.
    # Or use a different block structure.
    #
    # Let me try: Split A into 4x4, B into 4x6.
    # Use Strassen on the 4x4 part? No, Strassen is for 2x2.
    #
    # What about: M_{4,4,6} ≅ M_{4,2,3} ⊗ M_{1,2,2}? That doesn't make sense.
    #
    # Let me try a direct approach with a known decomposition.
    # The current program achieves rank 77. Let me see if I can do better with a cleaner Kronecker product.
    #
    # M_{4,4,6}: Let's use the fact that 4 = 2*2 and 6 = 2*3.
    # A (4x4) = kron(A1, A2) where A1 is 2x2 (block indices) and A2 is 2x2 (inner indices)
    # B (4x6) = kron(B1, B2) where B1 is 2x2 (block indices) and B2 is 2x3 (inner indices)
    # C (4x6) = kron(C1, C2) where C1 is 2x2 and C2 is 2x3
    #
    # C = A*B = kron(A1, A2) * kron(B1, B2) = kron(A1*B1, A2*B2)
    # So C1 = A1*B1 (2x2 * 2x2, rank 7 via Strassen)
    # C2 = A2*B2 (2x2 * 2x3, rank ?)
    #
    # The tensor M_{4,4,6} is the Kronecker product of M_{2,2,2} and M_{2,2,3}.
    # Rank(M_{4,4,6}) = Rank(M_{2,2,2}) * Rank(M_{2,2,3})
    #
    # If Rank(M_{2,2,3}) = 6, then total rank = 7 * 6 = 42.
    # If Rank(M_{2,2,3}) = 12, then total rank = 7 * 12 = 84.
    #
    # I'll assume Rank(M_{2,2,3}) = 6 and try to construct it. If it doesn't work, I'll fall back.
    #
    # Rank-6 decomposition of 2x2 * 2x3:
    # I found a reference that says the rank is 6. Here's the construction:
    # Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Define 6 products:
    # p1 = a * e
    # p2 = b * h  
    # p3 = a * f
    # p4 = b * i
    # p5 = a * g
    # p6 = b * j
    # p7 = c * e
    # p8 = d * h
    # p9 = c * f
    # p10 = d * i
    # p11 = c * g
    # p12 = d * j
    #
    # That's 12. To get 6, we need to combine.
    #
    # Actually, I think I'm overcomplicating this. Let me just use the current approach but try to optimize.
    # The current program has rank 77. Let me see if I can get a lower rank with a different strategy.
    #
    # What if I use:
    # - Top level: schoolbook 2x2 (4 terms)
    # - Inner level: Strassen 2x2*2x2 (7 terms) + schoolbook 2x2*2x1 (4 terms) = 11 terms
    # - Total: 4 * 11 = 44
    #
    # That's much better! Let me implement this.
    
    R = 4 * 11  # 44
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level schoolbook: 4 terms (s=0,1,2,3)
    # s=0: A00 * B00 -> C00
    # s=1: A01 * B11 -> C01  (wait, need to be careful with indices)
    # Actually for schoolbook 2x2 * 2x2:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # But B is 2x3, not 2x2. Let me reconsider.
    #
    # A is 4x4, split into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # B is 4x6, split into 2x2 blocks of 2x3: B = [[B00, B01], [B10, B11]]
    # C is 4x6, split into 2x2 blocks of 2x3: C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10  (2x2 * 2x3 + 2x2 * 2x3)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For schoolbook at top level, we have 4 terms:
    # s=0: A00 * B00 -> contributes to C00
    # s=1: A01 * B10 -> contributes to C00
    # s=2: A00 * B01 -> contributes to C01
    # s=3: A01 * B11 -> contributes to C01
    # s=4: A10 * B00 -> contributes to C10
    # s=5: A11 * B10 -> contributes to C10
    # s=6: A10 * B01 -> contributes to C11
    # s=7: A11 * B11 -> contributes to C11
    #
    # That's 8 terms, not 4. Schoolbook for 2x2 * 2x2 (where each block is a matrix) has 4 terms,
    # but here each "multiplication" A_ik * B_kj is a 2x2 * 2x3 multiplication.
    # So top-level schoolbook has 4 terms (for the 2x2 block matrix multiplication),
    # but each term involves a 2x2 * 2x3 inner multiplication.
    #
    # Wait, I'm confusing myself. Let me restart with a clear plan.
    #
    # Plan: 
    # 1. Top level: Strassen for 2x2 block structure (7 terms)
    # 2. Each term: inner multiplication of 2x2 by 2x3
    # 3. Inner: use rank-6 decomposition if possible, else schoolbook (12 terms)
    #
    # If inner rank is 6: total = 7 * 6 = 42
    # If inner rank is 12: total = 7 * 12 = 84
    #
    # Let me just implement with inner rank 6 and see if it works.
    # I'll use a simple rank-6 decomposition for 2x2 * 2x3 based on:
    #   For each of the 3 columns of B, do a 2x2 * 2x1 multiplication.
    #   2x2 * 2x1 has rank 2 (each output row is a dot product, but we can't share).
    #   Actually, 2x2 * 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #   This requires 4 products. So 3 columns * 4 = 12. Not 6.
    #
    # I give up on rank-6 for 2x2*2x3. Let me use a different approach.
    #
    # New plan: Use the current program's approach but verify it's correct.
    # The current program has rank 77. Let me just return a valid decomposition with rank <= 77.
    # Actually, let me try to get rank 44 with schoolbook top level (4 terms... wait, 8 terms) * inner.
    
    # Let me just implement a clean version with top-level schoolbook (8 terms) and inner Strassen+schoolbook.
    # Top: 8 terms (schoolbook for 2x2 block mult)
    # Inner: 7 (Strassen for 2x2*2x2) + 4 (schoolbook for 2x2*2x1) = 11
    # Total: 8 * 11 = 88. Worse than 77.
    
    # Let me stick with Strassen top level (7 terms) and optimize the inner.
    # Inner: 2x2 * 2x3. Use Strassen for first 2 columns (7 terms) + schoolbook for last column (4 terms) = 11.
    # Total: 7 * 11 = 77. Same as current.
    
    # To beat 77, I need a better inner decomposition.
    # What if I use a rank-8 decomposition for 2x2 * 2x3?
    # Then total = 7 * 8 = 56.
    
    # Let me try to find a rank-8 decomposition for 2x2 * 2x3.
    # I'll use Strassen for the 2x2*2x2 part (7 terms) and 1 term for the last column? No, that doesn't work.
    
    # Actually, let me just implement the current approach cleanly and return it.
    # The task says to rewrite from scratch, but the approach is the same.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 * 2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 * 2x2 (columns 0,1)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2 * 2x1 (column 2)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
