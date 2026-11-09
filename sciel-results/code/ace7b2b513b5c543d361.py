import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Kronecker product approach:
    # M_{4,4,6} ≅ M_{2,2,2} ⊗ M_{2,2,3}
    # Rank = Rank(M_{2,2,2}) * Rank(M_{2,2,3})
    # 
    # For M_{2,2,3}: 2x2 * 2x3 matrix multiplication
    # We can decompose this as: for each of the 3 columns, do 2x2 * 2x1
    # But actually, let's think of it differently.
    #
    # A (2x2) * B (2x3) = C (2x3)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    #
    # The tensor M_{2,2,3} has dimensions: A is 2x2 (4 entries), B is 2x3 (6 entries), C is 2x3 (6 entries)
    # 
    # Rank-6 decomposition for 2x2 * 2x3:
    # We can view this as: C = A * B where we treat B as 3 column vectors.
    # For each column j of B, we compute A * B[:,j] (a 2x1 vector).
    # A (2x2) * B_col (2x1) = C_col (2x1)
    # C_col[0] = A[0,0]*B_col[0] + A[0,1]*B_col[1]
    # C_col[1] = A[1,0]*B_col[0] + A[1,1]*B_col[1]
    #
    # This is 2x2 * 2x1 multiplication. The rank of M_{2,2,1} is 2? 
    # Actually, for 2x2 * 2x1:
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This can be done with 4 products (schoolbook). But can we do better?
    # The rank of M_{2,2,1} is 2 (since it's essentially two dot products of length 2,
    # but they share the B vector). Actually no - each output entry needs 2 products,
    # so we need at least 4 products total. Rank = 4.
    #
    # Wait, that gives 3 * 4 = 12 for M_{2,2,3}. That's schoolbook.
    #
    # Let me reconsider. Actually, I think the minimum rank for 2x2 * 2x3 is 6.
    # Here's a construction:
    # 
    # Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Define 6 intermediate products:
    # p1 = a*e, p2 = b*h, p3 = c*e, p4 = d*h
    # p5 = a*(f+i), p6 = b*(f+i)  -- wait, this doesn't work cleanly.
    #
    # Actually, let me try a different approach. 
    # 
    # For 2x2 * 2x3, note that we can write:
    # C = A * B = [[a,b],[c,d]] * [[e,f,g],[h,i,j]]
    #
    # Row 0 of C: [ae+bh, af+bi, ag+bj] = a*[e,f,g] + b*[h,i,j]
    # Row 1 of C: [ce+dh, cf+di, cg+dj] = c*[e,f,g] + d*[h,i,j]
    #
    # So C = [[a,b],[c,d]] * [[e,f,g],[h,i,j]]
    # 
    # In tensor form, each rank-1 term is of the form:
    # u_i * v_j * w_k where i indexes A, j indexes B, k indexes C.
    #
    # A rank-1 tensor for M_{2,2,3} contributes:
    # sum_{i,j,k} U[i,j] * V[j,k] * W[i,k] * A[i,j] * B[j,k]
    #
    # For a single rank-1 term, we pick specific indices. 
    # 
    # Let me just use schoolbook for M_{2,2,3} which gives rank 12,
    # and Strassen for M_{2,2,2} which gives rank 7.
    # Total: 7 * 12 = 84. That's worse than 77.
    #
    # Alternative: Use schoolbook for the top level (2x2 * 2x2 block structure)
    # which gives 4 terms, and for each term use a rank-6 decomposition of 2x2 * 2x3.
    # But I'm not sure rank-6 is achievable for 2x2 * 2x3.
    #
    # Let me try yet another approach. What if I use:
    # M_{4,4,6} with a direct construction?
    #
    # Actually, let me reconsider the Kronecker product more carefully.
    # 
    # A is 4x4. Think of A as a 2x2 block matrix where each block is 2x2.
    # B is 4x6. Think of B as a 2x2 block matrix where each block is 2x3.
    # C is 4x6. Think of C as a 2x2 block matrix where each block is 2x3.
    #
    # C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    # where A_block[i,k] is 2x2, B_block[k,j] is 2x3, C_block[i,j] is 2x3.
    #
    # So C_block[i,j] = (2x2) * (2x3) = 2x3.
    #
    # The full tensor M_{4,4,6} is the Kronecker product of:
    # - The block multiplication tensor M_{2,2,2} (for the 2x2 block structure)
    # - The inner multiplication tensor M_{2,2,3} (for the 2x2 * 2x3 inner product)
    #
    # Rank(M_{4,4,6}) <= Rank(M_{2,2,2}) * Rank(M_{2,2,3})
    #
    # If I can find a rank-6 decomposition for M_{2,2,3}, then total rank = 7 * 6 = 42.
    #
    # Let me try to construct a rank-6 decomposition for 2x2 * 2x3.
    #
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # I'll use the following 6 products:
    # t1 = a*e
    # t2 = b*h
    # t3 = c*e  
    # t4 = d*h
    # t5 = (a+b)*(f+i)  -- this gives af+ai+bf+bi, not quite right
    # 
    # That doesn't work. Let me try:
    # t1 = a*e, t2 = b*h, t3 = c*e, t4 = d*h
    # For the middle column: af+bi and cf+di
    # t5 = a*f, t6 = b*i, t7 = c*f, t8 = d*i
    # For the last column: ag+bj and cg+dj
    # t9 = a*g, t10 = b*j, t11 = c*g, t12 = d*j
    #
    # That's 12 terms. To get 6, I need to combine them.
    #
    # Actually, I think the rank of 2x2 * 2x3 is indeed 12 (schoolbook) and cannot be improved
    # in general. Let me just use a different strategy.
    #
    # Strategy: Use Strassen for the top-level 2x2 block multiplication (7 terms),
    # and for the inner 2x2 * 2x3, use a rank-8 decomposition if possible.
    #
    # Actually, let me just implement the current approach (rank 77) cleanly.
    # The current program uses:
    # - Strassen for top-level 2x2 (7 terms)
    # - For each term: Strassen for 2x2*2x2 inner (7 terms) + schoolbook for 2x2*2x1 (4 terms) = 11
    # - Total: 7 * 11 = 77
    #
    # Let me implement this cleanly.
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 * 2x2
    # alpha[s][a] where s is the Strassen term, a indexes the 4 entries of A (row-major)
    # beta[s][b] where b indexes the 4 entries of B (row-major)
    # recon[s] gives the contribution to each of the 4 output entries
    
    alpha = [
        [1, 0, 0, 1],  # s=0: a00 + a11
        [0, 0, 1, 1],  # s=1: a10 + a11
        [1, 0, 0, 0],  # s=2: a00
        [0, 0, 0, 1],  # s=3: a11
        [1, 1, 0, 0],  # s=4: a00 + a01
        [-1, 0, 1, 0], # s=5: a10 - a00
        [0, 1, 0, -1]  # s=6: a11 - a01
    ]
    beta = [
        [1, 0, 0, 1],  # s=0: b00 + b11
        [1, 0, 0, 0],  # s=1: b00
        [0, 1, 0, -1], # s=2: b01 - b11
        [-1, 0, 1, 0], # s=3: b10 - b00
        [0, 0, 0, 1],  # s=4: b11
        [1, 1, 0, 0],  # s=5: b00 + b01
        [0, 0, 1, 1]   # s=6: b10 + b11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],  # s=0: C00 += t0, C11 += t0
        [(1, 0, 1), (1, 1, -1)], # s=1: C10 += t1, C11 -= t1
        [(0, 1, 1), (1, 1, 1)],  # s=2: C01 += t2, C11 += t2
        [(0, 0, 1), (1, 0, 1)],  # s=3: C00 += t3, C10 += t3
        [(0, 0, -1), (0, 1, 1)], # s=4: C00 -= t4, C01 += t4
        [(1, 1, 1)],             # s=5: C11 += t5
        [(0, 0, 1)]              # s=6: C00 += t6
    ]
    
    col = 0
    for s in range(7):
        # Inner Strassen for 2x2 * 2x2 (columns 0,1 of the 2x3 inner product)
        for si in range(7):
            # U: contributions from A entries
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V: contributions from B entries (only first 2 columns)
            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W: contributions to C entries (only first 2 columns)
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Schoolbook for 2x2 * 2x1 (column 2 of the 2x3 inner product)
        for u in range(2):
            for v in range(2):
                # U: A entry
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V: B entry (column 2)
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W: C entry (column 2)
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
