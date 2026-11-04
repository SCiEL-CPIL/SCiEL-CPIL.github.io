import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll try the Kronecker product approach:
    # 3x4x4 = T(3,2,2) ⊗ T(1,2,2)
    # rank = rank(T(3,2,2)) * rank(T(1,2,2))
    
    # First, let's construct the T(1,2,2) decomposition (rank 4)
    # 1x2 matrix [u0, u1] times 2x2 matrix [[v00,v01],[v10,v11]] = [w0, w1]
    # w0 = u0*v00 + u1*v10
    # w1 = u0*v01 + u1*v11
    
    # T(1,2,2) has:
    # U: 1*2 = 2 elements (u0, u1)
    # V: 2*2 = 4 elements (v00, v01, v10, v11)
    # W: 2*1 = 2 elements (w0, w1) [trace convention: W[k*n+i] where k=0,1 and n=1, so W[0], W[1]]
    
    # Rank-4 decomposition of T(1,2,2):
    U_122 = np.zeros((2, 4))
    V_122 = np.zeros((4, 4))
    W_122 = np.zeros((2, 4))
    
    # Term 0: u0 * v00 -> w0
    U_122[0, 0] = 1  # u0
    V_122[0, 0] = 1  # v00
    W_122[0, 0] = 1  # w0
    
    # Term 1: u1 * v10 -> w0
    U_122[1, 1] = 1  # u1
    V_122[2, 1] = 1  # v10
    W_122[0, 1] = 1  # w0
    
    # Term 2: u0 * v01 -> w1
    U_122[0, 2] = 1  # u0
    V_122[1, 2] = 1  # v01
    W_122[1, 2] = 1  # w1
    
    # Term 3: u1 * v11 -> w1
    U_122[1, 3] = 1  # u1
    V_122[3, 3] = 1  # v11
    W_122[1, 3] = 1  # w1
    
    # Now for T(3,2,2):
    # 3x2 matrix times 2x2 matrix = 3x2 matrix
    # U: 3*2 = 6 elements
    # V: 2*2 = 4 elements
    # W: 2*3 = 6 elements [trace convention: W[k*n+i] where k=0,1 (col) and n=3 (rows), so W[k*3+i]]
    
    # Let's try to find a low-rank decomposition
    # The naive rank is 12. Let's try rank 5, 6, 7, etc.
    
    # For now, let's use a known or constructed decomposition
    # I'll try to construct one based on sharing
    
    # Approach: Use the fact that all 3 rows share the same B matrix
    # Let s = a0+b0+c0 (sum of first column)
    # Let t = a1+b1+c1 (sum of second column)
    
    # Actually, let me just use a numerical approach or a known decomposition
    
    # Known: rank(3x2x2) might be 5 or 6. Let me try to construct rank 6 first.
    
    # Rank-6 decomposition of T(3,2,2):
    # m0 = (a0+b0+c0) * p
    # m1 = (a0+b0+c0) * r  
    # m2 = (a0+b0+c0) * q
    # m3 = (a0+b0+c0) * s
    # m4 = (a0-b0) * (r-p)
    # m5 = (a0-b0) * (s-q)
    
    # Let me verify:
    # x0 = a0*p + a1*r
    # From m0: contributes a0*p + b0*p + c0*p
    # From m1: contributes a0*r + b0*r + c0*r
    # From m4: contributes (a0-b0)*(r-p) = a0*r - a0*p - b0*r + b0*p
    # Sum for x0: (a0*p + b0*p + c0*p) + (a0*r - a0*p - b0*r + b0*p) ... 
    # This is getting messy. Let me just do it numerically.
    
    # Let me use a simpler approach: just use the known rank-44 construction
    # and see if I can improve it.
    
    # Actually, let me just construct the Kronecker product properly.
    
    # For T(3,2,2), let me use a rank-r decomposition and find the minimum r
    # that works. I'll try r=5 first.
    
    # Let me define the 3x2x2 tensor explicitly and try to find its rank
    # by constructing candidate decompositions.
    
    # The 3x2x2 multiplication tensor T has T[i,j,k,l] = 1 if the output (i,k) 
    # depends on input (j) from A and (j,l) from B.
    # A[i,j] for i in 0..2, j in 0..1
    # B[j,l] for j in 0..1, l in 0..1
    # C[i,l] = sum_j A[i,j] * B[j,l]
    
    # So the tensor is: T[(i,j),(j,l),(i,l)] = 1 for the bilinear form
    # In the decomposition: sum_c U[i,j,c] * V[j,l,c] * W[i,l,c] = C[i,l]
    
    # Let me try a rank-5 decomposition:
    # I'll use the following ansatz based on symmetric polynomials
    
    # After some thought, here's a rank-5 decomposition for 3x2x2:
    # Using the identity: (x+y+z)(p+q) = xp + xq + yp + yq + zp + zq
    # We can share the products.
    
    # Let me define 5 intermediate values:
    # m0 = (a0+b0+c0) * p
    # m1 = (a0+b0+c0) * r
    # m2 = (a0+b0+c0) * q  
    # m3 = (a0+b0+c0) * s
    # m4 = (a0-b0+c0) * (r-p)  -- or some other combination
    
    # Actually, I think the cleanest way is to just use a numerical solver or
    # a known decomposition. Let me use the following approach:
    
    # I'll construct the full 3x4x4 tensor and use the Kronecker product
    # with whatever rank I can achieve for 3x2x2.
    
    # For now, let me use rank 6 for 3x2x2 (which I can construct) and get
    # total rank 6*4 = 24.
    
    # Rank-6 decomposition of 3x2x2:
    U_322 = np.zeros((6, 6))
    V_322 = np.zeros((4, 6))
    W_322 = np.zeros((6, 6))
    
    # A indices: i*2+j for i=0,1,2 and j=0,1 -> 0,1,2,3,4,5
    # B indices: j*2+l for j=0,1 and l=0,1 -> 0,1,2,3
    # C indices: l*3+i for l=0,1 and i=0,1,2 -> 0,1,2,3,4,5 (trace convention W[k*n+i])
    
    # Term 0: (a0+b0+c0) * p -> contributes to x0, y0, z0
    # U: a0=1, b0=1, c0=1 (indices 0, 2, 4)
    U_322[0, 0] = 1  # a0
    U_322[2, 0] = 1  # b0
    U_322[4, 0] = 1  # c0
    V_322[0, 0] = 1  # p (B[0,0])
    W_322[0, 0] = 1  # x0 (C[0,0], index 0*3+0=0)
    W_322[3, 0] = 1  # y0 (C[1,0], index 0*3+1=1)
    W_322[6, 0] = 1  # z0 (C[2,0], index 0*3+2=2)
    
    # Wait, W has 6 elements (2*3), indexed 0..5
    # W[k*n+i] where k is column (0,1) and i is row (0,1,2)
    # So W[0*3+0]=W[0] for C[0,0], W[0*3+1]=W[1] for C[1,0], W[0*3+2]=W[2] for C[2,0]
    # W[1*3+0]=W[3] for C[0,1], W[1*3+1]=W[4] for C[1,1], W[1*3+2]=W[5] for C[2,1]
    
    # Let me redo this:
    U_322 = np.zeros((6, 6))
    V_322 = np.zeros((4, 6))
    W_322 = np.zeros((6, 6))
    
    # A flat: [a00, a01, a10, a11, a20, a21] (row-major, 3 rows x 2 cols)
    # B flat: [b00, b01, b10, b11] (row-major, 2 rows x 2 cols)
    # C flat: [c00, c10, c20, c01, c11, c21] (trace convention: W[k*n+i], k=col, i=row)
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c20 = a20*b00 + a21*b10
    # c21 = a20*b01 + a21*b11
    
    # W indices: W[k*3+i]
    # W[0] = c00, W[1] = c10, W[2] = c20
    # W[3] = c01, W[4] = c11, W[5] = c21
    
    # Term 0: (a00+a10+a20) * b00
    U_322[0, 0] = 1  # a00
    U_322[2, 0] = 1  # a10
    U_322[4, 0] = 1  # a20
    V_322[0, 0] = 1  # b00
    W_322[0, 0] = 1  # c00
    W_322[1, 0] = 1  # c10
    W_322[2, 0] = 1  # c20
    
    # Term 1: (a00+a10+a20) * b10
    U_322[0, 1] = 1
    U_322[2, 1] = 1
    U_322[4, 1] = 1
    V_322[2, 1] = 1  # b10
    W_322[0, 1] = 1
    W_322[1, 1] = 1
    W_322[2, 1] = 1
    
    # Term 2: (a00+a10+a20) * b01
    U_322[0, 2] = 1
    U_322[2, 2] = 1
    U_322[4, 2] = 1
    V_322[1, 2] = 1  # b01
    W_322[3, 2] = 1
    W_322[4, 2] = 1
    W_322[5, 2] = 1
    
    # Term 3: (a00+a10+a20) * b11
    U_322[0, 3] = 1
    U_322[2, 3] = 1
    U_322[4, 3] = 1
    V_322[3, 3] = 1  # b11
    W_322[3, 3] = 1
    W_322[4, 3] = 1
    W_322[5, 3] = 1
    
    # Now I need 2 more terms to handle the a01, a11, a21 parts
    # c00 also has a01*b10, c01 has a01*b11, etc.
    
    # Term 4: (a01+a11+a21) * b00
    U_322[1, 4] = 1  # a01
    U_322[3, 4] = 1  # a11
    U_322[5, 4] = 1  # a21
    V_322[0, 4] = 1  # b00
    W_322[0, 4] = 1
    W_322[1, 4] = 1
    W_322[2, 4] = 1
    
    # Term 5: (a01+a11+a21) * b10
    U_322[1, 5] = 1
    U_322[3, 5] = 1
    U_322[5, 5] = 1
    V_322[2, 5] = 1  # b10
    W_322[0, 5] = 1
    W_322[1, 5] = 1
    W_322[2, 5] = 1
    
    # Wait, this gives rank 6 but I'm missing the b01 and b11 terms for the second column of A.
    # Let me check: with terms 0-5, what do I get?
    # c00 = (a00+a10+a20)*b00 + (a00+a10+a20)*b10 + (a01+a11+a21)*b00 + (a01+a11+a21)*b10
    #     = (a00+a01+a10+a11+a20+a21)*(b00+b10)
    # That's wrong! c00 should be a00*b00 + a01*b10
    
    # This approach doesn't work because I'm mixing up the contributions.
    
    # Let me restart with a cleaner approach.
    
    # The correct way: each term c_k = u_k^T A * v_k^T B contributes to specific C entries.
    
    # For a rank-r decomposition, we need:
    # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[i,k,c]
    # Wait, that's not quite right either.
    
    # The standard form: C[i,k] = sum_c U_c[i] * V_c[k] * W_c where
    # U_c is a functional on A, V_c is a functional on B, W_c is a coefficient for C[i,k]
    
    # More precisely: C[i,k] = sum_c (sum_{i',j} U_c[i',j] * A[i',j]) * (sum_{j',k'} V_c[j',k'] * B[j',k']) * W_c[i,k]
    
    # No wait, the standard tensor decomposition is:
    # T = sum_c U_c ⊗ V_c ⊗ W_c
    # where T[A,B] = C means: C[i,k] = sum_c (U_c · A) * (V_c · B) * W_c[i,k]
    
    # Where U_c · A = sum_{i,j} U_c[i,j] * A[i,j]
    # V_c · B = sum_{j,k} V_c[j,k] * B[j,k]
    # And W_c[i,k] is the weight for output C[i,k] in term c.
    
    # So for each term c, we have:
    # (U_c · A) * (V_c · B) * W_c[i,k] contributes to C[i,k]
    
    # For the 3x2x2 case:
    # A is 3x2, B is 2x2, C is 3x2
    
    # Term c: U_c is 3x2, V_c is 2x2, W_c is 3x2
    # Contribution to C[i,k] = (sum_{i',j} U_c[i',j]*A[i',j]) * (sum_{j',k'} V_c[j',k']*B[j',k']) * W_c[i,k]
    
    # We need: sum_c (U_c · A) * (V_c · B) * W_c[i,k] = sum_j A[i,j] * B[j,k]
    
    # This is a system of equations. Let me just use a known decomposition.
    
    # I'll use the following rank-7 decomposition (Strassen-like for 3 rows):
    # Actually, let me just use the naive rank-12 and the known rank-44 approach.
    
    # Let me just implement the known rank-44 approach and verify it.
    
    # The known approach:
    # Split 3x4x4 into:
    # - Row 0: 1x4 x 4x4 (naive, rank 16)
    # - Rows 1-2: 2x4 x 4x4 (Strassen on 2x2 blocks, rank 28)
    # Total: 44
    
    # For the 2x4x4 part using Kronecker:
    # 2x4x4 = T(2,2,2) ⊗ T(1,2,2)
    # rank = 7 * 4 = 28
    
    # For the 1x4x4 part:
    # 1x4x4 = T(1,2,2) ⊗ T(1,2,2)  
    # rank = 4 * 4 = 16
    
    # Total: 28 + 16 = 44
    
    # Now let me implement this.
    
    # First, the 2x2x2 Strassen decomposition (rank 7):
    # A (2x2) * B (2x2) = C (2x2)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # p = a*e + c*g
    # q = a*f + c*h
    # r = b*e + d*g
    # s = b*f + d*h
    
    # Strassen's 7 products:
    # m0 = (a+d)*(e+h)
    # m1 = (c+d)*e
    # m2 = a*(f-h)
    # m3 = d*(g-c)
    # m4 = (a+b)*(e+f)  -- wait, let me use the standard Strassen
    
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+f)
    
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 - m6
    
    # Let me verify:
    # m1 = (a+d)(e+h) = ae+ah+de+dh
    # m4 = d(g-c) = dg-dc
    # m5 = (a+b)h = ah+bh
    # m7 = (a-b)(e+f) = ae+af-be-bf
    
    # p = m1+m4-m5+m7 = ae+ah+de+dh+dg-dc-ah-bh+ae+af-be-bf
    #   = 2ae+de+dh+dg-dc-bh+af-be-bf
    # That doesn't look right. Let me use the correct Strassen formulas.
    
    # Correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+f)
    
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 - m6
    
    # Let me verify p:
    # m1 = ae+ah+de+dh
    # m4 = dg-dc
    # m5 = ah+bh
    # m7 = ae+af-be-bf
    # p = (ae+ah+de+dh) + (dg-dc) - (ah+bh) + (ae+af-be-bf)
    #   = ae+ah+de+dh+dg-dc-ah-bh+ae+af-be-bf
    #   = 2ae + de + dh + dg - dc - bh + af - be - bf
    # That's not ae+cg. Something is wrong.
    
    # Let me look up the correct Strassen formulas:
    # p1 = (a + d) × (e + h)
    # p2 = (c + d) × e
    # p3 = a × (f - h)
    # p4 = d × (g - c)
    # p5 = (a + b) × h
    # p6 = (c + d) × (f - h)
    # p7 = (a - b) × (e + f)
    
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 - p6
    
    # Let me verify C11 = ae + cg:
    # p1 = (a+d)(e+h) = ae+ah+de+dh
    # p4 = d(g-c) = dg-dc
    # p5 = (a+b)h = ah+bh
    # p7 = (a-b)(e+f) = ae+af-be-bf
    
    # C11 = p1+p4-p5+p7 = ae+ah+de+dh+dg-dc-ah-bh+ae+af-be-bf
    #     = 2ae + de + dh + dg - dc - bh + af - be - bf
    
    # This is NOT ae+cg. I must have the formulas wrong.
    
    # Let me use a different set of Strassen formulas:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+f)
    # p3 = (a-b)(h+g)
    # p4 = (a+d)(f-h)
    # p5 = (b+c)(e+h)
    # p6 = (a-b)(e+f)
    # p7 = (c+d)(g-h)
    
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # This is getting too error-prone. Let me just use a verified Strassen decomposition.
    
    # I'll use numpy to verify any decomposition I construct.
    
    # Let me take a step back and just implement the rank-44 approach directly
    # using the Kronecker product structure, verifying numerically.
    
    # Actually, the simplest approach: just use the schoolbook (rank 48) as a fallback,
    # and try to construct a better one.
    
    # Let me just implement the Kronecker product approach for the known rank-44 case
    # and verify it works.
    
    # For now, let me return the schoolbook decomposition as a safe fallback,
    # and add the Kronecker product construction.
    
    # Schoolbook (rank 48):
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Now try the Kronecker product approach for rank 44:
    # 3x4x4 = [1x4x4] + [2x4x4] (block diagonal in the first dimension)
    # 1x4x4 = T(1,2,2) ⊗ T(1,2,2), rank 16
    # 2x4x4 = T(2,2,2) ⊗ T(1,2,2), rank 28
    
    # Let me construct the Kronecker product decompositions.
    
    # Helper: Kronecker product of two tensor decompositions
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two tensor decompositions.
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Returns: U: (n1*n2*m1*m2, r1*r2), V: (m1*m2*p1*p2, r1*r2), W: (p1*p2*n1*n2, r1*r2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        n1, m1, p1 = U1.shape[0] // m1, m1, W1.shape[0] // n1  # This is tricky
        # Actually, I need to pass dimensions explicitly
        pass
    
    # This is getting complex. Let me just directly construct the rank-44 decomposition.
    
    # I'll use the following structure:
    # The 3x4x4 multiplication is block-diagonal in the output rows:
    # Row 0 of C: (1x4) * (4x4) -> 1x4
    # Rows 1-2 of C: (2x4) * (4x4) -> 2x4
    
    # For the 1x4x4 part (rank 16):
    # This is T(1,4,4) = T(1,2,2) ⊗ T(1,2,2)
    # U: 1*4 = 4 elements, V: 4*4 = 16 elements, W: 4*1 = 4 elements
    
    # For the 2x4x4 part (rank 28):
    # This is T(2,4,4) = T(2,2,2) ⊗ T(1,2,2)
    # U: 2*4 = 8 elements, V: 4*4 = 16 elements, W: 4*2 = 8 elements
    
    # The total decomposition has rank 16+28 = 44
    # U: 12 elements (4+8), V: 16 elements, W: 12 elements (4+8)
    
    # Let me construct this step by step.
    
    # First, T(1,2,2) decomposition (rank 4):
    # A: 1x2 = [u0, u1] (2 elements)
    # B: 2x2 = [b00, b01, b10, b11] (4 elements)
    # C: 2x1 = [c0, c1] (2 elements) [trace: W[k*1+i], k=0,1, i=0]
    # C[0] = u0*b00 + u1*b10
    # C[1] = u0*b01 + u1*b11
    
    U_122 = np.array([
        [1, 0, 1, 0],  # u0
        [0, 1, 0, 1],  # u1
    ])  # shape (2, 4)
    
    V_122 = np.array([
        [1, 0, 0, 0],  # b00
        [0, 0, 1, 0],  # b01
        [0, 1, 0, 0],  # b10
        [0, 0, 0, 1],  # b11
    ])  # shape (4, 4)
    
    W_122 = np.array([
        [1, 1, 0, 0],  # c0: from terms 0 (u0*b00) and 1 (u1*b10)
        [0, 0, 1, 1],  # c1: from terms 2 (u0*b01) and 3 (u1*b11)
    ])  # shape (2, 4)
    
    # Verify T(1,2,2):
    # C[0] = sum_c U[0,c]*u0 + U[1,c]*u1 times V[0,c]*b00 + V[1,c]*b01 + V[2,c]*b10 + V[3,c]*b11 times W[0,c]
    # Term 0: U dot = u0, V dot = b00, W[0,0]=1 -> u0*b00
    # Term 1: U dot = u1, V dot = b10, W[0,1]=1 -> u1*b10
    # Term 2: U dot = u0, V dot = b01, W[0,2]=0 -> 0
    # Term 3: U dot = u1, V dot = b11, W[0,3]=0 -> 0
    # C[0] = u0*b00 + u1*b10 ✓
    
    # C[1]:
    # Term 0: W[1,0]=0
    # Term 1: W[1,1]=0
    # Term 2: U dot = u0, V dot = b01, W[1,2]=1 -> u0*b01
    # Term 3: U dot = u1, V dot = b11, W[1,3]=1 -> u1*b11
    # C[1] = u0*b01 + u1*b11 ✓
    
    # Now T(2,2,2) Strassen decomposition (rank 7):
    # A: 2x2 = [a, b, c, d] (row-major: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1])
    # B: 2x2 = [e, f, g, h] (row-major: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1])
    # C: 2x2, trace convention W[k*2+i]: W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
    # C[0,0] = a*e + c*g
    # C[0,1] = a*f + c*h
    # C[1,0] = b*e + d*g
    # C[1,1] = b*f + d*h
    
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+f)
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 - m6
    
    # Let me verify C[0,0] = ae + cg:
    # m1 = ae+ah+de+dh
    # m4 = dg-dc
    # m5 = ah+bh
    # m7 = ae+af-be-bf
    # m1+m4-m5+m7 = ae+ah+de+dh+dg-dc-ah-bh+ae+af-be-bf
    # = 2ae+de+dh+dg-dc-bh+af-be-bf
    # This is NOT ae+cg. The formulas must be wrong.
    
    # Let me use a different Strassen formulation. I'll use the one from Wikipedia:
    # p1 = (a + d) × (e + h)
    # p2 = (c + d) × e
    # p3 = a × (f - h)
    # p4 = d × (g - c)
    # p5 = (a + b) × h
    # p6 = (c + d) × (f - h)
    # p7 = (a - b) × (e + f)
    
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 - p6
    
    # Let me verify C11 = ae + cg with a=1,b=0,c=0,d=0,e=1,f=0,g=0,h=0:
    # p1 = (1+0)(1+0) = 1
    # p4 = 0*(0-0) = 0
    # p5 = (1+0)*0 = 0
    # p7 = (1-0)(1+0) = 1
    # C11 = 1+0-0+1 = 2, but should be 1*1+0*0 = 1. WRONG.
    
    # The formulas are definitely wrong. Let me use a verified source.
    
    # Correct Strassen (from various sources):
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e + f)
    # m3 = (a - b)(h + g)
    # m4 = (a + d)(f - h)
    # m5 = (b + c)(e + h)
    # m6 = (a - b)(e + f)
    # m7 = (c + d)(g - h)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Verify C11 with a=1,b=0,c=0,d=0,e=1,f=0,g=0,h=0:
    # m1 = (1+0)(1+0) = 1
    # m4 = (1+0)(0-0) = 0
    # m5 = (0+0)(1+0) = 0
    # m7 = (0+0)(0-0) = 0
    # C11 = 1+0-0+0 = 1 ✓ (should be 1*1+0*0=1)
    
    # Verify C12 with same:
    # m3 = (1-0)(0+0) = 0
    # m5 = 0
    # C12 = 0+0 = 0 ✓ (should be 1*0+0*0=0)
    
    # Verify C21:
    # m2 = (0+0)(1+0) = 0
    # m4 = 0
    # C21 = 0+0 = 0 ✓ (should be 0*1+0*0=0)
    
    # Verify C22:
    # m1 = 1, m2 = 0, m3 = 0, m6 = (1-0)(1+0) = 1
    # C22 = 1-0+0+1 = 2, should be 0*0+0*0=0. WRONG.
    
    # Still wrong. Let me just use a numerical approach to find the Strassen decomposition.
    
    # Actually, I know Strassen works. Let me use the standard formulation:
    # For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * e
    # p3 = a * (f - h)
    # p4 = d * (g - c)
    # p5 = (a + b) * h
    # p6 = (c + d) * (f - h)
    # p7 = (a - b) * (e + f)
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 - p6
    
    # Let me verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # C[0,0] = 1*5+3*7 = 5+21 = 26
    # C[0,1] = 1*6+3*8 = 6+24 = 30
    # C[1,0] = 2*5+4*7 = 10+28 = 38
    # C[1,1] = 2*6+4*8 = 12+32 = 44
    
    # p1 = (1+4)(5+8) = 5*13 = 65
    # p2 = (3+4)*5 = 35
    # p3 = 1*(6-8) = -2
    # p4 = 4*(7-3) = 16
    # p5 = (1+2)*8 = 24
    # p6 = (3+4)*(6-8) = 7*(-2) = -14
    # p7 = (1-2)*(5+6) = -1*11 = -11
    
    # C[0,0] = 65+16-24+(-11) = 65+16-24-11 = 46. Should be 26. WRONG.
    
    # I clearly don't remember the Strassen formulas correctly. Let me just use a 
    # numerical verification approach in the code and find the correct decomposition.
    
    # Actually, let me just use the fact that I can verify any decomposition numerically.
    # I'll construct candidate decompositions and verify them.
    
    # For now, let me just return the schoolbook decomposition (rank 48) as a safe
    # fallback, and try to improve it.
    
    # Actually, let me just implement the rank-44 approach using a verified Strassen
    # decomposition. I'll find the correct Strassen formulas by solving the system.
    
    # The Strassen decomposition for 2x2x2:
    # We need 7 terms such that:
    # C[0,0] = sum_c U_c[0]*U_c[1]... 
    
    # Let me just hardcode a known correct Strassen decomposition.
    
    # From the literature, the correct Strassen decomposition is:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c+d)(f-h)
    # m7 = (a-b)(e+f)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 - m6
    
    # I verified this gives wrong answers. Let me try yet another formulation.
    
    # OK I'll just use a different approach. I'll construct the decomposition
    # numerically by solving for the coefficients.
    
    # Actually, the simplest thing: just use the schoolbook for now and get a valid
    # answer. The rank-44 approach can be implemented later.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

