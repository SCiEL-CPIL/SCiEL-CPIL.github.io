import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    # We will construct a decomposition using a block approach.
    # 3x4x7 = 3x4x3 + 3x4x4
    # We will use:
    # 1. A rank-18 decomposition for 3x4x3 (or better if found, but 18 is a known bound? Let's check.)
    #    Actually, for 3x4x3, the schoolbook is 36.
    #    Strassen for 2x2x2 is rank 7.
    #    Let's try to construct a good decomposition for 3x4x3 and 3x4x4.
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Helper to combine two decompositions (block diagonal)
    # d1: (U1, V1, W1, r1) for n1 x m1 x p1
    # d2: (U2, V2, W2, r2) for n2 x m2 x p2
    # This assumes the decompositions are for the same n, m but different p splits?
    # No, the "sum" of multiplications is not simply block diagonal in U,V,W unless the indices don't overlap.
    # For 3x4x7 = 3x4x3 + 3x4x4, the A and B matrices are shared.
    # C = A * B_full. B_full = [B1 | B2].
    # C1 = A * B1 (3x3), C2 = A * B2 (3x4).
    # The tensor is T[i, j, k] = 1.
    # We can split the k dimension.
    # The decomposition for the first part (k=0..2) and second part (k=3..6) are independent in terms of the k-index in W and V.
    # However, U is shared.
    # If we have a decomposition for 3x4x3 with rank R1, and 3x4x4 with rank R2.
    # U1 is 12xR1, V1 is 4x3=12 x R1, W1 is 3x3=9 x R1.
    # U2 is 12xR2, V2 is 4x4=16 x R2, W2 is 4x3=12 x R2.
    # We can form U = [U1 | U2] (12 x (R1+R2)).
    # V: The V indices are j*p+k.
    # For the first part, j is 0..3, k is 0..2. Indices 0..11.
    # For the second part, j is 0..3, k is 3..6. Indices 12..27 (if p=7).
    # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank).
    # The V1 part corresponds to k=0..2. The indices in V are j*7+k.
    # The V2 part corresponds to k=3..6. The indices in V are j*7+k.
    # So we can place the columns of V1 into the rows corresponding to k=0..2 and V2 into rows for k=3..6.
    # Specifically, V1 has rows indexed by (j, k1) where k1 in 0..2. In the full V, these are rows j*7+k1.
    # V2 has rows indexed by (j, k2) where k2 in 3..6. In the full V, these are rows j*7+k2.
    # So we can construct V by scattering V1 and V2 into the correct rows.
    # Similarly for W.
    # W has shape (p*n, rank) = (7*3, rank) = (21, rank).
    # W1 indices are (k1, i) where k1 in 0..2. In full W, these are rows k1*3+i.
    # W2 indices are (k2, i) where k2 in 3..6. In full W, these are rows k2*3+i.
    # U is shared, so we just concatenate columns.
    
    def combine_split_p(n, m, p1, p2, d1, d2):
        # d1: U1 (n*m, r1), V1 (m*p1, r1), W1 (p1*n, r1)
        # d2: U2 (n*m, r2), V2 (m*p2, r2), W2 (p2*n, r2)
        # Full p = p1 + p2
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        R = r1 + r2
        
        U = np.hstack([U1, U2]) # (n*m, R)
        
        # Construct V (m*p, R)
        V = np.zeros((m * (p1 + p2), R))
        # Place V1
        for j in range(m):
            for k in range(p1):
                V[j * (p1 + p2) + k, :r1] = V1[j * p1 + k, :]
        # Place V2
        for j in range(m):
            for k in range(p2):
                V[j * (p1 + p2) + p1 + k, r1:] = V2[j * p2 + k, :]
                
        # Construct W (p*n, R)
        W = np.zeros(((p1 + p2) * n, R))
        # Place W1
        for k in range(p1):
            for i in range(n):
                W[k * n + i, :r1] = W1[k * n + i, :]
        # Place W2
        for k in range(p2):
            for i in range(n):
                W[(p1 + k) * n + i, r1:] = W2[k * n + i, :]
                
        return U, V, W, R

    # We need good decompositions for 3x4x3 and 3x4x4.
    # Let's try to find them or use known ones.
    # For 3x4x3:
    # Schoolbook is 36.
    # We can split 3x4x3 into 2x4x3 + 1x4x3.
    # 1x4x3 is rank 12 (schoolbook) or less? 1x4x3 is just 4*3=12 terms.
    # 2x4x3:
    # Split 2x4x3 into 2x2x3 + 2x2x3.
    # 2x2x3:
    # Split 2x2x3 into 2x2x2 + 2x2x1.
    # 2x2x2 is Strassen rank 7.
    # 2x2x1 is rank 4 (schoolbook 2*2*1=4).
    # So 2x2x3 rank <= 7+4=11.
    # 2x4x3 rank <= 11+11=22.
    # 3x4x3 rank <= 22+12=34.
    # This is better than 36.
    
    # Can we do better?
    # Maybe 3x4x3 has rank 24?
    # Let's check if we can use a known decomposition.
    # Actually, let's just implement the recursive splitting with Strassen for 2x2x2.
    
    def strassen_2x2x2():
        # Rank 7 decomposition for 2x2x2
        # U, V, W are 4x7
        # Indices: i in 0,1; j in 0,1; k in 0,1
        # U[i*2+j, r]
        # V[j*2+k, r]
        # W[k*2+i, r]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # c11 = M1 + M4 - M5 - M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M7
        
        # Let's map these to U, V, W.
        # r=0: M1. U: a11, a22. V: b11, b22. W: c11, c22.
        # U[0,0]=1, U[3,0]=1. V[0,0]=1, V[3,0]=1. W[0,0]=1, W[3,0]=1. (c11 is W[0*2+0,0], c22 is W[1*2+1,0])
        
        # r=1: M2. U: a21, a22. V: b11. W: c21, c22.
        # U[2,1]=1, U[3,1]=1. V[0,1]=1. W[1*2+0,1]=1 (c21), W[1*2+1,1]=1 (c22).
        
        # r=2: M3. U: a11. V: b21, b22. W: c12, c22.
        # U[0,2]=1. V[2,2]=1, V[3,2]=1. W[0*2+1,2]=1 (c12), W[1*2+1,2]=1 (c22).
        
        # r=3: M4. U: a22. V: b12, b11. W: c11, c21.
        # U[3,3]=1. V[1,3]=1, V[0,3]=-1. W[0,3]=1 (c11), W[2,3]=1 (c21).
        
        # r=4: M5. U: a11, a12. V: b22. W: c11, c12.
        # U[0,4]=1, U[1,4]=1. V[3,4]=1. W[0,4]=-1 (c11), W[1,4]=1 (c12).
        
        # r=5: M6. U: a21, a11. V: b11, b12. W: c21.
        # U[2,5]=1, U[0,5]=-1. V[0,5]=1, V[1,5]=1. W[2,5]=1 (c21).
        
        # r=6: M7. U: a12, a22. V: b21, b22. W: c11, c22.
        # U[1,6]=1, U[3,6]=-1. V[2,6]=1, V[3,6]=1. W[0,6]=-1 (c11), W[3,6]=1 (c22).
        
        # Let's fill the arrays
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[2,2]=1; V[3,2]=1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=-1; W[3,6]=1
        
        return U, V, W, 7

    def embed(n, m, p, sub_n, sub_m, sub_p, off_i, off_j, off_k, d_sub):
        # Embed a decomposition of sub_n x sub_m x sub_p into n x m x p
        # at offsets off_i, off_j, off_k
        U_sub, V_sub, W_sub, r = d_sub
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for i in range(sub_n):
            for j in range(sub_m):
                for k in range(sub_p):
                    # U index: (off_i + i) * m + (off_j + j)
                    u_idx = (off_i + i) * m + (off_j + j)
                    # V index: (off_j + j) * p + (off_k + k)
                    v_idx = (off_j + j) * p + (off_k + k)
                    # W index: (off_k + k) * n + (off_i + i)
                    w_idx = (off_k + k) * n + (off_i + i)
                    
                    U[u_idx, :] = U_sub[i * sub_m + j, :]
                    V[v_idx, :] = V_sub[j * sub_p + k, :]
                    W[w_idx, :] = W_sub[k * sub_n + i, :]
                    
        return U, V, W, r

    def add_decomps(d1, d2):
        U1, V1, W1, r1 = d1
        U2, V2, W2, r2 = d2
        R = r1 + r2
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        W = np.hstack([W1, W2])
        return U, V, W, R

    # Build 2x2x3
    def build_2x2x3():
        # 2x2x2 + 2x2x1
        d_222 = strassen_2x2x2()
        d_221 = schoolbook(2, 2, 1)
        
        # Embed 2x2x2 at k=0
        U1, V1, W1, r1 = embed(2, 2, 3, 2, 2, 2, 0, 0, 0, d_222)
        # Embed 2x2x1 at k=1
        U2, V2, W2, r2 = embed(2, 2, 3, 2, 2, 1, 0, 0, 1, d_221)
        
        return add_decomps((U1, V1, W1, r1), (U2, V2, W2, r2))

    # Build 2x4x3
    def build_2x4x3():
        # 2x2x3 + 2x2x3
        d1 = build_2x2x3()
        d2 = build_2x2x3()
        
        # Embed first at j=0..1
        U1, V1, W1, r1 = embed(2, 4, 3, 2, 2, 3, 0, 0, 0, d1)
        # Embed second at j=2..3
        U2, V2, W2, r2 = embed(2, 4, 3, 2, 2, 3, 0, 2, 0, d2)
        
        return add_decomps((U1, V1, W1, r1), (U2, V2, W2, r2))

    # Build 3x4x3
    def build_3x4x3():
        # 2x4x3 + 1x4x3
        d1 = build_2x4x3()
        d2 = schoolbook(1, 4, 3)
        
        # Embed first at i=0..1
        U1, V1, W1, r1 = embed(3, 4, 3, 2, 4, 3, 0, 0, 0, d1)
        # Embed second at i=2
        U2, V2, W2, r2 = embed(3, 4, 3, 1, 4, 3, 2, 0, 0, d2)
        
        return add_decomps((U1, V1, W1, r1), (U2, V2, W2, r2))

    # Build 3x4x4
    def build_3x4x4():
        # 3x4x3 + 3x4x1
        d1 = build_3x4x3()
        d2 = schoolbook(3, 4, 1)
        
        # Embed first at k=0..2
        U1, V1, W1, r1 = embed(3, 4, 4, 3, 4, 3, 0, 0, 0, d1)
        # Embed second at k=3
        U2, V2, W2, r2 = embed(3, 4, 4, 3, 4, 1, 0, 0, 3, d2)
        
        return add_decomps((U1, V1, W1, r1), (U2, V2, W2, r2))

    # Build 3x4x7
    def build_3x4x7():
        # 3x4x3 + 3x4x4
        d1 = build_3x4x3()
        d2 = build_3x4x4()
        
        # Embed first at k=0..2
        U1, V1, W1, r1 = embed(3, 4, 7, 3, 4, 3, 0, 0, 0, d1)
        # Embed second at k=3..6
        U2, V2, W2, r2 = embed(3, 4, 7, 3, 4, 4, 0, 0, 3, d2)
        
        return add_decomps((U1, V1, W1, r1), (U2, V2, W2, r2))

    U, V, W, rank = build_3x4x7()
    
    # Verify
    if time.time() < deadline - 5.0:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if not valid:
            # Fallback to schoolbook if construction failed
            U, V, W, rank = schoolbook(3, 4, 7)

    return {"U": U, "V": V, "W": W}
