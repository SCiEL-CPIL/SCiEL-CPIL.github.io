"""Matrix multiplication 6x6x6 using 3x3 block decomposition with 2x2 Strassen (rank 7)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s limit
    
    n, m, p = 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # C[i,k] = sum_c U[i,j]_c * V[j,k]_c * W[k,i]_c ... 
    # Actually the tensor is T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Where T is the matmul tensor: T_{i,j,k} = 1 if we compute C[i,k] += A[i,j]B[j,k]
    # Wait, the standard definition:
    # Matmul tensor M_{n,m,p} has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # M_{i,j,k} = 1.
    # Decomposition: M_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Here U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    # Let's re-verify the Strassen indices.
    # For 2x2:
    # U[0,0]=1, U[3,0]=1 (A00, A11)
    # V[0,0]=1, V[3,0]=1 (B00, B11)
    # W[0,0]=1, W[3,0]=1 (C00, C11) -> W[k*n+i] -> W[0*2+0]=W[0], W[1*2+1]=W[3]
    # Product: A00*B00*C00 + A11*B11*C11.
    # We want C00 to get A00*B00 and C11 to get A11*B11.
    # The term is U[i,j] V[j,k] W[k,i].
    # If i=0, k=0: W[0,0] is used. Term: U[0,j] V[j,0] W[0,0].
    # If j=0: U[0,0] V[0,0] W[0,0] = 1*1*1 = 1. This contributes to C[0,0]. Correct.
    # If i=1, k=1: W[1,1] -> W[1*2+1]=W[3].
    # If j=1: U[1,1] V[1,1] W[3,1]? No, V index is j*p+k. For 2x2, p=2. V[1*2+1]=V[3].
    # U[1,1] is U[1*2+1]=U[3].
    # So U[3] V[3] W[3] = 1*1*1 = 1. Contributes to C[1,1]. Correct.
    
    # Let's build the 2x2 Strassen matrices explicitly.
    # U is 4x7, V is 4x7, W is 4x7.
    # Indices:
    # U[i*2+j]
    # V[j*2+k]
    # W[k*2+i]
    
    U22 = np.zeros((4, 7), dtype=int)
    V22 = np.zeros((4, 7), dtype=int)
    W22 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00(0), A11(3) -> 1, 1
    # V: B00(0), B11(3) -> 1, 1
    # W: C00(0), C11(3) -> 1, 1
    U22[0,0] = 1; U22[3,0] = 1
    V22[0,0] = 1; V22[3,0] = 1
    W22[0,0] = 1; W22[3,0] = 1
    
    # M2 = (A10+A11)(B00)
    # U: A10(2), A11(3) -> 1, 1
    # V: B00(0) -> 1
    # W: C10(2), C11(3) -> 1, -1
    # Check: C10 += A10*B00 + A11*B00. C11 -= A11*B00.
    # Term for C10 (i=1, k=0): W[0*2+1]=W[2].
    # j=0: U[1,0]=U[2]=1, V[0,0]=V[0]=1, W[2]=1. Product 1. Correct.
    # j=1: U[1,1]=U[3]=1, V[1,0]=V[2]=0. Product 0.
    # Wait, V index is j*p+k. For j=1, k=0, V[1*2+0]=V[2]. V[2,1] is 0. Correct.
    # Term for C11 (i=1, k=1): W[1*2+1]=W[3].
    # j=0: U[2]=1, V[0]=1, W[3]=-1. Product -1. Correct.
    # j=1: U[3]=1, V[2]=0. Product 0.
    U22[2,1] = 1; U22[3,1] = 1
    V22[0,1] = 1
    W22[2,1] = 1; W22[3,1] = -1
    
    # M3 = (A00)(B10-B11)
    # U: A00(0) -> 1
    # V: B10(2), B11(3) -> 1, -1
    # W: C01(1), C11(3) -> 1, 1
    # Check C01 (i=0, k=1): W[1*2+0]=W[2]? No. W[k*n+i]. k=1, i=0. W[1*2+0]=W[2].
    # Wait. W index is k*n+i.
    # C01 is i=0, k=1. Index = 1*2+0 = 2.
    # C11 is i=1, k=1. Index = 1*2+1 = 3.
    # My previous W22[1,2] was for C01?
    # Let's re-verify W index mapping.
    # W[k*n+i].
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 2.
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # Let's restart W assignments with correct indices.
    # W[0] = C00
    # W[1] = C10
    # W[2] = C01
    # W[3] = C11
    
    # M1: C00 += 1, C11 += 1
    W22[0,0] = 1; W22[3,0] = 1
    
    # M2: C10 += 1, C11 -= 1
    # C10 is W[1]. C11 is W[3].
    W22[1,1] = 1; W22[3,1] = -1
    
    # M3: C01 += 1, C11 += 1
    # C01 is W[2]. C11 is W[3].
    W22[2,2] = 1; W22[3,2] = 1
    # U: A00(0) -> 1. U22[0,2]=1.
    # V: B10(2), B11(3) -> 1, -1. V22[2,2]=1, V22[3,2]=-1.
    U22[0,2] = 1
    V22[2,2] = 1; V22[3,2] = -1
    
    # M4: (A11)(B01-B00)
    # C00 += 1, C10 += 1
    # C00 is W[0]. C10 is W[1].
    W22[0,3] = 1; W22[1,3] = 1
    # U: A11(3) -> 1. U22[3,3]=1.
    # V: B01(1), B00(0) -> 1, -1. V22[1,3]=1, V22[0,3]=-1.
    U22[3,3] = 1
    V22[1,3] = 1; V22[0,3] = -1
    
    # M5: (A00+A01)(B11)
    # C00 -= 1, C01 += 1
    # C00 is W[0]. C01 is W[2].
    W22[0,4] = -1; W22[2,4] = 1
    # U: A00(0), A01(1) -> 1, 1. U22[0,4]=1, U22[1,4]=1.
    # V: B11(3) -> 1. V22[3,4]=1.
    U22[0,4] = 1; U22[1,4] = 1
    V22[3,4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # C11 += 1
    # C11 is W[3].
    W22[3,5] = 1
    # U: A10(2), A00(0) -> 1, -1. U22[2,5]=1, U22[0,5]=-1.
    # V: B00(0), B01(1) -> 1, 1. V22[0,5]=1, V22[1,5]=1.
    U22[2,5] = 1; U22[0,5] = -1
    V22[0,5] = 1; V22[1,5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # C00 += 1
    # C00 is W[0].
    W22[0,6] = 1
    # U: A01(1), A11(3) -> 1, -1. U22[1,6]=1, U22[3,6]=-1.
    # V: B10(2), B11(3) -> 1, 1. V22[2,6]=1, V22[3,6]=1.
    U22[1,6] = 1; U22[3,6] = -1
    V22[2,6] = 1; V22[3,6] = 1
    
    # Verify 2x2 Strassen
    def verify_decomp(U, V, W, n, m, p):
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Check if sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k
        # Actually, the tensor T_{i,j,k} = 1.
        # Decomposition: T_{i,j,k} = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # This is not quite right.
        # The matmul tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k].
        # So the tensor entry T_{i,j,k} is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        # Yes, T_{i,j,k} = 1 for all i,j,k.
        # The decomposition is: 1 = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Where U[i,j,c] is U[i*m+j, c], etc.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        return False
        return True

    if not verify_decomp(U22, V22, W22, 2, 2, 2):
        # Fallback to schoolbook 6x6
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now lift to 6x6 using 3x3 blocks of 2x2.
    # We can use the "naive" block sum (rank 189) or Laderman (rank 133).
    # Let's try to implement Laderman 19 for the 3x3 block structure.
    # If Laderman fails or is too complex, fall back to 189.
    
    # Laderman 19 for 3x3 matmul.
    # U33 is 9x19, V33 is 9x19, W33 is 9x19.
    # Indices: U[bi*3+bj], V[bj*3+bk], W[bk*3+bi].
    
    U33 = np.zeros((9, 19), dtype=int)
    V33 = np.zeros((9, 19), dtype=int)
    W33 = np.zeros((9, 19), dtype=int)
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 19)
    # Reference: https://en.wikipedia.org/wiki/Laderman%27s_algorithm
    
    # M1 = (A00+A11+A22)(B00+B11+B22)
    U33[0,0] = 1; U33[4,0] = 1; U33[8,0] = 1
    V33[0,0] = 1; V33[4,0] = 1; V33[8,0] = 1
    # C00 += 1, C11 += 1, C22 += 1
    W33[0,0] = 1; W33[4,0] = 1; W33[8,0] = 1
    
    # M2 = (A01+A21)(B10+B20)
    U33[1,1] = 1; U33[7,1] = 1
    V33[3,1] = 1; V33[6,1] = 1
    # C10 += 1, C20 += 1?
    # C[i,k].
    # A[0,1]B[1,0] -> C[0,0]. A[0,1]B[2,0] -> C[0,0]? No, C[0,0] is i=0, k=0.
    # A[i,j]B[j,k] -> C[i,k].
    # A[0,1]B[1,0] -> C[0,0].
    # A[0,1]B[2,0] -> C[0,0].
    # A[2,1]B[1,0] -> C[2,0].
    # A[2,1]B[2,0] -> C[2,0].
    # So C00 += 2? No, the sum is over j.
    # M2 = (A01+A21)(B10+B20) = A01B10 + A01B20 + A21B10 + A21B20.
    # Terms:
    # A01B10 -> C00
    # A01B20 -> C00
    # A21B10 -> C20
    # A21B20 -> C20
    # So C00 += 2, C20 += 2.
    # But we need integer coefficients.
    # Laderman uses specific combinations.
    
    # Let's use a known correct Laderman implementation.
    # It's easier to just use the 189 rank solution which is guaranteed correct by construction from Strassen.
    # 189 is score 1.1429.
    # Can we do better?
    # 133 is score 1.624.
    # If I can't get 133 right quickly, 189 is a safe bet.
    # But 189 is already the best known in the inspirations (score 1.1429).
    # Wait, the current program has score 1.0.
    # The inspirations have score 1.1429.
    # So I MUST implement at least 189.
    
    # Let's implement the 189 rank solution (Naive Block Strassen).
    # It's simple and robust.
    
    total_rank = 9 * 7
    U = np.zeros((36, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)
    W = np.zeros((36, total_rank), dtype=int)
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                # A block (bi, bj) * B block (bj, bk) -> contributes to C block (bi, bk)
                for c in range(7):
                    # U part: A block (bi, bj)
                    for ii in range(2):
                        for jj in range(2):
                            u_idx_22 = ii*2+jj
                            val = U22[u_idx_22, c]
                            if val != 0:
                                u_idx_66 = (bi*2+ii)*6 + (bj*2+jj)
                                U[u_idx_66, col] = val
                    # V part: B block (bj, bk)
                    for jj in range(2):
                        for kk in range(2):
                            v_idx_22 = jj*2+kk
                            val = V22[v_idx_22, c]
                            if val != 0:
                                v_idx_66 = (bj*2+jj)*6 + (bk*2+kk)
                                V[v_idx_66, col] = val
                    # W part: C block (bi, bk)
                    for kk in range(2):
                        for ii in range(2):
                            w_idx_22 = kk*2+ii
                            val = W22[w_idx_22, c]
                            if val != 0:
                                w_idx_66 = (bi*2+ii)*6 + (bk*2+kk)
                                W[w_idx_66, col] = val
                    col += 1
    
    # Final verification
    if not verify_decomp(U, V, W, 6, 6, 6):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
