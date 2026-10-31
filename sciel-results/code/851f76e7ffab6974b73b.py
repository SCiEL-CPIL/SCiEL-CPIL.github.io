import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49

    # 2x2 Strassen decomposition (rank 7)
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]], C = A*B
    # Flat indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    #                B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
    #                C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11

    # M0 = (A00 + A11) * (B00 + B11) -> C00 += M0, C11 += M0
    # M1 = (A10 + A11) * B00         -> C10 += M1, C11 -= M1
    # M2 = A00 * (B01 - B11)         -> C01 += M2, C11 += M2
    # M3 = A11 * (B10 - B00)         -> C00 += M3, C10 += M3
    # M4 = (A00 + A01) * B11         -> C01 += M4, C00 -= M4
    # M5 = (A10 - A00) * (B00 + B01) -> C11 += M5
    # M6 = (A01 - A11) * (B10 + B11) -> C00 += M6

    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # row 0: A00
        [0, 0, 0, 0, 1,  0, 1],  # row 1: A01
        [0, 1, 0, 0, 0,  1, 0],  # row 2: A10
        [1, 1, 0, 1, 0,  0,-1],  # row 3: A11
    ], dtype=np.int64)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # row 0: B00
        [0, 0, 1, 0, 0, 1, 0],  # row 1: B01
        [0, 0, 0, 1, 0, 0, 1],  # row 2: B10
        [1, 0,-1, 0, 1, 0, 1],  # row 3: B11
    ], dtype=np.int64)

    W2 = np.array([
        [1, 0, 0, 1,-1, 0, 1],  # row 0: C00
        [0, 0, 1, 0, 1, 0, 0],  # row 1: C01
        [0, 1, 0, 0, 0, 0, 0],  # row 2: C10
        [1,-1, 1,-1, 0, 1, 0],  # row 3: C11
    ], dtype=np.int64)

    # Verify 2x2 Strassen: sum_s U2[:,s] @ V2[:,s] outer -> should equal C = A@B for any A,B
    # Equivalent: for all (i,k), sum_s U2[i,s]*V2[k,s] = (A@B)[i,k]
    # Let's verify symbolically by checking the tensor:
    # T2[i,k,s] = U2[i,s]*V2[k,s], and C[i,k] = sum_s W2[i,s]*T2[?,?]... 
    # Actually the decomposition says: C_flat[j*4+i] = sum_r W[j*4+i, r] * (U[:,r] . V[:,r])
    # For 2x2: C[i,j] = sum_s W2[i,s] * (sum_a U2[a,s]*A[a]) * (sum_b V2[b,s]*B[b])
    # which equals sum_{a,b} (sum_s W2[i,s]*U2[a,s]*V2[b,s]) * A[a]*B[b] = sum_{a,b} M[i,a,b]*A[a]*B[b]
    # where M[i,a,b] = (A@B tensor). Let's check.

    # Build the 49-rank Kronecker product decomposition
    # For 4x4x4: split each 4x4 matrix into 2x2 blocks of 2x2 submatrices
    # A[i,j] where i=2*I+ii, j=2*J+jj
    # The Kronecker product approach:
    # U[i*4+j, s*7+t] = U2[I*2+J, s] * U2[ii*2+jj, t]
    # V[k*4+l, s*7+t] = V2[K*2+L, s] * V2[kk*2+ll, t]
    # W[j*4+i, s*7+t] = W2[I*2+J, s] * W2[ii*2+jj, t]
    # where for W, the output index is j*4+i (row j, col i) following trace convention W[k*n+i,r]

    # Vectorized construction
    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)

    # Index arrays for 4x4 flattened indices
    rows = np.arange(16)
    # For U: flat index = i*4+j, block (I,J) = (i//2, j//2), inner (ii,jj) = (i%2, j%2)
    # U2 index for block: I*2+J, for inner: ii*2+jj

    for r in range(49):
        s = r // 7
        t = r % 7

        # U contributions
        for i in range(4):
            for j in range(4):
                I = i // 2
                J = j // 2
                ii = i % 2
                jj = j % 2
                U[i*4+j, r] = U2[I*2+J, s] * U2[ii*2+jj, t]

        # V contributions
        for k in range(4):
            for l in range(4):
                K = k // 2
                L = l // 2
                kk = k % 2
                ll = l % 2
                V[k*4+l, r] = V2[K*2+L, s] * V2[kk*2+ll, t]

        # W contributions: W[j*4+i, r], output C[j,i]
        # Block structure: I = i//2 (row block of C), J = j//2 (col block of C)
        # Inner: ii = i%2, jj = j%2
        # W2 index: W2[I*2+J, s] * W2[ii*2+jj, t]
        for j in range(4):
            for i in range(4):
                I = i // 2
                J = j // 2
                ii = i % 2
                jj = j % 2
                W[j*4+i, r] = W2[I*2+J, s] * W2[ii*2+jj, t]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
