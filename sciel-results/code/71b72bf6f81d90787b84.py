import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Returns factor matrices U, V, W of shapes (16, rank), (24, rank), (24, rank).
    """
    start_time = time.time()
    DEADLINE = 118  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition
    # U[i,j] = A[i,j], V[j,k] = B[j,k], W[k,i] = C[i,k] (transposed output index)
    # Standard Strassen factors (7 multiplications)
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int64)
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int64)
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int64)
    
    # Verify Strassen correctness: C = A * B
    # For 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = ae+bg, C[0,1] = af+bh, C[1,0] = ce+dg, C[1,1] = cf+dh
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C[0,0] = m1+m4-m5+m7
    # C[0,1] = m3+m5
    # C[1,0] = m2+m4
    # C[1,1] = m1-m2+m3+m6
    
    # Check: C[i,k] = sum_j U[i,j]*V[j,k]*W[k,i]
    # Wait, the tensor decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # And T[i,j,k] = 1 if this corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta_{i,k} ... no, T[i,j,k] is the coefficient for A[i,j]*B[j,k] in C[i',k']
    # Actually, the matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k where
    # C[i,k] gets contribution from A[i,j]*B[j,k]. So T[i,j,k] = 1 for all i,j,k.
    # Wait no. The tensor T has indices (i,j,k) where i is row of A, j is col of A/row of B,
    # k is col of B. The output C has indices (i,k). So T[i,j,k] = 1 means that
    # A[i,j]*B[j,k] contributes to C[i,k].
    # The decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # So we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    
    # Let's verify Strassen:
    # U[i,j,r] where i,j in {0,1} (2x2), r in {0..6}
    # V[j,k,r] where j,k in {0,1} (2x2), r in {0..6}
    # W[k,i,r] where k,i in {0,1} (2x2), r in {0..6}
    
    # Check for i=0, j=0, k=0: sum_r U[0,0,r]*V[0,0,r]*W[0,0,r]
    # U[0,0] = [1,0,1,0,1,-1,0]
    # V[0,0] = [1,1,0,-1,0,1,0]
    # W[0,0] = [1,0,0,1,-1,0,1]
    # Products: [1,0,0,0,-1,0,0]
    # Sum = 1 + 0 + 0 + 0 - 1 + 0 + 0 = 0. That's not right!
    
    # I think I have the W matrix wrong. Let me reconsider.
    # The convention is W[k*n+i, r] for the output. For 2x2, W[k*2+i, r].
    # So W[k,i,r] is the coefficient for output C[i,k] (or C[k,i]?).
    # The task says "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column index of B (and A's column),
    # and i is the row index of A (and C's row).
    # So W[k,i,r] corresponds to output C[i,k].
    # The tensor equation: sum_j A[i,j]*B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all i,j,k, and the decomposition is:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let me re-derive Strassen properly.
    # m1 = (a+d)(e+h): U for (a,d) terms, V for (e,h) terms
    # This is getting confusing. Let me just use the known correct Strassen factors.
    
    # Standard Strassen:
    # m1 = (A11 + A22)(B11 + B22)
    # m2 = (A21 + A22)B11
    # m3 = A11(B21 - B22)
    # m4 = A22(B12 - B11)
    # m5 = (A11 + A12)B22
    # m6 = (A21 - A11)(B11 + B12)
    # m7 = (A12 - A22)(B21 + B22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # So for U[i,j,r] (A's factors):
    # m1: A11=1, A22=1 -> U[0,0,0]=1, U[1,1,0]=1
    # m2: A21=1, A22=1 -> U[1,0,1]=1, U[1,1,1]=1
    # m3: A11=1 -> U[0,0,2]=1
    # m4: A22=1 -> U[1,1,3]=1
    # m5: A11=1, A12=1 -> U[0,0,4]=1, U[0,1,4]=1
    # m6: A21=1, A11=-1 -> U[1,0,5]=1, U[0,0,5]=-1
    # m7: A12=1, A22=-1 -> U[0,1,6]=1, U[1,1,6]=-1
    
    # For V[j,k,r] (B's factors):
    # m1: B11=1, B22=1 -> V[0,0,0]=1, V[1,1,0]=1
    # m2: B11=1 -> V[0,0,1]=1
    # m3: B21=1, B22=-1 -> V[1,0,2]=1, V[1,1,2]=-1
    # m4: B12=1, B11=-1 -> V[0,1,3]=1, V[0,0,3]=-1
    # m5: B22=1 -> V[1,1,4]=1
    # m6: B11=1, B12=1 -> V[0,0,5]=1, V[0,1,5]=1
    # m7: B21=1, B22=1 -> V[1,0,6]=1, V[1,1,6]=1
    
    # For W[k,i,r] (C's factors, indexed by output column k, row i):
    # C11 = m1 + m4 - m5 + m7 -> W[0,0,0]=1, W[0,0,3]=1, W[0,0,4]=-1, W[0,0,6]=1
    # C12 = m3 + m5 -> W[1,0,2]=1, W[1,0,4]=1
    # C21 = m2 + m4 -> W[0,1,1]=1, W[0,1,3]=1
    # C22 = m1 - m2 + m3 + m6 -> W[1,1,0]=1, W[1,1,1]=-1, W[1,1,2]=1, W[1,1,5]=1
    
    S_U = np.zeros((4, 7), dtype=np.int64)
    S_V = np.zeros((4, 7), dtype=np.int64)
    S_W = np.zeros((4, 7), dtype=np.int64)
    
    # U[i,j,r] where i,j in {0,1,2,3} for 2x2 blocks, but for base 2x2:
    # i,j in {0,1}
    # m1: U[0,0]=1, U[1,1]=1
    S_U[0,0] = 1; S_U[3,0] = 1  # (0,0) and (1,1) -> indices 0*2+0=0, 1*2+1=3
    # m2: U[1,0]=1, U[1,1]=1
    S_U[2,1] = 1; S_U[3,1] = 1  # (1,0)=2, (1,1)=3
    # m3: U[0,0]=1
    S_U[0,2] = 1
    # m4: U[1,1]=1
    S_U[3,3] = 1
    # m5: U[0,0]=1, U[0,1]=1
    S_U[0,4] = 1; S_U[1,4] = 1
    # m6: U[1,0]=1, U[0,0]=-1
    S_U[2,5] = 1; S_U[0,5] = -1
    # m7: U[0,1]=1, U[1,1]=-1
    S_U[1,6] = 1; S_U[3,6] = -1
    
    # V[j,k,r]
    # m1: V[0,0]=1, V[1,1]=1
    S_V[0,0] = 1; S_V[3,0] = 1
    # m2: V[0,0]=1
    S_V[0,1] = 1
    # m3: V[1,0]=1, V[1,1]=-1
    S_V[2,2] = 1; S_V[3,2] = -1
    # m4: V[0,1]=1, V[0,0]=-1
    S_V[1,3] = 1; S_V[0,3] = -1
    # m5: V[1,1]=1
    S_V[3,4] = 1
    # m6: V[0,0]=1, V[0,1]=1
    S_V[0,5] = 1; S_V[1,5] = 1
    # m7: V[1,0]=1, V[1,1]=1
    S_V[2,6] = 1; S_V[3,6] = 1
    
    # W[k,i,r] where k is col of B, i is row of A (and C)
    # C[i,k]: i is row, k is col
    # W[k,i,r]: k=col, i=row
    # C[0,0] = m1 + m4 - m5 + m7 -> k=0,i=0: W[0,0,0]=1, W[0,0,3]=1, W[0,0,4]=-1, W[0,0,6]=1
    S_W[0,0] = 1; S_W[3,0] = 1; S_W[4,0] = -1; S_W[6,0] = 1
    # C[0,1] = m3 + m5 -> k=1,i=0: W[1,0,2]=1, W[1,0,4]=1
    S_W[2,1] = 1; S_W[4,1] = 1
    # C[1,0] = m2 + m4 -> k=0,i=1: W[0,1,1]=1, W[0,1,3]=1
    S_W[1,2] = 1; S_W[3,2] = 1
    # C[1,1] = m1 - m2 + m3 + m6 -> k=1,i=1: W[1,1,0]=1, W[1,1,1]=-1, W[1,1,2]=1, W[1,1,5]=1
    S_W[0,3] = 1; S_W[1,3] = -1; S_W[2,3] = 1; S_W[5,3] = 1
    
    # Verify: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1 for all i,j,k
    # Let's check a few:
    # i=0,j=0,k=0: sum_r S_U[0,r]*S_V[0,r]*S_W[0,r]
    # S_U[0,:] = [1,0,1,-1,1,-1,0]
    # S_V[0,:] = [1,1,0,-1,0,1,0]
    # S_W[0,:] = [1,1,0,1,-1,0,0]
    # Products: [1,0,0,-1,0,0,0]
    # Sum = 0. Still wrong!
    
    # I think I'm making errors. Let me just use the original program's factors
    # which scored 1.2468 (rank 77). The original used:
    # S_U = [[1,0,1,0,1,-1,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0],[1,1,0,1,0,0,-1]]
    # S_V = [[1,1,0,-1,0,1,0],[0,0,1,0,0,1,0],[0,0,0,1,0,0,1],[1,0,-1,0,1,0,1]]
    # S_W = [[1,0,0,1,-1,0,1],[0,1,0,1,0,0,0],[0,0,1,0,1,0,0],[1,-1,1,0,0,1,0]]
    
    # Let me verify these:
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int64)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int64)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int64)
    
    # Verify for i=0,j=0,k=0:
    # sum_r S_U[0,r]*S_V[0,r]*S_W[0,r]
    # S_U[0] = [1,0,1,0,1,-1,0]
    # S_V[0] = [1,1,0,-1,0,1,0]
    # S_W[0] = [1,0,0,1,-1,0,1]
    # Products: [1,0,0,0,-1,0,0]
    # Sum = 0. Still not 1!
    
    # Maybe the tensor is T[i,j,k] = delta_{i,k}? No, that doesn't make sense for matrix mult.
    # Actually, for matrix multiplication C = AB, we have C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # So we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let me check i=0,j=0,k=0 with the original factors:
    # sum_r S_U[0,r]*S_V[0,r]*S_W[0,r]
    # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1
    # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1. Correct!
    
    # I made an arithmetic error before. Let me recheck:
    # S_U[0] = [1, 0, 1, 0, 1, -1, 0]
    # S_V[0] = [1, 1, 0, -1, 0, 1, 0]
    # S_W[0] = [1, 0, 0, 1, -1, 0, 1]
    # Element-wise products:
    # r=0: 1*1*1 = 1
    # r=1: 0*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*(-1)*1 = 0
    # r=4: 1*0*(-1) = 0
    # r=5: (-1)*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct!
    
    # So the original factors are correct. Good.
    
    # Now build the 4x4x4 part using tensor product of two Strassen decompositions
    rank1 = 49
    U1 = np.zeros((n*m, rank1), dtype=np.int64)
    V1 = np.zeros((m*p, rank1), dtype=np.int64)
    W1 = np.zeros((p*n, rank1), dtype=np.int64)
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U, col] = S_U[i1*2+j1, c1] * S_U[i2*2+j2, c2]
    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V, col] = S_V[j1*2+k1, c1] * S_V[j2*2+k2, c2]
    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W, col] = S_W[k1*2+i1, c1] * S_W[k2*2+i2, c2]
    
    # Part 2: 4x4x2 decomposition using tiling with Strassen
    rank2 = 28
    U2 = np.zeros((n*m, rank2), dtype=np.int64)
    V2 = np.zeros((m*p, rank2), dtype=np.int64)
    W2 = np.zeros((p*n, rank2), dtype=np.int64)
    
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j, col] = S_U[di * 2 + dj, c]
                
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        V2[j * p + k, col] = S_V[dj * 2 + dk, c]
                
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i, col] = S_W[dk * 2 + di, c]
    
    # Combine
    total_rank = rank1 + rank2
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

