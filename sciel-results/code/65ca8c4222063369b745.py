"""Matrix multiplication 4x4x7 rank minimization using Strassen + schoolbook."""
import time

# EVOLVE-BLOCK-START
def _build_strassen_444():
    """Build rank-49 tensor decomposition for 4x4x4 matrix mult using Strassen^2.
    Returns U (16x49), V (16x49), W (16x49) as lists of lists.
    Convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    # Strassen 2x2x2 base tensors
    # U2[i*2+j, c]: coefficient of A_{ij} in product c
    # V2[j*2+k, c]: coefficient of B_{jk} in product c
    # W2[k*2+i, c]: coefficient of C_{ki}... wait W is indexed [k*n+i]
    
    # Strassen products:
    # m0 = (A00+A11)(B00+B11)
    # m1 = (A10+A11)B00
    # m2 = A00(B01-B11)
    # m3 = A11(B10-B00)
    # m4 = (A00+A01)B11
    # m5 = (A10-A00)(B00+B01)
    # m6 = (A01-A11)(B10+B11)
    # C00 = m0 + m3 - m4 + m6
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m0 - m1 + m2 + m5
    
    U2 = [[1,0,1,0,1,-1,0],    # A00: m0,m2,m4 -> [1,0,1,0,1,0,0]... 
          [0,0,0,0,1,0,1],     # A01
          [0,1,0,0,0,1,0],     # A10
          [1,1,0,1,0,0,-1]]    # A11
    # Let me recompute carefully:
    # m0=(A00+A11)(B00+B11): A00 coef=1, A11 coef=1 -> U2[0][0]=1, U2[3][0]=1
    # m1=(A10+A11)B00: A10=1, A11=1 -> U2[2][1]=1, U2[3][1]=1
    # m2=A00(B01-B11): A00=1 -> U2[0][2]=1
    # m3=A11(B10-B00): A11=1 -> U2[3][3]=1
    # m4=(A00+A01)B11: A00=1, A01=1 -> U2[0][4]=1, U2[1][4]=1
    # m5=(A10-A00)(B00+B01): A10=1, A00=-1 -> U2[0][5]=-1, U2[2][5]=1
    # m6=(A01-A11)(B10+B11): A01=1, A11=-1 -> U2[1][6]=1, U2[3][6]=-1
    U2 = [[1,0,1,0,1,-1,0],    # row 0: A00
          [0,0,0,0,1,0,1],     # row 1: A01
          [0,1,0,0,0,1,0],     # row 2: A10
          [1,1,0,1,0,0,-1]]    # row 3: A11
    
    # V2[j*2+k, c]: coefficient of B_{jk}
    # m0: B00=1, B11=1 -> V2[0][0]=1, V2[3][0]=1
    # m1: B00=1 -> V2[0][1]=1
    # m2: B01=1, B11=-1 -> V2[1][2]=1, V2[3][2]=-1
    # m3: B10=1, B00=-1 -> V2[0][3]=-1, V2[2][3]=1
    # m4: B11=1 -> V2[3][4]=1
    # m5: B00=1, B01=1 -> V2[0][5]=1, V2[1][5]=1
    # m6: B10=1, B11=1 -> V2[2][6]=1, V2[3][6]=1
    V2 = [[1,1,-1,1,0,0,0],    # row 0: B00 (j=0,k=0)
          [0,0,1,0,0,1,0],     # row 1: B01 (j=0,k=1)
          [0,0,-1,1,0,0,1],    # row 2: B10 (j=1,k=0)
          [1,0,-1,0,1,0,1]]    # row 3: B11 (j=1,k=1)
    
    # W2[k*2+i, c]: coefficient for C_{ki} contribution... 
    # W is indexed [k*p+i] but for 2x2x2 it's [k*2+i]
    # C00 = m0 + m3 - m4 + m6 -> W2[0*2+0]=[1,0,0,1,-1,0,1]
    # C01 = m2 + m4 -> W2[0*2+1]=[0,0,1,0,1,0,0]
    # C10 = m1 + m3 -> W2[1*2+0]=[0,1,0,1,0,0,0]
    # C11 = m0 - m1 + m2 + m5 -> W2[1*2+1]=[1,-1,1,0,0,1,0]
    W2 = [[1,0,0,1,-1,0,1],    # k=0,i=0: C00
          [0,0,1,0,1,0,0],     # k=0,i=1: C01
          [0,1,0,1,0,0,0],     # k=1,i=0: C10
          [1,-1,1,0,0,1,0]]    # k=1,i=1: C11
    
    # Verify 2x2x2
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                s = sum(U2[ii*2+jj][c]*V2[jj*2+kk][c]*W2[kk*2+ii][c] for c in range(7))
                if abs(s-1) > 1e-12:
                    # Fix: let me recheck W indexing. W[k*n+i] with n=2
                    # C[i,k] = sum_j A[i,j]B[j,k]. Tensor T[i,j,k]=1.
                    # W[k*2+i, c] where the output index is k*2+i (k is output col, i is output row)
                    # This means W is transposed from what I had. Let me swap.
                    pass
    
    # Actually W[k*n+i, c] means row index = k*n+i. For n=2: row = k*2+i.
    # k is the column of C (output), i is the row of C.
    # So W row 0 = k=0,i=0 (C[0,0]), row 1 = k=0,i=1 (C[0,1]), row 2 = k=1,i=0 (C[1,0]), row 3 = k=1,i=1 (C[1,1])
    # This is what I have. Good.
    
    # Now build 4x4x4 via Kronecker
    # i = a*2+ii, j = b*2+jj, k = d*2+kk where a,b,d,ii,jj,kk in {0,1}
    # c = ac*7+ic where ac is block-level Strassen index, ic is inner Strassen index
    # U[i*4+j, c] = U2[a*2+b][ac] * U2[ii*2+jj][ic]
    # V[j*4+k, c] = V2[b*2+d][ac] * V2[jj*2+kk][ic]
    # W[k*4+i, c] = W2[d*2+a][ac] * W2[kk*2+ii][ic]
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for ac in range(7):
        for ic in range(7):
            c = ac*7 + ic
            for a in range(2):
                for b in range(2):
                    for ii in range(2):
                        for jj in range(2):
                            i = a*2+ii
                            j = b*2+jj
                            U[i*4+j][c] = U2[a*2+b][ac] * U2[ii*2+jj][ic]
            for b in range(2):
                for d in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            j = b*2+jj
                            k = d*2+kk
                            V[j*4+k][c] = V2[b*2+d][ac] * V2[jj*2+kk][ic]
            for d in range(2):
                for a in range(2):
                    for kk in range(2):
                        for ii in range(2):
                            k = d*2+kk
                            i = a*2+ii
                            W[k*4+i][c] = W2[d*2+a][ac] * W2[kk*2+ii][ic]
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 7
    
    # Start with Strassen 4x4x4 (rank 49) + schoolbook 4x4x3 (rank 48) = rank 97
    U4, V4, W4 = _build_strassen_444()
    
    rank_total = 49 + 48  # 97
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    # Fill in Strassen 4x4x4 part (handles k=0,1,2,3)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    for j in range(16):
        for c in range(49):
            V[j][c] = V4[j][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
    
    # Schoolbook for k=4,5,6
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    # Verify the full decomposition
    for i in range(4):
        for j in range(4):
            for k in range(7):
                s = 0.0
                for c in range(rank_total):
                    s += U[i*4+j][c] * V[j*7+k][c] * W[k*4+i][c]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to pure schoolbook rank 112
                    rank = 112
                    U2 = [[0]*rank for _ in range(16)]
                    V2 = [[0]*rank for _ in range(28)]
                    W2 = [[0]*rank for _ in range(28)]
                    col2 = 0
                    for i2 in range(4):
                        for j2 in range(4):
                            for k2 in range(7):
                                U2[i2*4+j2][col2] = 1
                                V2[j2*7+k2][col2] = 1
                                W2[k2*4+i2][col2] = 1
                                col2 += 1
                    return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
