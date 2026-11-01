import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions are 2, 5, 5
    # Split m=5 into 2, 2, 1
    # Split p=5 into 2, 2, 1
    # Split n=2 into 2 (no split needed, or just one block of size 2)
    
    # We will sum the decompositions of the blocks.
    # Block (bi, bj, bk) where bi in [0], bj in [0,1,2], bk in [0,1,2]
    # bi=0 corresponds to n-size 2
    # bj=0,1 correspond to m-size 2, bj=2 corresponds to m-size 1
    # bk=0,1 correspond to p-size 2, bk=2 corresponds to p-size 1
    
    # Global U, V, W
    # We don't know the final rank in advance, so we use lists of lists or numpy arrays.
    # Let's use lists of lists for U, V, W and append columns.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block_strassen(i0, j0, k0):
        # Adds 7 columns for a 2x2x2 block starting at i0, j0, k0
        # Local indices 0,1 for i, j, k
        # Global U index: i*m + j
        # Global V index: j*p + k
        # Global W index: k*n + i
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = [0]*(n*m)
        v1 = [0]*(m*p)
        w1 = [0]*(p*n)
        u1[i0*m + j0] = 1
        u1[(i0+1)*m + (j0+1)] = 1
        v1[j0*p + k0] = 1
        v1[(j0+1)*p + (k0+1)] = 1
        w1[k0*n + i0] = 1
        w1[(k0+1)*n + (i0+1)] = 1
        U_cols.append(u1)
        V_cols.append(v1)
        W_cols.append(w1)
        
        # Term 2: M2 = (A10+A11)B00
        u2 = [0]*(n*m)
        v2 = [0]*(m*p)
        w2 = [0]*(p*n)
        u2[(i0+1)*m + j0] = 1
        u2[(i0+1)*m + (j0+1)] = 1
        v2[j0*p + k0] = 1
        w2[(k0+1)*n + (i0+1)] = 1 # C10
        U_cols.append(u2)
        V_cols.append(v2)
        W_cols.append(w2)
        
        # Term 3: M3 = A00(B01-B11)
        u3 = [0]*(n*m)
        v3 = [0]*(m*p)
        w3 = [0]*(p*n)
        u3[i0*m + j0] = 1
        v3[j0*p + (k0+1)] = 1
        v3[(j0+1)*p + (k0+1)] = -1
        w3[k0*n + i0] = 1 # C01? No, C01 is k=1, i=0.
        # W index: k*n + i.
        # C01: k=1, i=0 -> 1*n + 0 = n.
        # Wait, W index is k*n + i.
        # For C01, k=1, i=0. Index = 1*n + 0 = n.
        # For C00, k=0, i=0. Index = 0.
        # Let's re-verify W indices for Strassen.
        # C00: k=0, i=0.
        # C01: k=1, i=0.
        # C10: k=0, i=1.
        # C11: k=1, i=1.
        
        # Recalculating W for Term 3:
        # M3 contributes to C01.
        # C01 is at k=1, i=0.
        # Global W index: (k0+1)*n + i0.
        w3 = [0]*(p*n)
        w3[(k0+1)*n + i0] = 1
        U_cols.append(u3)
        V_cols.append(v3)
        W_cols.append(w3)
        
        # Term 4: M4 = A11(B10-B00)
        u4 = [0]*(n*m)
        v4 = [0]*(m*p)
        w4 = [0]*(p*n)
        u4[(i0+1)*m + (j0+1)] = 1
        v4[(j0+1)*p + k0] = 1
        v4[j0*p + k0] = -1
        # M4 contributes to C00 and C10.
        # C00: k=0, i=0. Index k0*n + i0.
        # C10: k=0, i=1. Index k0*n + (i0+1).
        w4[k0*n + i0] = 1
        w4[k0*n + (i0+1)] = 1
        U_cols.append(u4)
        V_cols.append(v4)
        W_cols.append(w4)
        
        # Term 5: M5 = (A00+A01)B11
        u5 = [0]*(n*m)
        v5 = [0]*(m*p)
        w5 = [0]*(p*n)
        u5[i0*m + j0] = 1
        u5[i0*m + (j0+1)] = 1
        v5[(j0+1)*p + (k0+1)] = 1
        # M5 contributes to C00 (-1) and C01 (+1).
        # C00: k=0, i=0.
        # C01: k=1, i=0.
        w5[k0*n + i0] = -1
        w5[(k0+1)*n + i0] = 1
        U_cols.append(u5)
        V_cols.append(v5)
        W_cols.append(w5)
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = [0]*(n*m)
        v6 = [0]*(m*p)
        w6 = [0]*(p*n)
        u6[(i0+1)*m + j0] = 1
        u6[i0*m + j0] = -1
        v6[j0*p + k0] = 1
        v6[j0*p + (k0+1)] = 1
        # M6 contributes to C11 (+1).
        # C11: k=1, i=1.
        w6[(k0+1)*n + (i0+1)] = 1
        U_cols.append(u6)
        V_cols.append(v6)
        W_cols.append(w6)
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = [0]*(n*m)
        v7 = [0]*(m*p)
        w7 = [0]*(p*n)
        u7[i0*m + (j0+1)] = 1
        u7[(i0+1)*m + (j0+1)] = -1
        v7[(j0+1)*p + k0] = 1
        v7[(j0+1)*p + (k0+1)] = 1
        # M7 contributes to C00 (+1).
        # C00: k=0, i=0.
        w7[k0*n + i0] = 1
        U_cols.append(u7)
        V_cols.append(v7)
        W_cols.append(w7)

    def add_block_schoolbook(i0, j0, k0, ni, nj, nk):
        # Adds ni*nj*nk columns for a ni x nj x nk block
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    u = [0]*(n*m)
                    v = [0]*(m*p)
                    w = [0]*(p*n)
                    i = i0 + ii
                    j = j0 + jj
                    k = k0 + kk
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    # Define block starts and sizes
    # m splits: 2, 2, 1
    # p splits: 2, 2, 1
    # n splits: 2
    
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    n_splits = [2]
    
    j0 = 0
    p0 = 0 # k0
    for bj, mj in enumerate(m_splits):
        k0 = 0
        for bk, pk in enumerate(p_splits):
            for bi, ni in enumerate(n_splits):
                if mj == 2 and pk == 2:
                    add_block_strassen(bi*ni, j0, k0)
                else:
                    add_block_schoolbook(bi*ni, j0, k0, ni, mj, pk)
            k0 += pk
        j0 += mj

    # Convert lists of columns to matrices U, V, W
    # U is (n*m) x rank
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n*m):
            U[i][c] = U_cols[c][i]
        for i in range(m*p):
            V[i][c] = V_cols[c][i]
        for i in range(p*n):
            W[i][c] = W_cols[c][i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
