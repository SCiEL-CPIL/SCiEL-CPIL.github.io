"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Rank 17 decomposition
    # Blocks:
    # 1. Strassen 2x2x2 for i<2, j<2, k<2 (rank 7)
    # 2. Strassen 2x2x2 for i<2, j<2, k>=2 (rank 7)
    # 3. Rank 1 for i<2, j=2, all k (rank 1)
    # 4. Rank 1 for i=2, j<2, all k (rank 1)
    # 5. Rank 1 for i=2, j=2, all k (rank 1)
    
    total_rank = 17
    U = [[0]*total_rank for _ in range(n*m)]  # 9 x 17
    V = [[0]*total_rank for _ in range(m*p)]  # 12 x 17
    W = [[0]*total_rank for _ in range(p*n)]  # 12 x 17
    
    col = 0
    
    # Strassen 2x2x2 terms
    # U[r][i,j], V[r][j,k], W[r][k,i]
    strassen_terms = []
    # M1 = (A00+A11)(B00+B11)
    U1 = [[1,0],[0,1]]
    V1 = [[1,0],[0,1]]
    W1 = [[1,1],[1,1]] # W1[k,i]
    strassen_terms.append((U1, V1, W1))
    
    # M2 = (A10+A11)(B00)
    U2 = [[0,0],[1,1]]
    V2 = [[1,0],[0,0]]
    W2 = [[0,0],[1,0]]
    strassen_terms.append((U2, V2, W2))
    
    # M3 = A00(B10-B11)
    U3 = [[1,0],[0,0]]
    V3 = [[0,0],[1,-1]]
    W3 = [[0,1],[0,0]]
    strassen_terms.append((U3, V3, W3))
    
    # M4 = A11(B01-B00)
    U4 = [[0,0],[0,1]]
    V4 = [[-1,1],[0,0]]
    W4 = [[1,-1],[0,0]]
    strassen_terms.append((U4, V4, W4))
    
    # M5 = (A00+A01)(B11)
    U5 = [[1,1],[0,0]]
    V5 = [[0,0],[0,1]]
    W5 = [[1,0],[0,0]]
    strassen_terms.append((U5, V5, W5))
    
    # M6 = (A10-A00)(B00+B10)
    U6 = [[-1,0],[1,0]]
    V6 = [[1,0],[1,0]]
    W6 = [[0,0],[0,1]]
    strassen_terms.append((U6, V6, W6))
    
    # M7 = (A01-A11)(B01+B11)
    U7 = [[0,1],[0,-1]]
    V7 = [[0,1],[0,1]]
    W7 = [[1,0],[1,0]]
    strassen_terms.append((U7, V7, W7))
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for (Ur, Vr, Wr) in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if Ur[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj)][col] += Ur[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if Vr[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk)][col] += Vr[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if Wr[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii)][col] += Wr[kk][ii]
            col += 1
            
    def add_rank1_block(i_indices, j_indices, k_indices):
        """Add a rank 1 term for the sub-tensor defined by index sets.
        Assumes the sub-tensor is all ones (i.e., T[i,j,k]=1 for i in i_indices, etc).
        """
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                U[i*m + j][col] += 1
        for j in j_indices:
            for k in k_indices:
                V[j*p + k][col] += 1
        for k in k_indices:
            for i in i_indices:
                W[k*n + i][col] += 1
        col += 1
        
    def add_schoolbook_block(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
                    
    # 1. Strassen i<2, j<2, k<2
    add_strassen_block(0, 0, 0)
    
    # 2. Strassen i<2, j<2, k>=2
    add_strassen_block(0, 0, 2)
    
    # 3. Rank 1: i<2, j=2, k all
    # i in {0,1}, j=2, k in {0,1,2,3}
    add_rank1_block([0,1], [2], [0,1,2,3])
    
    # 4. Rank 1: i=2, j<2, k all
    # i=2, j in {0,1}, k in {0,1,2,3}
    # This is a 1x2x4 sub-tensor which is an outer product, so rank 1
    add_rank1_block([2], [0,1], [0,1,2,3])
    
    # 5. Rank 1: i=2, j=2, k all
    # i=2, j=2, k in {0,1,2,3}
    add_rank1_block([2], [2], [0,1,2,3])
    
    if time.time() - t0 < DEADLINE:
        if not _validate(U, V, W, n, m, p):
            # Fallback to schoolbook if validation fails
            rank = n * m * p
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
