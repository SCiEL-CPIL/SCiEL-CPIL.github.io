"""Matrix multiplication 3x4 x 4x4 with Winograd-accelerated tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    terms = []  # list of (U_vec, V_vec, W_vec)
    
    def winograd_block(i0, i1, j0, j1, k0, k1):
        """Add Winograd terms for A rows i0,i1; B rows j0,j1; C cols k0,k1"""
        # p1 = (A[i0,j0]+A[i1,j0]) * (B[j0,k0]+B[j1,k0])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i0*m+j0] = 1; U[i1*m+j0] = 1
        V[j0*p+k0] = 1; V[j1*p+k0] = 1
        W[k0*n+i0] = 1; W[k1*n+i0] = 1; W[k0*n+i1] = 1; W[k1*n+i1] = 1
        terms.append((U[:], V[:], W[:]))
        
        # p2 = A[i0,j0] * (B[j0,k0]-B[j0,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i0*m+j0] = 1
        V[j0*p+k0] = 1; V[j0*p+k1] = -1
        W[k0*n+i0] = 1; W[k1*n+i0] = -1
        terms.append((U[:], V[:], W[:]))
        
        # p3 = A[i1,j0] * (B[j1,k0]-B[j1,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i1*m+j0] = 1
        V[j1*p+k0] = 1; V[j1*p+k1] = -1
        W[k1*n+i0] = 1; W[k0*n+i1] = -1
        terms.append((U[:], V[:], W[:]))
        
        # p4 = A[i1,j1] * (B[j0,k1]+B[j1,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i1*m+j1] = 1
        V[j0*p+k1] = 1; V[j1*p+k1] = 1
        W[k1*n+i1] = 1
        terms.append((U[:], V[:], W[:]))
        
        # p5 = (A[i0,j0]+A[i0,j1]) * (B[j0,k0]+B[j0,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i0*m+j0] = 1; U[i0*m+j1] = 1
        V[j0*p+k0] = 1; V[j0*p+k1] = 1
        W[k0*n+i0] = -1; W[k1*n+i0] = 1; W[k0*n+i1] = 1; W[k1*n+i1] = -1
        terms.append((U[:], V[:], W[:]))
        
        # p6 = (A[i1,j0]+A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i1*m+j0] = 1; U[i1*m+j1] = 1
        V[j1*p+k0] = 1; V[j1*p+k1] = 1
        W[k0*n+i1] = 1; W[k1*n+i1] = 1
        terms.append((U[:], V[:], W[:]))
        
        # p7 = (A[i0,j1]-A[i1,j1]) * (B[j0,k1]-B[j1,k1])
        U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
        U[i0*m+j1] = 1; U[i1*m+j1] = -1
        V[j0*p+k1] = 1; V[j1*p+k1] = -1
        W[k0*n+i0] = 1; W[k1*n+i0] = -1; W[k0*n+i1] = -1; W[k1*n+i1] = 1
        terms.append((U[:], V[:], W[:]))
    
    # Apply Winograd to 4 blocks for rows 0-1
    winograd_block(0, 1, 0, 1, 0, 1)
    winograd_block(0, 1, 0, 1, 2, 3)
    winograd_block(0, 1, 2, 3, 0, 1)
    winograd_block(0, 1, 2, 3, 2, 3)
    
    # Row 2: schoolbook
    for j in range(m):
        for k in range(p):
            U = [0]*(n*m); V = [0]*(m*p); W = [0]*(p*n)
            U[2*m+j] = 1
            V[j*p+k] = 1
            W[k*n+2] = 1
            terms.append((U[:], V[:], W[:]))
    
    rank = len(terms)
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for col, (U, V, W) in enumerate(terms):
        for i in range(n*m):
            U_mat[i][col] = U[i]
        for i in range(m*p):
            V_mat[i][col] = V[i]
        for i in range(p*n):
            W_mat[i][col] = W[i]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
