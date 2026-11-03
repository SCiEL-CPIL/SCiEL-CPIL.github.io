import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition helper
    # Adds 7 columns to U, V, W
    # i0, i1: rows of A
    # j0, j1: cols of A / rows of B
    # k0, k1: cols of B
    def add_strassen(i0, i1, j0, j1, k0, k1, start_col):
        # Strassen terms
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices in local 2x2 block
        # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        
        # U coefficients (for A)
        # V coefficients (for B)
        # W coefficients (for C)
        
        # M1
        u = [1, 0, 0, 1] # a+d
        v = [1, 0, 0, 1] # e+h
        w = [1, 0, 0, -1] # C00+=1, C01+=0, C10+=0, C11+=-1 ? 
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the W vectors for each M
        w1 = [1, 0, 0, -1] # C00=1, C11=-1? No.
        # C00 gets +M1
        # C11 gets +M1
        # So w1 = [1, 0, 0, 1]
        
        w1 = [1, 0, 0, 1]
        w2 = [0, 0, 1, -1] # C10+=M2, C11-=M2
        w3 = [0, 1, 0, 1] # C01+=M3, C11+=M3
        w4 = [1, 0, 1, 0] # C00+=M4, C10+=M4
        w5 = [0, 1, 0, -1] # C01+=M5, C11-=M5
        w6 = [0, 0, 0, 1] # C11+=M6
        w7 = [1, 0, 0, 0] # C00+=M7
        
        u1 = [1, 0, 0, 1] # a+d
        v1 = [1, 0, 0, 1] # e+h
        
        u2 = [0, 0, 1, 1] # c+d
        v2 = [1, 0, 0, 0] # e
        
        u3 = [1, 0, 0, 0] # a
        v3 = [0, 1, 0, -1] # f-h
        
        u4 = [0, 0, 0, 1] # d
        v4 = [0, 0, 1, -1] # g-e
        
        u5 = [1, 1, 0, 0] # a+b
        v5 = [0, 0, 0, 1] # h
        
        u6 = [0, 0, 1, -1] # c-a
        v6 = [1, 1, 0, 0] # e+f
        
        u7 = [0, 1, 0, -1] # b-d
        v7 = [0, 0, 1, 1] # g+h
        
        terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for idx, (u, v, w) in enumerate(terms):
            r = start_col + idx
            # Map u to U indices
            # u[0] -> A[i0, j0] -> U[i0*m + j0]
            # u[1] -> A[i0, j1] -> U[i0*m + j1]
            # u[2] -> A[i1, j0] -> U[i1*m + j0]
            # u[3] -> A[i1, j1] -> U[i1*m + j1]
            
            U[i0*m + j0][r] = u[0]
            U[i0*m + j1][r] = u[1]
            U[i1*m + j0][r] = u[2]
            U[i1*m + j1][r] = u[3]
            
            # Map v to V indices
            # v[0] -> B[j0, k0] -> V[j0*p + k0]
            # v[1] -> B[j0, k1] -> V[j0*p + k1]
            # v[2] -> B[j1, k0] -> V[j1*p + k0]
            # v[3] -> B[j1, k1] -> V[j1*p + k1]
            
            V[j0*p + k0][r] = v[0]
            V[j0*p + k1][r] = v[1]
            V[j1*p + k0][r] = v[2]
            V[j1*p + k1][r] = v[3]
            
            # Map w to W indices
            # w[0] -> C[i0, k0] -> W[k0*n + i0]
            # w[1] -> C[i0, k1] -> W[k1*n + i0]
            # w[2] -> C[i1, k0] -> W[k0*n + i1]
            # w[3] -> C[i1, k1] -> W[k1*n + i1]
            
            W[k0*n + i0][r] = w[0]
            W[k1*n + i0][r] = w[1]
            W[k0*n + i1][r] = w[2]
            W[k1*n + i1][r] = w[3]
            
    # Top 2 rows (i=0,1)
    # Left 2 cols of B (k=0,1) -> C[:, 0:2]
    # A cols 0,1 (j=0,1) and 2,3 (j=2,3)
    
    # Block 1: i=0,1; j=0,1; k=0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 2: i=0,1; j=2,3; k=0,1
    add_strassen(0, 1, 2, 3, 0, 1, 7)
    
    # Right 2 cols of B (k=2,3) -> C[:, 2:4]
    
    # Block 3: i=0,1; j=0,1; k=2,3
    add_strassen(0, 1, 0, 1, 2, 3, 14)
    
    # Block 4: i=0,1; j=2,3; k=2,3
    add_strassen(0, 1, 2, 3, 2, 3, 21)
    
    # Bottom 1 row (i=2)
    # This is 1x4x4. Split into 1x4x2 blocks.
    # 1x2x2 multiplication has rank 4.
    # We need 4 blocks of 1x2x2.
    # Total 16 rank.
    
    # Block 5: i=2; j=0,1; k=0,1
    # Rank 4.
    # A[2, j] B[j, k]
    # Terms:
    # A[2,0]B[0,0], A[2,0]B[0,1], A[2,1]B[1,0], A[2,1]B[1,1]
    # W index: k*n + i
    
    def add_trivial_1x2x2(i, j0, j1, k0, k1, start_col):
        # 4 terms
        for idx in range(4):
            r = start_col + idx
            # j = j0 + (idx % 2)
            # k = k0 + (idx // 2)
            j = j0 + (idx % 2)
            k = k0 + (idx // 2)
            
            U[i*m + j][r] = 1
            V[j*p + k][r] = 1
            W[k*n + i][r] = 1
            
    add_trivial_1x2x2(2, 0, 1, 0, 1, 28)
    add_trivial_1x2x2(2, 2, 3, 0, 1, 32)
    add_trivial_1x2x2(2, 0, 1, 2, 3, 36)
    add_trivial_1x2x2(2, 2, 3, 2, 3, 40)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
