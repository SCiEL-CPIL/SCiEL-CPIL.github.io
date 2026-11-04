"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposition of (3,4,8) matrix multiplication.
    Rank: 88 via splitting shared dim 4 into 2+2, using Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Structure: 2 j-halves, each with:
    #   2 k-halves, each with 2 Strassen blocks (7 terms each) = 28
    #   1x2x8 part: 16 terms
    # Total per j-half: 28 + 16 = 44
    # Total: 88
    rank_total = 88
    
    U = [[0.0] * rank_total for _ in range(n * m)]  # 12 x 88
    V = [[0.0] * rank_total for _ in range(m * p)]  # 32 x 88
    W = [[0.0] * rank_total for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    def add_term(u, v, w):
        nonlocal col
        for i in range(n * m):
            U[i][col] = u[i]
        for i in range(m * p):
            V[i][col] = v[i]
        for i in range(p * n):
            W[i][col] = w[i]
        col += 1
    
    def strassen_2x2x2(a00, a01, a10, a11, br0, br1, k0, k1, cr0, cr1):
        """Add 7 Strassen terms for (2,2,2) sub-problem.
        a* are U indices (i*m+j), br* are B row indices, k* are B col indices,
        cr* are C row indices.
        """
        nonlocal col
        # m1 = (a00+a11)(b_r0_k0 + b_r1_k1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a00] = 1.0; u[a11] = 1.0
        v[br0*p+k0] = 1.0; v[br1*p+k1] = 1.0
        w[k0*n+cr0] = 1.0; w[k1*n+cr1] = 1.0
        add_term(u, v, w)
        
        # m2 = (a10+a11)(b_r0_k1 + b_r1_k1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a10] = 1.0; u[a11] = 1.0
        v[br0*p+k1] = 1.0; v[br1*p+k1] = 1.0
        w[k0*n+cr1] = 1.0; w[k1*n+cr1] = -1.0
        add_term(u, v, w)
        
        # m3 = a00(b_r0_k1 - b_r1_k1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a00] = 1.0
        v[br0*p+k1] = 1.0; v[br1*p+k1] = -1.0
        w[k1*n+cr0] = 1.0
        add_term(u, v, w)
        
        # m4 = a11(b_r0_k0 - b_r1_k0)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a11] = 1.0
        v[br0*p+k0] = 1.0; v[br1*p+k0] = -1.0
        w[k0*n+cr0] = -1.0; w[k1*n+cr1] = 1.0
        add_term(u, v, w)
        
        # m5 = (a00+a01) * b_r0_k0
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a00] = 1.0; u[a01] = 1.0
        v[br0*p+k0] = 1.0
        w[k0*n+cr0] = 1.0; w[k0*n+cr1] = -1.0
        add_term(u, v, w)
        
        # m6 = (a10+a11) * b_r1_k1
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a10] = 1.0; u[a11] = 1.0
        v[br1*p+k1] = 1.0
        w[k0*n+cr1] = 1.0; w[k1*n+cr1] = -1.0
        add_term(u, v, w)
        
        # m7 = (a00-a01) * b_r1_k0
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[a00] = 1.0; u[a01] = -1.0
        v[br1*p+k0] = 1.0
        w[k0*n+cr0] = -1.0; w[k1*n+cr0] = 1.0
        add_term(u, v, w)
    
    # Build decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into [0,1] and [2,3]
    for jh in range(2):
        j0 = jh * 2
        j1 = jh * 2 + 1
        
        # Rows 0,1 of A: (2,2,8) problem
        # Split k into [0:4] and [4:8]
        for kh in range(2):
            k_start = kh * 4
            # Strassen block 1: cols k_start, k_start+1
            strassen_2x2x2(
                a00=0*m+j0, a01=0*m+j1,
                a10=1*m+j0, a11=1*m+j1,
                br0=j0, br1=j1,
                k0=k_start, k1=k_start+1,
                cr0=0, cr1=1
            )
            # Strassen block 2: cols k_start+2, k_start+3
            strassen_2x2x2(
                a00=0*m+j0, a01=0*m+j1,
                a10=1*m+j0, a11=1*m+j1,
                br0=j0, br1=j1,
                k0=k_start+2, k1=k_start+3,
                cr0=0, cr1=1
            )
        
        # Row 2 of A: (1,2,8) problem, rank 16
        for k in range(8):
            u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
            u[2*m+j0] = 1.0
            v[j0*p+k] = 1.0
            w[k*n+2] = 1.0
            add_term(u, v, w)
            
            u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
            u[2*m+j1] = 1.0
            v[j1*p+k] = 1.0
            w[k*n+2] = 1.0
            add_term(u, v, w)
    
    assert col == rank_total, f"Expected {rank_total}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
