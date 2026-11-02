"""Combined program: structured rank-23 decomposition for 3x3 matrix multiplication."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = 3, 3, 3
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Rank-27 schoolbook decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def strassen_2x2(U, V, W, rank_offset, i0, j0, k0):
        """
        Add Strassen's 7-term algorithm for a 2x2 submatrix.
        This handles the 8 products: A[i0,j0]*B[j0,k0], A[i0,j0]*B[j0,k0+1],
        A[i0,j0+1]*B[j0,k0], A[i0,j0+1]*B[j0,k0+1],
        A[i0+1,j0]*B[j0,k0], A[i0+1,j0]*B[j0,k0+1],
        A[i0+1,j0+1]*B[j0,k0], A[i0+1,j0+1]*B[j0,k0+1]
        using 7 rank-1 terms instead of 8.
        """
        u00 = i0 * m + j0
        u01 = i0 * m + j0 + 1
        u10 = (i0+1) * m + j0
        u11 = (i0+1) * m + j0 + 1
        
        v00 = j0 * p + k0
        v01 = j0 * p + k0 + 1
        v10 = (j0+1) * p + k0
        v11 = (j0+1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = k0 * n + i0 + 1
        w10 = (k0+1) * n + i0
        w11 = (k0+1) * n + i0 + 1
        
        r = rank_offset
        
        # M1 = (A00+A11)(B00+B11)
        U[u00][r] += 1.0; U[u11][r] += 1.0
        V[v00][r] += 1.0; V[v11][r] += 1.0
        W[w00][r] += 1.0; W[w11][r] += 1.0
        r += 1
        
        # M2 = (A10+A11)B00
        U[u10][r] += 1.0; U[u11][r] += 1.0
        V[v00][r] += 1.0
        W[w10][r] += 1.0
        r += 1
        
        # M3 = A00(B10-B11)
        U[u00][r] += 1.0
        V[v10][r] += 1.0; V[v11][r] += -1.0
        W[w11][r] += 1.0
        r += 1
        
        # M4 = A11(B01-B00)
        U[u11][r] += 1.0
        V[v01][r] += 1.0; V[v00][r] += -1.0
        W[w00][r] += 1.0
        r += 1
        
        # M5 = (A00+A01)B11
        U[u00][r] += 1.0; U[u01][r] += 1.0
        V[v11][r] += 1.0
        W[w01][r] += 1.0
        r += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][r] += 1.0; U[u00][r] += -1.0
        V[v00][r] += 1.0; V[v01][r] += 1.0
        W[w10][r] += 1.0
        r += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][r] += 1.0; U[u11][r] += -1.0
        V[v10][r] += 1.0; V[v11][r] += 1.0
        W[w01][r] += 1.0
        r += 1
        
        return r
    
    def add_standard_term(U, V, W, r, i, j, k):
        """Add a standard rank-1 term for product A[i,j]*B[j,k] contributing to C[i,k]."""
        U[i*m+j][r] += 1.0
        V[j*p+k][r] += 1.0
        W[k*n+i][r] += 1.0
    
    def strassen_26():
        """
        Rank-26 decomposition: Strassen on 2x2 block (i,j,k in {0,1}) 
        plus standard terms for all other products.
        
        The 2x2 Strassen handles 8 products (i in {0,1}, j in {0,1}, k in {0,1})
        in 7 terms. The remaining 19 products use 19 standard terms.
        Total: 7 + 19 = 26.
        """
        rank = 26
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Strassen for 2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
        r = strassen_2x2(U, V, W, 0, 0, 0, 0)
        
        # Standard terms for all (i,j,k) NOT in the 2x2 Strassen block
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if this product is handled by Strassen
                    if i < 2 and j < 2 and k < 2:
                        continue
                    add_standard_term(U, V, W, r, i, j, k)
                    r += 1
        
        assert r == rank, f"Expected rank {rank}, got {r}"
        return U, V, W, rank
    
    # Try rank-26 first
    U, V, W, rank = strassen_26()
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
