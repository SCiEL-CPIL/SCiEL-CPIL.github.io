"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p
    
    # Construct the schoolbook decomposition using Kronecker-product structure
    # U = identity-like selection for A indices (i,j)
    # V = identity-like selection for B indices (j,k)  
    # W = identity-like selection for C indices (i,k) with trace convention W[k*n+i, r]
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (row i, col j of A)
                u_idx = i * m + j
                # V index: j*p + k (row j, col k of B)
                v_idx = j * p + k
                # W index: k*n + i (trace convention: col k, row i of C)
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Verify the decomposition is exact
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        # Also check that no spurious terms exist
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    # Extract indices
                    i_a, j_a = divmod(a, m)
                    j_b, k_b = divmod(b, p)
                    k_c, i_c = divmod(c, n)
                    expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                    if total != expected:
                        return False
        return True
    
    # The schoolbook decomposition is exact by construction, but verify for safety
    if not verify(U, V, W, n, m, p):
        # This should never happen, but return a fallback
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
