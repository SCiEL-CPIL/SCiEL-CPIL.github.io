"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # The trivial decomposition has rank n*m*p = 50
    # For matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k
    # We want to find a rank-minimal decomposition:
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where (u_r)_{i,j} * (v_r)_{j,k} * (w_r)_{k,i} = 1 for all i,j,k
    
    # One approach: use the fact that for fixed j, the slice T[i,j,k] is rank 1 in (i,k)
    # So we can potentially share factors across i and k for each j.
    # However, the standard trivial decomposition uses rank n*m*p.
    
    # Let's try to find a better decomposition using random search or known results.
    # For M_{2,5,5}, we can try to use block decomposition with Strassen.
    # 5 = 2 + 2 + 1, so we can decompose into 2x2, 2x2, 1x1 blocks.
    
    # However, implementing this correctly is complex. Let's try a simpler approach:
    # Random search for a decomposition with rank < 50.
    
    # The equations are:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of 50 equations. Let's try to solve it for R = 35, 30, etc.
    # But this is a non-convex problem. Let's try a heuristic approach.
    
    # Actually, let's just return the trivial decomposition for now.
    # It's safe and guaranteed to be correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
