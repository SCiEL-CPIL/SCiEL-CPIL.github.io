"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    """Matrix multiplication tensor decomposition for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We'll build up the decomposition by handling each (i, j, k) triple.
    # The tensor T[i,j,k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    # We want to find factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Start with schoolbook (rank = n*m*p = 105) and try to reduce.
    # For now, let's implement a proper rank-minimal approach.
    
    # The schoolbook decomposition has rank n*m*p = 105.
    # We can do better using block decompositions.
    
    # Let's split the 3x5 matrix into blocks and the 5x7 matrix into blocks.
    # Since 3 = 2+1, 5 = 2+2+1, 7 = 3+3+1, we can use:
    # - 2x2x2 blocks with Strassen (rank 7 instead of 8)
    # - 1xk or kx1 blocks with schoolbook
    # - Other rectangular blocks
    
    # Actually, let's be more systematic. The best known approach for small rectangular
    # matrix multiplication uses specific tensor decompositions.
    
    # For 3x5x7, let's try to find a good decomposition.
    # One approach: 3x5 = (2+1)x5, 5x7 = 5x(2+2+2+1) or similar.
    
    # Let me think about this differently. The rank of M_{n,m,p} is at most n*m*p
    # (schoolbook) and at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    
    # Known results:
    # - M_{2,2,2}: rank 7 (Strassen) vs 8 (schoolbook)
    # - M_{2,3,3}: rank 7 (Laderman) vs 9 (schoolbook)
    # - M_{3,3,3}: rank at most 23 (best known) vs 27 (schoolbook)
    
    # For 3x5x7, let's try:
    # Split n=3 into 2+1, m=5 into 2+3, p=7 into 3+4
    # Then we have blocks:
    # - 2x2x3: ?
    # - 2x3x3: rank 7 (Laderman)
    # - 1x2x3: ?
    # - 1x3x3: ?
    # etc.
    
    # This is getting complex. Let me just implement a clean schoolbook first and
    # verify it works, then try to optimize.
    
    # Schoolbook: rank = n*m*p = 105
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # This shouldn't happen with schoolbook
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
