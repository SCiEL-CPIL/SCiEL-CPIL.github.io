"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
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
    
    # Verify the decomposition is correct
    # The tensor T[a, b, c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes (i,j) as i*m+j, b indexes (j,k) as j*p+k, c indexes (k,i) as k*n+i
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check a few entries to verify correctness
    # T[i*m+j, j*p+k, k*n+i] should be 1, all other entries should be 0
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute T[a,b,c] from the decomposition
                val = 0.0
                for r in range(rank):
                    val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                if abs(val - 1.0) > 1e-9:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    # If verification fails, return the schoolbook anyway (it should always be correct)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
