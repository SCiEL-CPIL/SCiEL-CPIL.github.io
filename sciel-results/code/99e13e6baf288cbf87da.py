"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    # Schoolbook decomposition of rank n*m*p
    # For 2x3x7, rank is 42.
    # Note: While lower ranks may exist (e.g., via 2x3x3 rank-16 blocks),
    # exact coefficients are not trivial to generate on-the-fly.
    # We provide the guaranteed valid schoolbook decomposition.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, col j -> i*m + j
                # V index: row j of B, col k -> j*p + k
                # W index: row k of C, col i -> k*n + i (Trace convention W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1
                
                column += 1
                
                # Safety check for time
                if time.time() - start_time > 100:
                    return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
