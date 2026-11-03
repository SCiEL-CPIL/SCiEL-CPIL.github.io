import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 124
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the sets
    S1 = {0, 1}
    S2 = {2, 3, 4}
    
    # Helper to get global indices
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    col = 0
    
    # 1. Add Strassen terms for the 2x2x2 block (indices in S1)
    # The 7 Strassen vectors derived above
    
    # M1
    U[u_idx(0,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1
    W[w_idx(1,1)][col] = 1
    col += 1
    
    # M2
    U[u_idx(1,0)][col] = 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,1)][col] = 1  # C_10 (i=1, k=0) -> w_idx(0,1)
    W[w_idx(1,1)][col] = -1 # C_11 (i=1, k=1) -> w_idx(1,1)
    col += 1
    
    # M3
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = -1
    W[w_idx(1,0)][col] = 1  # C_01 (i=0, k=1) -> w_idx(1,0)
    W[w_idx(1,1)][col] = 1  # C_11 (i=1, k=1) -> w_idx(1,1)
    col += 1
    
    # M4
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,0)][col] = 1
    V[v_idx(0,0)][col] = -1
    W[w_idx(0,0)][col] = 1  # C_00
    W[w_idx(0,1)][col] = 1  # C_10
    col += 1
    
    # M5
    U[u_idx(0,0)][col] = 1
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = -1 # C_00
    W[w_idx(1,0)][col] = 1  # C_01
    col += 1
    
    # M6
    U[u_idx(1,0)][col] = 1
    U[u_idx(0,0)][col] = -1
    V[v_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1
    W[w_idx(1,1)][col] = 1  # C_11
    col += 1
    
    # M7
    U[u_idx(0,1)][col] = 1
    U[u_idx(1,1)][col] = -1
    V[v_idx(1,0)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1  # C_00
    col += 1
    
    # 2. Add standard rank-1 terms for all other triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block handled by Strassen
                if i in S1 and j in S1 and k in S1:
                    continue
                
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(k, i)][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
