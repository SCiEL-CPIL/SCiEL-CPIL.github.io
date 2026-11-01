import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix.
    
    The decomposition uses 7 independent 2x2*2x2 Strassen blocks,
    each contributing rank 7, for a total rank of 49.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Matrix shapes:
    # U: (n*m, rank) = (4, 49) - factors for A
    # V: (m*p, rank) = (28, 49) - factors for B  
    # W: (p*n, rank) = (28, 49) - factors for C
    
    num_blocks = p // 2  # 7 blocks of size 2x2
    rank_per_block = 7   # Strassen rank for 2x2*2x2
    total_rank = num_blocks * rank_per_block  # 49
    
    # Initialize factor matrices as lists of lists
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # U index for A[i,j] is i*m + j
    u_a = 0 * m + 0  # A[0,0]
    u_b = 0 * m + 1  # A[0,1]
    u_c = 1 * m + 0  # A[1,0]
    u_d = 1 * m + 1  # A[1,1]
    
    col = 0
    for b in range(num_blocks):
        # V indices for B block b (columns 2b, 2b+1)
        v_e = 0 * p + (2 * b)       # B[0, 2b]
        v_f = 0 * p + (2 * b + 1)   # B[0, 2b+1]
        v_g = 1 * p + (2 * b)       # B[1, 2b]
        v_h = 1 * p + (2 * b + 1)   # B[1, 2b+1]
        
        # W indices for C block b (columns 2b, 2b+1)
        # W index for C[i, k] is k*n + i
        w_11 = (2 * b) * n + 0      # C[0, 2b]
        w_21 = (2 * b) * n + 1      # C[1, 2b]
        w_12 = (2 * b + 1) * n + 0  # C[0, 2b+1]
        w_22 = (2 * b + 1) * n + 1  # C[1, 2b+1]
        
        # Strassen term 1: m1 = (a+d)(e+h)
        # Contributes to C[0,0] and C[1,1]
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # Strassen term 2: m2 = (c+d)e
        # Contributes to C[1,0] and C[1,1]
        U[u_c][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        W[w_21][col] += 1
        W[w_22][col] += -1
        col += 1
        
        # Strassen term 3: m3 = a(f-h)
        # Contributes to C[0,1] and C[1,1]
        U[u_a][col] += 1
        V[v_f][col] += 1
        V[v_h][col] += -1
        W[w_12][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # Strassen term 4: m4 = d(g-e)
        # Contributes to C[0,0] and C[1,0]
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_e][col] += -1
        W[w_11][col] += 1
        W[w_21][col] += 1
        col += 1
        
        # Strassen term 5: m5 = (a+b)h
        # Contributes to C[0,0] and C[0,1]
        U[u_a][col] += 1
        U[u_b][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += -1
        W[w_12][col] += 1
        col += 1
        
        # Strassen term 6: m6 = (c-a)(e+f)
        # Contributes to C[1,1]
        U[u_c][col] += 1
        U[u_a][col] += -1
        V[v_e][col] += 1
        V[v_f][col] += 1
        W[w_22][col] += 1
        col += 1
        
        # Strassen term 7: m7 = (b-d)(g+h)
        # Contributes to C[0,0]
        U[u_b][col] += 1
        U[u_d][col] += -1
        V[v_g][col] += 1
        V[v_h][col] += 1
        W[w_11][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
