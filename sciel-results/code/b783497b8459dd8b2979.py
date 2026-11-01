"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Rank 7 for Strassen on first 2 cols, Rank 28 for Schoolbook on remaining 7 cols
    rank_strassen = 7
    rank_schoolbook = 4 * (p - 2)
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # shape (4, rank)
    V = [[0.0] * total_rank for _ in range(m * p)]  # shape (18, rank)
    W = [[0.0] * total_rank for _ in range(p * n)]  # shape (18, rank)
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # --- Strassen Part (Cols 0, 1) ---
    # Indices for A (2x2): i*2 + j
    # Indices for B (2x2 part of 2x9): j*9 + k (k in 0,1)
    # Indices for C (2x2 part of 2x9): k*2 + i (k in 0,1)
    
    # M1 = (a+d)(e+h) -> c00=1, c10=0, c01=1, c11=1
    col = 0
    add_term(0, 1, 0, 1, 0, 1, col) # a*e
    add_term(3, 1, 0, 1, 0, 1, col) # d*e
    add_term(0, 1, 10, 1, 0, 1, col) # a*h
    add_term(3, 1, 10, 1, 0, 1, col) # d*h
    # Wait, Strassen combines these into one factor per product.
    # My add_term logic above was for schoolbook.
    # For Strassen, each factor r has multiple non-zero entries in U, V, W.
    
    # Let's reset and do it properly.
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Strassen Factors
    # F0: (a+d)(e+h)
    U[0][0] += 1.0; U[3][0] += 1.0
    V[0][0] += 1.0; V[10][0] += 1.0
    W[0][0] += 1.0 # c00
    W[1][0] += 1.0 # c01 (k=0, i=1)
    W[3][0] += 1.0 # c11 (k=1, i=1)
    
    # F1: (c+d)e
    U[2][1] += 1.0; U[3][1] += 1.0
    V[0][1] += 1.0
    W[2][1] += 1.0 # c10 (k=1, i=0)
    W[3][1] += -1.0 # c11 (k=1, i=1)
    
    # F2: a(f-h)
    U[0][2] += 1.0
    V[1][2] += 1.0; V[10][2] += -1.0
    W[1][2] += -1.0 # c01
    W[3][2] += 1.0 # c11
    
    # F3: d(g-e)
    U[3][3] += 1.0
    V[9][3] += 1.0; V[0][3] += -1.0
    W[0][3] += 1.0 # c00
    W[2][3] += 1.0 # c10
    
    # F4: (a+b)h
    U[0][4] += 1.0; U[1][4] += 1.0
    V[10][4] += 1.0
    W[0][4] += -1.0 # c00
    
    # F5: (c-a)(e+f)
    U[2][5] += 1.0; U[0][5] += -1.0
    V[0][5] += 1.0; V[1][5] += 1.0
    W[3][5] += 1.0 # c11
    
    # F6: (b-d)(g+h)
    U[1][6] += 1.0; U[3][6] += -1.0
    V[9][6] += 1.0; V[10][6] += 1.0
    W[0][6] += 1.0 # c00
    
    # --- Schoolbook Part (Cols 2..8) ---
    col = 7
    for k in range(2, p):
        for i in range(n):
            for j in range(m):
                # Term: A[i,j] * B[j,k] -> C[k,i]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] += 1.0
                V[v_idx][col] += 1.0
                W[w_idx][col] += 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
