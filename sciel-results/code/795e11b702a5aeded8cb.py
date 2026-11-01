"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 30)
    V = [[0] * rank for _ in range(m * p)]  # shape (30, 30)
    W = [[0] * rank for _ in range(p * n)]  # shape (20, 30)
    
    for jp in range(m):  # j' in 0,1,2
        for kp in range(p):  # k' in 0..9
            c = jp * p + kp
            # U[i*m+j, c] = 1 if j == jp, else 0
            for i in range(n):
                U[i * m + jp][c] = 1
            # V[j*p+k, c] = 1 if j==jp and k==kp, else 0
            V[jp * p + kp][c] = 1
            # W[k*n+i, c] = 1 for all k, i
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
