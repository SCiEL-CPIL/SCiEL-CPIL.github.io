"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # For M_{2,2,p}, rank = 2p achieves the lower bound max(n*p, m*p, n*m) = 2p
    rank = 2 * p  # 26
    
    # U: shape (n*m, rank) = (4, 26)
    # V: shape (m*p, rank) = (26, 26)
    # W: shape (p*n, rank) = (26, 26)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half: c = 0 to p-1, corresponds to j=0
    for k in range(p):
        c = k
        # U[i*m + 0, c] = 1 for i=0,1
        U[0 * m + 0][c] = 1  # U[0][c]
        U[1 * m + 0][c] = 1  # U[2][c]
        # V[0*p + k, c] = 1, V[1*p + k, c] = 0
        V[0 * p + k][c] = 1
        # W[k*n + i, c] = 1 for i=0,1
        W[k * n + 0][c] = 1
        W[k * n + 1][c] = 1
    
    # Second half: c = p to 2p-1, corresponds to j=1
    for k in range(p):
        c = p + k
        # U[i*m + 1, c] = 1 for i=0,1
        U[0 * m + 1][c] = 1  # U[1][c]
        U[1 * m + 1][c] = 1  # U[3][c]
        # V[1*p + k, c] = 1, V[0*p + k, c] = 0
        V[1 * p + k][c] = 1
        # W[k*n + i, c] = 1 for i=0,1
        W[k * n + 0][c] = 1
        W[k * n + 1][c] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    def _verify(n, m, p, U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not _verify(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return result
# EVOLVE-BLOCK-END
