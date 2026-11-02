"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    def verify(U, V, W, tol=1e-9):
        """Verify: sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k*n+i,r] = delta(j,j') for all i,j,j',k"""
        rank = len(U[0])
        U_arr = np.array(U)  # (nm, rank)
        V_arr = np.array(V)  # (mp, rank)
        W_arr = np.array(W)  # (pn, rank)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for r in range(rank):
                            s += U_arr[i*m+j, r] * V_arr[jp*p+k, r] * W_arr[k*n+i, r]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    # Construction: rank = m = 3
    # U[i*m+j, r] = 1 if j==r and r<m, else 0
    # V[j*p+k, r] = 1 if j==r and r<m, else 0
    # W[k*n+i, r] = 1 for r<m
    rank = m  # 3
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if j == r:
                    U[i * m + j][r] = 1
    
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if j == r:
                    V[j * p + k][r] = 1
    
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
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
