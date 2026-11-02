import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, leave 20s margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, rank = schoolbook()
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    if not verify(U, V, W):
        # Should never happen, but just in case
        pass
    
    # Try Strassen-based decomposition
    if time.time() - start_time < DEADLINE:
        U_s, V_s, W_s, rank_s = strassen_tiling()
        if verify(U_s, V_s, W_s):
            if rank_s < best_rank:
                best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s
    
    # Try to improve with optimization
    if time.time() - start_time < DEADLINE - 10:
        # Try various ranks
        for target_rank in range(best_rank - 1, max(20, best_rank - 20), -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            U_t, V_t, W_t = try_optimize(target_rank)
            if U_t is not None and verify(U_t, V_t, W_t):
                best_U, best_V, best_W, best_rank = U_t, V_t, W_t, target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
