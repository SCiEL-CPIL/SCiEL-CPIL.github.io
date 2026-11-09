"""Randomized exact rank search for 3x4x4x8 matrix multiplication tensor."""
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    random.seed(42)
    
    def verify(U, V, W, n, m, p):
        """Verify tensor identity: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
        return U, V, W
    
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Try various ranks from low to high
    for r in range(24, best_rank):
        if r < max(n * m, m * p, p * n):
            continue
        
        found = False
        for attempt in range(100):
            U = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
            
            W = [[0] * r for _ in range(p * n)]
            consistent = True
            
            for i in range(n):
                for k in range(p):
                    A = [[U[i*m+j][c] * V[j*p+k][c] for c in range(r)] for j in range(m)]
                    A_np = np.array(A, dtype=float)
                    b_np = np.ones(m, dtype=float)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                        if np.allclose(A_np @ w, b_np, atol=1e-9):
                            w_int = np.round(w).astype(int)
                            if np.allclose(A_np @ w_int, b_np, atol=1e-9):
                                W[k*n+i] = list(w_int)
                            else:
                                consistent = False
                                break
                        else:
                            consistent = False
                            break
                    except Exception:
                        consistent = False
                        break
                
                if not consistent:
                    break
            
            if consistent and verify(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
