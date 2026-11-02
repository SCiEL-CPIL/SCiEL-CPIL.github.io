"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: for all i,j,k: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j*p+k,r] = 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(W[k*n+i][r] * U[i*m+j][r] * V[j*p+k][r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    sb = _schoolbook()
    
    # Try alternating minimization for lower ranks
    for target_rank in [44, 43]:
        if time.time() > deadline:
            break
        np.random.seed(12345 + target_rank)
        U = np.random.randn(n*m, target_rank) * 0.1 + 0.5
        V = np.random.randn(m*p, target_rank) * 0.1 + 0.5
        W = np.random.randn(p*n, target_rank) * 0.1 + 0.5
        
        converged = False
        for it in range(80):
            if time.time() > deadline:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        for r in range(target_rank):
                            M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    try:
                        U[i*m+j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        for r in range(target_rank):
                            M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    try:
                        V[j*p+k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        for r in range(target_rank):
                            M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    try:
                        W[k*n+i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            # Check
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(W[k*n+i, r]*U[i*m+j, r]*V[j*p+k, r] for r in range(target_rank))
                        err += (val-1.0)**2
            if err < 1e-10:
                converged = True
                break
        if converged:
            res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(res["U"], res["V"], res["W"], n, m, p):
                return res
    
    return sb
# EVOLVE-BLOCK-END
