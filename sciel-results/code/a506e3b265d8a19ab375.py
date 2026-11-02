import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (4, rank), (6, rank), (6, rank).
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all i in [0,2), j in [0,2), k in [0,3)
    """
    n, m, p = DIMENSIONS  # 2, 2, 3
    
    def verify(U, V, W, rank):
        """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Rank-11 solution (verified Strassen-like)
    def rank_11():
        rank = 11
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        def su(i,j,r,v): U[i*m+j][r]=v
        def sv(j,k,r,v): V[j*p+k][r]=v
        def sw(k,i,r,v): W[k*n+i][r]=v
        su(0,0,0,1); su(1,1,0,1); sv(0,0,0,1); sv(1,1,0,1); sw(0,0,0,1); sw(1,1,0,1)
        su(1,0,1,1); su(1,1,1,1); sv(0,0,1,1); sw(0,1,1,1); sw(1,1,1,-1)
        su(0,0,2,1); sv(0,1,2,1); sv(1,1,2,-1); sw(1,0,2,1); sw(1,1,2,1)
        su(1,1,3,1); sv(1,0,3,1); sv(0,0,3,-1); sw(0,0,3,1); sw(0,1,3,1)
        su(0,0,4,1); su(0,1,4,1); sv(1,1,4,1); sw(0,0,4,-1); sw(1,0,4,1)
        su(1,0,5,1); su(0,0,5,-1); sv(0,0,5,1); sv(0,1,5,1); sw(1,1,5,1)
        su(0,1,6,1); su(1,1,6,-1); sv(1,0,6,1); sv(1,1,6,1); sw(0,0,6,1)
        su(0,0,7,1); sv(0,2,7,1); sw(2,0,7,1)
        su(0,1,8,1); sv(1,2,8,1); sw(2,0,8,1)
        su(1,0,9,1); sv(0,2,9,1); sw(2,1,9,1)
        su(1,1,10,1); sv(1,2,10,1); sw(2,1,10,1)
        return U, V, W
    
    U11, V11, W11 = rank_11()
    if verify(U11, V11, W11, 11):
        # Try to find rank 10
        np.random.seed(123)
        for trial in range(200):
            U = np.random.randn(n*m, 10) * 0.3
            V = np.random.randn(m*p, 10) * 0.3
            W_sol = np.zeros((p*n, 10))
            ok = True
            for i in range(n):
                for k in range(p):
                    M = np.array([[U[i*m+j, r] * V[j*p+k, r] for r in range(10)] for j in range(m)])
                    rhs = np.ones(m)
                    try:
                        w, res, rk, sv = np.linalg.lstsq(M, rhs, rcond=None)
                        if np.allclose(M @ w, rhs, atol=1e-10):
                            W_sol[k*n+i] = w
                        else:
                            ok = False
                            break
                    except:
                        ok = False
                        break
                if not ok:
                    break
            if ok and verify(U.tolist(), V.tolist(), W_sol.tolist(), 10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
        # Return rank 11
        return {"U": U11, "V": V11, "W": W11}
    else:
        # Fallback: trivial rank 12
        rank = 12
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
