"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Schoolbook fallback
    sb_rank = n * m * p
    def schoolbook():
        U = [[0]*sb_rank for _ in range(n*m)]
        V = [[0]*sb_rank for _ in range(m*p)]
        W = [[0]*sb_rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = sb_rank
    
    def verify(U, V, W, r, tol=1e-5):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(float(np.dot(U[i*m+j], V[j*p+k]*W[k*n+i])) - 1.0) > tol:
                        return False
        return True
    
    indices = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    
    def attempt(r, seed, iters=3000):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, r)*0.3
        V = rng.randn(m*p, r)*0.3
        W = rng.randn(p*n, r)*0.3
        lr = 0.02
        for it in range(iters):
            if time.time() > deadline:
                return None
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for (i,j,k) in indices:
                ui = U[i*m+j]; vk = V[j*p+k]; wk = W[k*n+i]
                e = float(np.dot(ui, vk*wk)) - 1.0
                dU[i*m+j] += e*vk*wk
                dV[j*p+k] += e*ui*wk
                dW[k*n+i] += e*ui*vk
            U -= lr*dU; V -= lr*dV; W -= lr*dW
            if it % 500 == 0:
                mx = 0
                for (i,j,k) in indices:
                    mx = max(mx, abs(float(np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]))-1.0))
                if mx < 1e-6:
                    return (U,V,W)
        mx = 0
        for (i,j,k) in indices:
            mx = max(mx, abs(float(np.dot(U[i*m+j], V[j*p+k]*W[k*n+i]))-1.0))
        if mx < 1e-5:
            return (U,V,W)
        return None
    
    lb = max(n*m, m*p, p*n)
    for r in range(lb, best_rank+1):
        if time.time() > deadline - 5:
            break
        for s in range(30):
            if time.time() > deadline - 5:
                break
            res = attempt(r, seed=r*1000+s, iters=2000)
            if res:
                U,V,W = res
                Ul = [[float(x) for x in U[i]] for i in range(n*m)]
                Vl = [[float(x) for x in V[i]] for i in range(m*p)]
                Wl = [[float(x) for x in W[i]] for i in range(p*n)]
                if verify(Ul, Vl, Wl, r):
                    best_rank = r
                    best_U, best_V, best_W = Ul, Vl, Wl
                    break
        if best_rank == lb:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
