"""Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    random.seed(42)
    np.random.seed(42)
    
    nonzero = [(i, k, i, j, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def loss_fn(U, V, W, r):
        s = 0.0
        for (i, k, ip, jp, kp) in nonzero:
            v = sum(W[ri, k*n+i] * U[ri, ip*m+jp] * V[ri, jp*p+kp] for ri in range(r))
            s += (v - 1.0) ** 2
        return s
    
    def full_verify(U, V, W, r):
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for j2 in range(m):
                            for kp in range(p):
                                v = sum(W[ri, k*n+i] * U[ri, ip*m+jp] * V[ri, j2*p+kp] for ri in range(r))
                                t = 1.0 if (ip==i and jp==j2 and kp==k) else 0.0
                                if abs(v - t) > 1e-9:
                                    return False
        return True
    
    for r in [18, 20, 22, 24, 26, 28, 30, 32, 34, 36]:
        if time.time() > deadline - 10:
            break
        for restart in range(15):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(r, n*m) * 0.1
            V = np.random.randn(r, m*p) * 0.1
            W = np.random.randn(r, n*p) * 0.1
            lr = 0.01
            for it in range(4000):
                if time.time() > deadline - 3:
                    break
                dU = np.zeros_like(U); dV = np.zeros_like(V); dW = np.zeros_like(W)
                for (i,k,ip,jp,kp) in nonzero:
                    wi = k*n+i; ui = ip*m+jp; vi = jp*p+kp
                    v = sum(W[ri,wi]*U[ri,ui]*V[ri,vi] for ri in range(r))
                    e = v - 1.0
                    for ri in range(r):
                        dU[ri,ui] += 2*e*W[ri,wi]*V[ri,vi]
                        dV[ri,vi] += 2*e*W[ri,wi]*U[ri,ui]
                        dW[ri,wi] += 2*e*U[ri,ui]*V[ri,vi]
                U -= lr*dU; V -= lr*dV; W -= lr*dW
                if it % 500 == 0 and loss_fn(U,V,W,r) < 1e-7:
                    break
            if full_verify(U, V, W, r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1; V[j*p+k][c] = 1; W[k*n+i][c] = 1; c += 1
    return {"U": U, "V": V, "W": W}
