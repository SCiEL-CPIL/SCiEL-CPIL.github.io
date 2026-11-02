import numpy as np
import time
from fractions import Fraction

def construct():
    start = time.time()
    DEADLINE = start + 110  # safety margin (120s limit, keep 10s buffer)
    
    n, m, p = 3, 3, 7
    
    def time_ok():
        return time.time() < DEADLINE
    
    # Schoolbook fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    def residual_f(U, V, W):
        r = U.shape[1]
        Ur = U.reshape(n,m,r)
        Vr = V.reshape(m,p,r)
        Wr = W.reshape(p,n,r)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.max(np.abs(T - 1.0))
    
    def als(U, V, W, iters=200, tol=1e-10):
        r = U.shape[1]
        for it in range(iters):
            if not time_ok():
                break
            # Solve U
            for i in range(n):
                for j in range(m):
                    # 7 eqs: sum_c U[i*m+j,c] * V[j*p+k,c]*W[k*n+i,c] = 1
                    A = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(r):
                            A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # least squares
                    U[i*m+j,:] = np.linalg.lstsq(A, b, rcond=None)[0]
            # Solve V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(r):
                            A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    V[j*p+k,:] = np.linalg.lstsq(A, b, rcond=None)[0]
            # Solve W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(r):
                            A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    W[k*n+i,:] = np.linalg.lstsq(A, b, rcond=None)[0]
            if residual_f(U,V,W) < tol:
                break
        return residual_f(U,V,W)
    
    # Search for minimal rank
    best_solution = None
    best_rank = n*m*p  # 63
    
    for r in range(25, 64):
        if not time_ok():
            break
        found = False
        for trial in range(20):
            if not time_ok():
                break
            rng = np.random.default_rng(12345 + trial + r*1000)
            U = rng.normal(size=(n*m, r)) * 0.5
            V = rng.normal(size=(m*p, r)) * 0.5
            W = rng.normal(size=(p*n, r)) * 0.5
            res = als(U, V, W)
            if res < 1e-8:
                found = True
                best_solution = (U.copy(), V.copy(), W.copy())
                best_rank = r
                break
        if found:
            break
    
    if best_solution is None:
        return schoolbook()
    
    U, V, W = best_solution
    # Round to half-integers
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify exactness with Fraction
    def exact_residual(Uf, Vf, Wf):
        r = len(Uf[0])
        maxerr = Fraction(0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(Uf[i*m+j][c]) * Fraction(Vf[j*p+k][c]) * Fraction(Wf[k*n+i][c])
                    err = abs(s - 1)
                    if err > maxerr:
                        maxerr = err
        return maxerr
    
    err = exact_residual(U_r.tolist(), V_r.tolist(), W_r.tolist())
    if err == 0:
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Rounding failed; try local repair or fall back
    # Try a few local perturbations of the rounded solution
    # (simple: return schoolbook to be safe, or try next rank)
    return schoolbook()
