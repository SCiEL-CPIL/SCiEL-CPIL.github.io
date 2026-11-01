import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _residual(U, V, W, n, m, p):
    """Compute residual for rank-r decomposition."""
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)

def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U

def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V

def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W

def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _try_rank10_structural(deadline):
    """Try to find rank-10 by exploiting Strassen structure for 2x2x2 + 2 extra."""
    n, m, p = 2, 2, 3
    r = 10
    
    # Strassen for 2x2x2: 7 terms
    # M1 = A00*(B01+B11), M2 = A10*(B00+B01), M3 = A00*A11, M4 = A11*(B10-B00)
    # M5 = A01*(B00+B11), M6 = A10-A00*(B11-B10), M7 = A01-A11*(B01-B10)
    # C00 = M3+M4-M5+M7, C01 = M1+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # For the third column (k=2), we need:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # This is rank 2: two terms
    
    # Total: 7 + 2 = 9? But the Strassen terms already involve B02? No, they don't.
    # So we have 9 terms. Let me check if this is valid.
    
    # Actually, let me just try to build a rank-9 or rank-10 and verify.
    
    # Let me try rank 9 first: 7 Strassen + 2 for column 2
    # But the Strassen decomposition is for B[:, 0:2], not including column 2.
    # So the 7 Strassen terms handle C[:, 0:2], and 2 more handle C[:, 2].
    # Total rank = 9.
    
    # Let me construct this explicitly.
    
    U9 = [[0]*9 for _ in range(4)]
    V9 = [[0]*9 for _ in range(6)]
    W9 = [[0]*9 for _ in range(6)]
    
    # Strassen terms (indices 0-6)
    # M1 = A00*(B01+B11): U[0,0]=1, V[0,1]=1, V[1,1]=1, W[1,0]=1, W[1,1]=1
    U9[0][0] = 1  # A00
    V9[1][0] = 1  # B01
    V9[4][0] = 1  # B11
    W9[1][0] = 1  # contributes to C01
    W9[1][1] = 1  # contributes to C11
    
    # M2 = A10*(B00+B01): U[1,0]=1, V[0,0]=1, V[1,0]=1, W[0,1]=1, W[1,1]=1
    U9[2][1] = 1  # A10
    V9[0][1] = 1  # B00
    V9[1][1] = 1  # B01
    W9[0][1] = 1  # C10
    W9[1][1] = 1  # C11
    
    # M3 = A00*A11: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    U9[0][2] = 1
    U9[3][2] = 1
    V9[0][2] = 1
    V9[3][2] = 1
    W9[0][2] = 1
    W9[1][2] = 1
    
    # This is getting complicated. Let me just use the known rank-11 and try ALS to reduce to 10.
    
    sol11 = _rank11_solution()
    U11, V11, W11 = sol11["U"], sol11["V"], sol11["W"]
    U_warm = [row[:r] for row in U11]
    V_warm = [row[:r] for row in V11]
    W_warm = [row[:r] for row in W11]
    
    inits = [(U_warm, V_warm, W_warm)]
    for _ in range(50):
        if time.time() > deadline:
            break
        U_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
        V_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
        W_rand = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
        inits.append((U_rand, V_rand, W_rand))
    
    for U0, V0, W0 in inits:
        if time.time() > deadline:
            break
        U, V, W = _als_optimize(U0, V0, W0, n, m, p, r, iters=60, deadline=deadline)
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
    
    return None

def _try_rank10_scipy(deadline):
    """Use scipy.optimize to find rank-10 solution."""
    n, m, p = 2, 2, 3
    r = 10
    num_params = n*m*r + m*p*r + p*n*r  # 4*10 + 6*10 + 6*10 = 160
    
    def _flatten(U, V, W):
        return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
    
    def _unflatten(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
        W = x[n*m*r+m*p*r:].reshape(p*n, r)
        return U, V, W
    
    def _objective(x):
        U, V, W = _unflatten(x)
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    res.append(s - 1.0)
        return res
    
    # Start from rank-11 truncated
    sol11 = _rank11_solution()
    U11, V11, W11 = sol11["U"], sol11["V"], sol11["W"]
    x0 = _flatten([row[:r] for row in U11], [row[:r] for row in V11], [row[:r] for row in W11])
    
    try:
        result = least_squares(_objective, x0, method='lm', max_nfev=5000, xtol=1e-12, ftol=1e-12)
        U, V, W = _unflatten(result.x)
        Ur, Vr, Wr = _round_half(U.tolist()), _round_half(V.tolist()), _round_half(W.tolist())
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    best = _rank11_solution()
    
    if time.time() < deadline:
        result10 = _try_rank10_structural(deadline)
        if result10 is not None:
            best = {"U": result10[0], "V": result10[1], "W": result10[2]}
    
    if time.time() < deadline and best["U"][0] and len(best["U"][0]) > 10:
        result10b = _try_rank10_scipy(deadline)
        if result10b is not None:
            best = {"U": result10b[0], "V": result10b[1], "W": result10b[2]}
    
    return best
