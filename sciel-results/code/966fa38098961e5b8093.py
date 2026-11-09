import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]
T0 = time.time()
DEADLINE = T0 + 110  # safety margin before 120s

def _build_T():
    n, m, p = DIMENSIONS
    # T[i, j, k] with i=(a,b) in [0,n*m), j=(b,c) in [0,m*p), k=(a,c) in [0,p*n)
    # T[i,j,k] = 1 if consistent (a,b),(b,c),(a,c) else 0
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T[i, j, k] = 1.0
    return T

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
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
    return {"U": U, "V": V, "W": W}

def _als(T, R, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    n, m, p = T.shape
    U = rng.normal(size=(n, R))
    V = rng.normal(size=(m, R))
    W = rng.normal(size=(p, R))
    for it in range(iters):
        # Update U
        # Precompute sums of squares for V and W
        V_sq_sums = np.sum(V**2, axis=0) # shape (R,)
        W_sq_sums = np.sum(W**2, axis=0) # shape (R,)
        
        for i in range(n):
            for r in range(R):
                # Num: sum_{j,k} T[i, j, k] V[j, r] W[k, r]
                # Mat is outer product of V[:, r] and W[:, r] -> shape (m, p)
                mat = np.outer(V[:, r], W[:, r])
                num = np.sum(T[i] * mat)
                den = V_sq_sums[r] * W_sq_sums[r]
                U[i, r] = num / den if den > 1e-12 else 0.0

        # Update V
        U_sq_sums = np.sum(U**2, axis=0) # shape (R,)
        W_sq_sums = np.sum(W**2, axis=0) # shape (R,)
        
        for j in range(m):
            for r in range(R):
                # Num: sum_{i,k} T[i, j, k] U[i, r] W[k, r]
                # Mat is outer product of U[:, r] and W[:, r] -> shape (n, p)
                mat = np.outer(U[:, r], W[:, r])
                num = np.sum(T[:, j, :] * mat)
                den = U_sq_sums[r] * W_sq_sums[r]
                V[j, r] = num / den if den > 1e-12 else 0.0

        # Update W
        U_sq_sums = np.sum(U**2, axis=0) # shape (R,)
        V_sq_sums = np.sum(V**2, axis=0) # shape (R,)
        
        for k in range(p):
            for r in range(R):
                # Num: sum_{i,j} T[i, j, k] U[i, r] V[j, r]
                # Mat is outer product of U[:, r] and V[:, r] -> shape (n, m)
                mat = np.outer(U[:, r], V[:, r])
                num = np.sum(T[:, :, k] * mat)
                den = U_sq_sums[r] * V_sq_sums[r]
                W[k, r] = num / den if den > 1e-12 else 0.0

    # residual
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    res = np.linalg.norm(T - recon)
    return U, V, W, res

def _exact_verify(U, V, W, T_int):
    n, m, p = T_int.shape
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(R):
                    s += Fraction(U[i][r]) * Fraction(V[j][r]) * Fraction(W[k][r])
                if s != T_int[i, j, k]:
                    return False
    return True

def _to_frac_list(arr, denom):
    # round arr to nearest multiple of 1/denom
    out = []
    for row in arr:
        r = []
        for x in row:
            r.append(round(x * denom) / denom)
        out.append(r)
    return out

def construct():
    T = _build_T()
    T_int = T.astype(int)
    best = None
    best_rank = None
    for R in range(12, 36):
        if time.time() > DEADLINE:
            break
        found = None
        for seed in range(0, 40):
            if time.time() > DEADLINE:
                break
            U, V, W, res = _als(T, R, iters=150, seed=seed)
            if res < 1e-5:
                # try to exact-ify
                for denom in [1, 2, 3, 4, 6, 12]:
                    Uf = _to_frac_list(U, denom)
                    Vf = _to_frac_list(V, denom)
                    Wf = _to_frac_list(W, denom)
                    if _exact_verify(Uf, Vf, Wf, T_int):
                        found = (Uf, Vf, Wf)
                        break
                if found:
                    break
        if found:
            Uf, Vf, Wf = found
            return {"U": Uf, "V": Vf, "W": Wf}
    # fallback
    return _schoolbook()
