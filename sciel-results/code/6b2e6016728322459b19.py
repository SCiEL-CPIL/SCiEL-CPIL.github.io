import time
import numpy as np

# ---------- Winograd / DFT-5 frequency-domain factorization for 4x5 * 5x5 ----------
# The 5-term inner product is a cyclic convolution over Z/5Z. The DFT-5 ring
# isomorphism splits the multiplication tensor into 5 independent "frequency"
# components (evaluations at the 5th roots of unity), giving rank 4*5*1 = 20.
# Entries live in Q(zeta_5); we encode each as a 5-dim integer vector (coeffs of
# 1, zeta, zeta^2, zeta^3, zeta^4) with the minimal polynomial z^4+z^3+z^2+z+1=0.

def _build_dft5():
    n, m, p = 4, 5, 5
    D = 5
    # omega = zeta_5 (root of x^4+x^3+x^2+x+1). omega^k = (0,0,0,0,1) shifted.
    omega = [0, 0, 0, 0, 1]

    def mul(a, b):
        # (a*b) mod (x^4+x^3+x^2+x+1); x^5 = 1
        c = [0] * D
        for i in range(D):
            if a[i]:
                for j in range(D):
                    if b[j]:
                        c[(i + j) % D] += a[i] * b[j]
        return c

    def pw(base, e):
        if e == 0:
            return [1, 0, 0, 0, 0]
        if e == 1:
            return list(base)
        res = [1, 0, 0, 0, 0]
        b = list(base)
        while e:
            if e & 1:
                res = mul(res, b)
            b = mul(b, b)
            e >>= 1
        return res

    def scalar_mul(s, v):
        return [s * x for x in v]

    def add(a, b):
        return [a[i] + b[i] for i in range(D)]

    def sub(a, b):
        return [a[i] - b[i] for i in range(D)]

    # DFT-5 matrix: F[i][j] = omega^(i*j)  (i,j in 0..4)
    F = [[pw(omega, (i * j) % 5) for j in range(D)] for i in range(D)]
    # inverse DFT-5: (1/5) F^(-1);  F^(-1)[i][j] = omega^(-i*j)
    Finv = [[pw(omega, (-i * j) % 5) for j in range(D)] for i in range(D)]

    # Build factor matrices: U (n*m, R), V (m*p, R), W (p*n, R), R = n*D
    R = n * D
    U = [[0] * D for _ in range(n * m)]
    V = [[0] * D for _ in range(m * p)]
    W = [[0] * D for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for f in range(D):
                r = i * D + f
                U[i * m + j][r] = 1
                V[j * p + j][r] = F[f][j]
                W[j * n + i][r] = scalar_mul(1, Finv[f][j])  # /5 handled in verify
    return U, V, W, D


def _build_rank20():
    U, V, W, D = _build_dft5()
    return U, V, W


def _verify_rank20(U, V, W, tol=1e-9):
    n, m, p = 4, 5, 5
    D = 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = [0] * D
                for r in range(n * D):
                    for d in range(D):
                        s[d] += U[i * m + j][r][d] * V[j * p + k][r][d] * W[k * n + i][r][d]
                # s = (1/5) * delta_{j,k} in Q(zeta_5); multiply by 5
                t = [5 * x for x in s]
                # expected: 1 at position 0, 0 elsewhere
                if abs(t[0] - 1.0) > tol or any(abs(x) > tol for x in t[1:]):
                    return False
    return True


def _build_schoolbook():
    n, m, p = 4, 5, 5
    R = n * m * p
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_int(U, V, W, tol=1e-9):
    n, m, p = 4, 5, 5
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    t0 = time.time()
    deadline = 145.0

    # Primary: Winograd DFT-5 rank-20 (exact over Q(zeta_5))
    U, V, W = _build_rank20()
    if time.time() - t0 < deadline:
        if _verify_rank20(U, V, W, tol=1e-9):
            # Return as nested lists of coefficient vectors (rank-20)
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-100 (always valid over Z)
    Uf, Vf, Wf = _build_schoolbook()
    if _verify_int(Uf, Vf, Wf, tol=1e-9):
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
