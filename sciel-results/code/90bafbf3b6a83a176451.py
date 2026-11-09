import time
import math
import numpy as np

DIMENSIONS = [6, 6, 6]

START_TIME = time.time()
TIME_LIMIT = 120.0  # safety margin before 150s kill


def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 10.0)


def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two tensor decompositions.
    Ua: (n1*m1, r1), Va: (m1*p1, r1), Wa: (p1*n1, r1)
    Ub: (n2*m2, r2), Vb: (m2*p2, r2), Wb: (p2*n2, r2)
    Result: n=n1*n2, m=m1*m2, p=p1*p2, rank=r1*r2
    W index convention: W[k*n + i, r] for output C[i, k].
    """
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]

    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r = r1 * r2

    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            # U
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1 * m1 + j1, r1i]
                    if a == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2 * m2 + j2, r2i]
                            if b == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, r_idx] = a * b
            # V
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1 * p1 + k1, r1i]
                    if a == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2 * p2 + k2, r2i]
                            if b == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, r_idx] = a * b
            # W
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1 * n1 + i1, r1i]
                    if a == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2 * n2 + i2, r2i]
                            if b == 0:
                                continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i, r_idx] = a * b

    return U, V, W


def _strassen_2x2():
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M4 = d(g-c)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1

    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C01 = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    # C10 = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W


def _laderman_3x3():
    n, m, p = 3, 3, 3
    r = 23
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    # A indices: a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7,i=8
    # B indices: w=0,x=1,y=2,z=3,u=4,v=5,p=6,q=7,r=8
    # C indices: C[i,k] with i=0,1,2 and k=0,1,2

    # P0 = (a+d)(w+z)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # P1 = (a+b)(w+x)
    U[0, 1] = 1; U[1, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    # P2 = (b-d)(x-z)
    U[1, 2] = 1; U[3, 2] = -1
    V[1, 2] = 1; V[3, 2] = -1
    # P3 = (c+f)(y+v)
    U[2, 3] = 1; U[5, 3] = 1
    V[2, 3] = 1; V[5, 3] = 1
    # P4 = (f-i)(v-q)
    U[5, 4] = 1; U[8, 4] = -1
    V[5, 4] = 1; V[7, 4] = -1
    # P5 = (i+g)(r+p)
    U[8, 5] = 1; U[6, 5] = 1
    V[8, 5] = 1; V[6, 5] = 1
    # P6 = (d+e)(z+u)
    U[3, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[4, 6] = 1
    # P7 = (e-g)(u-p)
    U[4, 7] = 1; U[6, 7] = -1
    V[4, 7] = 1; V[6, 7] = -1
    # P8 = (b+e)(x+u)
    U[1, 8] = 1; U[4, 8] = 1
    V[1, 8] = 1; V[4, 8] = 1
    # P9 = (a+e)(w+u)
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[4, 9] = 1
    # P10 = (c+i)(y+r)
    U[2, 10] = 1; U[8, 10] = 1
    V[2, 10] = 1; V[8, 10] = 1
    # P11 = (d+g)(z+p)
    U[3, 11] = 1; U[6, 11] = 1
    V[3, 11] = 1; V[6, 11] = 1
    # P12 = (a+c)(w+y)
    U[0, 12] = 1; U[2, 12] = 1
    V[0, 12] = 1; V[2, 12] = 1
    # P13 = (d+f)(z+v)
    U[3, 13] = 1; U[5, 13] = 1
    V[3, 13] = 1; V[5, 13] = 1
    # P14 = (g+i)(p+r)
    U[6, 14] = 1; U[8, 14] = 1
    V[6, 14] = 1; V[8, 14] = 1
    # P15 = (b+f+h)(x+v+q)
    U[1, 15] = 1; U[5, 15] = 1; U[7, 15] = 1
    V[1, 15] = 1; V[5, 15] = 1; V[7, 15] = 1
    # P16 = (a+d+e)(w+z+u)
    U[0, 16] = 1; U[3, 16] = 1; U[4, 16] = 1
    V[0, 16] = 1; V[3, 16] = 1; V[4, 16] = 1
    # P17 = (b+e+h)(x+u+q)
    U[1, 17] = 1; U[4, 17] = 1; U[7, 17] = 1
    V[1, 17] = 1; V[4, 17] = 1; V[7, 17] = 1
    # P18 = (c+f+i)(y+v+r)
    U[2, 18] = 1; U[5, 18] = 1; U[8, 18] = 1
    V[2, 18] = 1; V[5, 18] = 1; V[8, 18] = 1
    # P19 = (a+d+g)(w+z+p)
    U[0, 19] = 1; U[3, 19] = 1; U[6, 19] = 1
    V[0, 19] = 1; V[3, 19] = 1; V[6, 19] = 1
    # P20 = (a+b+c)(w+x+y)
    U[0, 20] = 1; U[1, 20] = 1; U[2, 20] = 1
    V[0, 20] = 1; V[1, 20] = 1; V[2, 20] = 1
    # P21 = (d+e+f)(z+u+v)
    U[3, 21] = 1; U[4, 21] = 1; U[5, 21] = 1
    V[3, 21] = 1; V[4, 21] = 1; V[5, 21] = 1
    # P22 = (g+h+i)(p+q+r)
    U[6, 22] = 1; U[7, 22] = 1; U[8, 22] = 1
    V[6, 22] = 1; V[7, 22] = 1; V[8, 22] = 1

    # W coefficients: W[k*3+i, r] such that C[i,k] = sum_r W[k*3+i,r] * P_r
    # C[0,0] = a*w + b*z + c*p = P0+P9-P12+P16 - P20 ... let me compute
    # Actually, let me just solve it numerically with integer rounding
    W = _solve_W(U, V, n, m, p)
    return U, V, W


def _solve_W(U, V, n, m, p):
    """Solve for W such that the decomposition exactly reproduces C = A B.
    For each r, define G_r[i,k] = sum_j U[i*m+j, r] * V[j*p+k, r].
    We need: for all (i,k): sum_r W[k*n+i, r] * G_r[i,k] = 1.
    This is p*n equations with p*n*r unknowns. Solve via least squares.
    """
    r = U.shape[1]
    # Precompute G_r matrices
    G = np.zeros((r, n, p))
    for rr in range(r):
        for i in range(n):
            for k in range(p):
                g = 0.0
                for j in range(m):
                    g += U[i * m + j, rr] * V[j * p + k, rr]
                G[rr, i, k] = g
    
    # Build system: for each output (k,i), sum_r W[k*n+i, r] * G[r,i,k] = 1
    # This is a (p*n) x (p*n*r) system, but actually each equation only
    # involves r unknowns (one per r for that specific (k,i) position).
    # So it's really p*n independent systems of r unknowns each.
    # But wait - that's not right. Each equation for output (i,k) involves
    # W[k*n+i, r] for all r, weighted by G[r,i,k].
    # So we have p*n equations, each with r unknowns (the W entries for that
    # specific output position). These are INDEPENDENT systems!
    
    W = np.zeros((p * n, r))
    for i in range(n):
        for k in range(p):
            # Equation: sum_r W[k*n+i, r] * G[r, i, k] = 1
            # This is r unknowns, 1 equation. Under-determined.
            # We need additional constraints. Actually, the full tensor
            # identity requires ALL (i',k') outputs to be correct simultaneously.
            # But since G_r are fixed by U,V, the system is:
            # For output (i',k'): sum_r W[k'*n+i', r] * G[r, i', k'] = delta_{i'i} * delta_{k'k}
            # Wait no - we want the full multiplication tensor, which is 1 for
            # ALL (i',k') pairs (it's the identity tensor for matmul).
            # Actually the matmul tensor T[i,k] = 1 for all i,k (each output
            # entry is a bilinear form). So we need:
            # For ALL (i',k'): sum_r W[k'*n+i', r] * G[r, i', k'] = 1
            # This means each row of W (indexed by k'*n+i') must satisfy:
            # W[k'*n+i', :] @ G[:, i', k'] = 1
            # These are p*n independent under-determined systems.
            # Pick minimum norm solution.
            g_vec = G[:, i, k]  # shape (r,)
            if np.allclose(g_vec, 0):
                W[k * n + i, :] = 0
            else:
                # min norm solution to w @ g = 1 is w = g / (g@g)
                denom = np.dot(g_vec, g_vec)
                if abs(denom) < 1e-12:
                    W[k * n + i, :] = 0
                else:
                    W[k * n + i, :] = g_vec / denom
    # Round to nearest integer
    W = np.rint(W)
    # Verify exactness
    for i in range(n):
        for k in range(p):
            val = 0.0
            for rr in range(r):
                val += W[k * n + i, rr] * G[rr, i, k]
            if abs(val - 1.0) > 1e-6:
                raise ValueError("Laderman W solve not exact at (%d,%d): %s" % (i, k, val))
    return W


def _verify(U, V, W, n, m, p, seed=0):
    """Check that the decomposition reproduces C = A B for random A, B."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-3, 4, size=(n, m)).astype(float)
    B = rng.integers(-3, 4, size=(m, p)).astype(float)
    C_true = A @ B
    # Reconstruct: T[i,k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] V[j*p+k, r]
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    C_rec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for rr in range(U.shape[1]):
                gj = 0.0
                for j in range(m):
                    gj += U[i * m + j, rr] * V[j * p + k, rr]
                s += W[k * n + i, rr] * gj
            C_rec[i, k] = s
    return np.allclose(C_rec, C_true, atol=1e-9)


def construct():
    n, m, p = DIMENSIONS
    if _check_time():
        # Fallback: schoolbook 6x6
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try the improved Kronecker composition: Strassen(2x2, r=7) x Laderman(3x3, r=23)
    try:
        U2, V2, W2 = _strassen_2x2()
        U3, V3, W3 = _laderman_3x3()
        U, V, W = _kron_decompose(U2, V2, W2, U3, V3, W3,
                                  2, 2, 2, 3, 3, 3)
        # Verify exactness on a couple of random instances
        ok = _verify(U, V, W, n, m, p, seed=1) and _verify(U, V, W, n, m, p, seed=7)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Fallback: Strassen(2x2) x schoolbook(3x3), rank 189 (known-good)
    U2, V2, W2 = _strassen_2x2()
    n3, m3, p3 = 3, 3, 3
    r3 = n3 * m3 * p3
    U3 = np.zeros((n3 * m3, r3))
    V3 = np.zeros((m3 * p3, r3))
    W3 = np.zeros((p3 * n3, r3))
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j, col] = 1
                V3[j * p3 + k, col] = 1
                W3[k * n3 + i, col] = 1
                col += 1
    U, V, W = _kron_decompose(U2, V2, W2, U3, V3, W3, 2, 2, 2, 3, 3, 3)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
