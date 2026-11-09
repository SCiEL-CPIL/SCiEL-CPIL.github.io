import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0  # safety margin before 150s kill


def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 10.0)


def _strassen_2x2():
    """Strassen 2x2 decomposition, rank 7.
    Returns U (4x7), V (4x7), W (4x7) with W row index = k*n+i.
    A = [[a,b],[d,e]], B = [[w,x],[z,u]], C = AB.
    M1=(a+d)(w+u)  M2=(d)(w+z)  M3=a(x-z)  M4=e(w+x)
    M5=(a+b)u      M6=(d-b)(x+u)  M7=(e-a)(z+w)
    C00=M1+M4-M5+M7  C01=M3+M5  C10=M2+M4  C11=M1-M2+M3+M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # U rows: a=0,d=1,b=2,e=3
    U[0, 0] = 1; U[1, 0] = 1   # a+d
    U[1, 1] = 1                # d
    U[0, 2] = 1                # a
    U[3, 3] = 1                # e
    U[0, 4] = 1; U[2, 4] = 1   # a+b
    U[1, 5] = 1; U[2, 5] = -1  # d-b
    U[3, 6] = 1; U[0, 6] = -1  # e-a
    # V rows: w=0,z=1,x=2,u=3
    V[0, 0] = 1; V[3, 0] = 1   # w+u
    V[0, 1] = 1; V[1, 1] = 1   # w+z
    V[2, 2] = 1; V[1, 2] = -1  # x-z
    V[0, 3] = 1; V[2, 3] = 1   # w+x
    V[3, 4] = 1                # u
    V[2, 5] = 1; V[3, 5] = 1   # x+u
    V[1, 6] = 1; V[0, 6] = 1   # z+w
    # W rows: k*n+i -> C00=0, C10=1, C01=2, C11=3
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1   # C00
    W[1, 1] = 1; W[1, 3] = 1                              # C10
    W[2, 2] = 1; W[2, 4] = 1                              # C01
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1   # C11
    return U, V, W


def _laderman_UV():
    """Laderman 3x3 decomposition, rank 23.
    Returns U (9x23), V (9x23). W is solved separately.
    A = [[a,b,c],[d,e,f],[g,h,i]] (rows: a=0,b=1,c=2,d=3,e=4,f=5,g=6,h=7,i=8)
    B = [[w,x,y],[z,u,v],[p,q,r]] (rows: w=0,x=1,y=2,z=3,u=4,v=5,p=6,q=7,r=8)
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    # P1 = (a+d)*(w+z)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # P2 = (b+e)*(x+u)
    U[1, 1] = 1; U[4, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1
    # P3 = (c+f)*(y+v)
    U[2, 2] = 1; U[5, 2] = 1
    V[2, 2] = 1; V[5, 2] = 1
    # P4 = (d+g)*(z+p)
    U[3, 3] = 1; U[6, 3] = 1
    V[3, 3] = 1; V[6, 3] = 1
    # P5 = (e+h)*(u+q)
    U[4, 4] = 1; U[7, 4] = 1
    V[4, 4] = 1; V[7, 4] = 1
    # P6 = (f+i)*(v+r)
    U[5, 5] = 1; U[8, 5] = 1
    V[5, 5] = 1; V[8, 5] = 1
    # P7 = (a+b)*(w+x)
    U[0, 6] = 1; U[1, 6] = 1
    V[0, 6] = 1; V[1, 6] = 1
    # P8 = (d+e)*(z+u)
    U[3, 7] = 1; U[4, 7] = 1
    V[3, 7] = 1; V[4, 7] = 1
    # P9 = (g+h)*(p+q)
    U[6, 8] = 1; U[7, 8] = 1
    V[6, 8] = 1; V[7, 8] = 1
    # P10 = (b+c)*(x+y)
    U[1, 9] = 1; U[2, 9] = 1
    V[1, 9] = 1; V[2, 9] = 1
    # P11 = (e+f)*(u+v)
    U[4, 10] = 1; U[5, 10] = 1
    V[4, 10] = 1; V[5, 10] = 1
    # P12 = (h+i)*(q+r)
    U[7, 11] = 1; U[8, 11] = 1
    V[7, 11] = 1; V[8, 11] = 1
    # P13 = (a+c)*(w+y)
    U[0, 12] = 1; U[2, 12] = 1
    V[0, 12] = 1; V[2, 12] = 1
    # P14 = (d+f)*(z+v)
    U[3, 13] = 1; U[5, 13] = 1
    V[3, 13] = 1; V[5, 13] = 1
    # P15 = (g+i)*(p+r)
    U[6, 14] = 1; U[8, 14] = 1
    V[6, 14] = 1; V[8, 14] = 1
    # P16 = (a+d+g)*(w+z+p)
    U[0, 15] = 1; U[3, 15] = 1; U[6, 15] = 1
    V[0, 15] = 1; V[3, 15] = 1; V[6, 15] = 1
    # P17 = (b+e+h)*(x+u+q)
    U[1, 16] = 1; U[4, 16] = 1; U[7, 16] = 1
    V[1, 16] = 1; V[4, 16] = 1; V[7, 16] = 1
    # P18 = (c+f+i)*(y+v+r)
    U[2, 17] = 1; U[5, 17] = 1; U[8, 17] = 1
    V[2, 17] = 1; V[5, 17] = 1; V[8, 17] = 1
    # P19 = (a+e+i)*(w+u+r)
    U[0, 18] = 1; U[4, 18] = 1; U[8, 18] = 1
    V[0, 18] = 1; V[4, 18] = 1; V[8, 18] = 1
    # P20 = (d+h+i)*(z+q+r)
    U[3, 19] = 1; U[7, 19] = 1; U[8, 19] = 1
    V[3, 19] = 1; V[7, 19] = 1; V[8, 19] = 1
    # P21 = (b+d+f)*(x+z+v)
    U[1, 20] = 1; U[3, 20] = 1; U[5, 20] = 1
    V[1, 20] = 1; V[3, 20] = 1; V[5, 20] = 1
    # P22 = (c+e+g)*(y+u+p)
    U[2, 21] = 1; U[4, 21] = 1; U[6, 21] = 1
    V[2, 21] = 1; V[4, 21] = 1; V[6, 21] = 1
    # P23 = (a+b+c+d+e+f+g+h+i)*(w+x+y+z+u+v+p+q+r)
    U[0, 22] = 1; U[1, 22] = 1; U[2, 22] = 1; U[3, 22] = 1
    U[4, 22] = 1; U[5, 22] = 1; U[6, 22] = 1; U[7, 22] = 1; U[8, 22] = 1
    V[0, 22] = 1; V[1, 22] = 1; V[2, 22] = 1; V[3, 22] = 1
    V[4, 22] = 1; V[5, 22] = 1; V[6, 22] = 1; V[7, 22] = 1; V[8, 22] = 1
    return U, V


def _solve_W(U, V, n, m, p, r):
    """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r) such that
    for all i,j,k: C_ik = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
    Equivalently sum_r W_{ki}^r U_{ij}^r = (A B)_{ik} as a bilinear identity.
    For each output position (i,k) we need: for all (j), the row of A (index j)
    times ... Actually the tensor constraint is:
        T_{i,j,k} = sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}  = delta? No.
    The matrix multiplication tensor has T_{i,j,k} = 1 for all i,j,k (C_ik = sum_j A_ij B_jk).
    So the decomposition must satisfy: for every (i,j,k),
        sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r} = 1.
    This is a system in W: for fixed (i,k), sum_r W_{k,i,r} * (U_{i,j,r} V_{j,k,r}) = 1 for all j.
    So for each (i,k), we have m equations in r unknowns W_{k,i,0..r-1}.
    """
    W = np.zeros((p * n, r))
    for i in range(n):
        for k in range(p):
            # Build linear system: for each j, sum_r W_{k,i,r} * (U_{i,j,r} V_{j,k,r}) = 1
            # m equations, r unknowns -> least squares (r >= m expected, use lsq)
            M = np.zeros((m, r))
            for j in range(m):
                Urow = U[i * m + j, :]
                Vrow = V[j * p + k, :]
                M[j, :] = Urow * Vrow
            rhs = np.ones(m)
            sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
            W[k * n + i, :] = sol
    return W


def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions.
    Square blocks inferred from shapes.
    """
    import math
    n1 = int(math.isqrt(Ua.shape[0])); m1 = int(math.isqrt(Va.shape[0])); p1 = int(math.isqrt(Wa.shape[0]))
    n2 = int(math.isqrt(Ub.shape[0])); m2 = int(math.isqrt(Vb.shape[0])); p2 = int(math.isqrt(Wb.shape[0]))
    r1 = Ua.shape[1]; r2 = Ub.shape[1]
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


def _verify(U, V, W, n, m, p, iters=8):
    """Verify tensor decomposition: sum_r U_{ij}^r V_{jk}^r W_{ki}^r == 1 for all i,j,k."""
    rng = np.random.default_rng(0)
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    for _ in range(iters):
        A = rng.integers(-3, 4, (n, m)).astype(float)
        B = rng.integers(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            scalar = np.dot(U[:, r], A.flatten()) * np.dot(V[:, r], B.flatten())
            if scalar != 0:
                C_computed += scalar * W[:, r].reshape(n, p)
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    # Fallback always available
    def fallback():
        U, V, W = _schoolbook(6, 6, 6)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if _check_time():
        return fallback()

    # Best known so far: Strassen(7) x Laderman(23) = 161
    best = None
    try:
        U3, V3 = _laderman_UV()
        W3 = _solve_W(U3, V3, 3, 3, 3, 23)
        if _verify(U3, V3, W3, 3, 3, 3):
            U2, V2, W2 = _strassen_2x2()
            if _verify(U2, V2, W2, 2, 2, 2):
                U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
                if _verify(U, V, W, 6, 6, 6):
                    best = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    if best is not None and not _check_time():
        # Try to see if we can do even better with a direct rank-? search is out of scope.
        return {"U": best["U"].tolist(), "V": best["V"].tolist(), "W": best["W"].tolist()}

    # Second fallback: Strassen(7) x schoolbook 3x3 (27) = 189
    try:
        U3s, V3s, W3s = _schoolbook(3, 3, 3)
        U2, V2, W2 = _strassen_2x2()
        U, V, W = _kron_decompose(U3s, V3s, W3s, U2, V2, W2)
        if _verify(U, V, W, 6, 6, 6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    return fallback()
