import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _build_laderman_23():
    """Return U (9,23), V (9,23), W (9,23) for rank-23 decomposition of 3x3*3x3.
    U[i*3+j], V[j*3+k], W[k*3+i]. Returns None if validation fails."""
    def mask(rows):
        m = 0
        for a, b in rows:
            m |= 1 << (a*3 + b)
        return m

    terms = [
        (mask([(0,0)]), mask([(0,0)])),
        (mask([(0,2)]), mask([(2,0)])),
        (mask([(2,0)]), mask([(0,2)])),
        (mask([(0,1),(1,1)]), mask([(1,0),(2,0)])),
        (mask([(0,1),(2,1)]), mask([(1,2),(2,2)])),
        (mask([(0,0),(1,1)]), mask([(2,1),(2,2)])),
        (mask([(1,0),(1,1)]), mask([(0,1),(2,1)])),
        (mask([(0,0),(0,1)]), mask([(1,0),(2,0)])),
        (mask([(0,0),(1,0)]), mask([(1,1),(2,1)])),
        (mask([(0,2),(2,2)]), mask([(0,0),(1,0)])),
        (mask([(0,2),(1,2)]), mask([(1,0),(2,0)])),
        (mask([(0,2),(1,2)]), mask([(2,0),(2,1)])),
        (mask([(0,0),(0,2)]), mask([(0,1),(2,1)])),
        (mask([(0,1),(1,0)]), mask([(1,0),(2,0)])),
        (mask([(0,2),(1,2)]), mask([(1,0),(1,2)])),
        (mask([(0,1),(1,2)]), mask([(0,0),(1,0)])),
        (mask([(0,0),(2,0)]), mask([(0,1),(2,1)])),
        (mask([(1,0),(1,1)]), mask([(0,0),(0,1)])),
        (mask([(1,2),(2,2)]), mask([(0,0),(0,2)])),
        (mask([(1,0),(2,0)]), mask([(0,1),(2,1)])),
        (mask([(0,1),(2,1)]), mask([(0,2),(2,2)])),
        (mask([(1,2),(2,2)]), mask([(0,1),(2,1)])),
        (mask([(0,1),(0,2)]), mask([(2,0),(2,1)])),
    ]

    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    for c in range(23):
        um, vm = terms[c]
        for bit in range(9):
            if um & (1 << bit):
                U[bit, c] = 1.0
            if vm & (1 << bit):
                V[bit, c] = 1.0

    # Build P matrix: P[c, i*9+k] = sum_j U[i,j,c]*V[j,k,c]
    P = np.zeros((23, 81))
    for c in range(23):
        for i in range(3):
            for k in range(3):
                s = 0.0
                for j in range(3):
                    s += U[i*3+j, c] * V[j*3+k, c]
                P[c, i*9+k] = s

    W = np.zeros((9, 23))
    for i0 in range(3):
        for k0 in range(3):
            b = np.zeros(81)
            b[i0*9+k0] = 1.0
            x, *_ = np.linalg.lstsq(P, b, rcond=None)
            if not np.allclose(P @ x, b, atol=1e-12):
                return None
            W[k0*3+i0, :] = x

    # Validate: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = 0.0
                for c in range(23):
                    r += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                if abs(r - 1.0) > 1e-10:
                    return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    start = time.time()
    deadline = start + 120

    def schoolbook():
        rank = n * m * p
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    result = _build_laderman_23()
    if result is not None:
        U23, V23, W23 = result
        # Combine with schoolbook for k=3..6
        rank = 23 + 4 * 3 * 3  # 23 + 36 = 59
        U = np.zeros((num_u, rank))
        V = np.zeros((num_v, rank))
        W = np.zeros((num_w, rank))
        U[:, :23] = U23
        V[:9, :23] = V23
        W[:9, :23] = W23
        col = 23
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        # Validate
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = 0.0
                    for c in range(rank):
                        r += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(r - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if time.time() > deadline - 30:
            return schoolbook()

    return schoolbook()
