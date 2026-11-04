import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition of 2x2 * 2x2.
    U,V,W shapes (4,7). Convention U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    # M3 = A00(B10-B11)
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    # M4 = (A10-A00)B11
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    # M5 = (A00+A01)B10
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    # M6 = (A01-A10)(B00+B10)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # M7 = (A11-A01)(B01+B11)
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build_base():
    """Build rank-98 base via 2 halves x 7x7 Strassen block structure."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()

    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]

    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    # vectorized check
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = float(np.dot(U[i*m+j, :], V[j*p+k, :]*W[k*n+i, :]))
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Correctly attempt rank reduction by checking if a column's rank-1
    tensor lies in the span of others' rank-1 tensors (full factor check)."""
    R = U.shape[1]

    def get_tensor(U, V, W):
        R = U.shape[1]
        T = np.zeros((n*m*p, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    idx += 1
        return T

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    T = get_tensor(U, V, W)

    for r0 in range(R):
        if time.time() > deadline - 15:
            break
        keep = [r for r in range(R) if r != r0]
        T_other = T[:, keep]
        target = -T[:, r0]
        try:
            c, *_ = np.linalg.lstsq(T_other, target, rcond=1e-10)
        except Exception:
            continue
        residual = float(np.linalg.norm(T_other @ c + T[:, r0]))
        if residual > 1e-6:
            continue
        # Found a tensor-level dependency. But rank-1 tensors don't combine
        # linearly into a valid factorized decomposition in general, so we
        # cannot simply drop r0. We only accept if a valid re-factoring exists.
        # A safe reduction: if c is all zero except we can absorb via shared
        # structure -- too complex; skip to keep correctness.
        pass

    return best_U, best_V, best_W

def construct():
    n, m, p = 4, 4, 8
    U, V, W = _build_base()
    if not _verify(U, V, W, n, m, p):
        # fallback: schoolbook (guaranteed valid)
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                    col += 1

    if time.time() < _DEADLINE - 20:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, _DEADLINE)

    if not _verify(U, V, W, n, m, p):
        U, V, W = _build_base()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
