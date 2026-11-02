import time
import numpy as np

_DEADLINE = time.time() + 145.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    n, m, p = 2, 3, 5

    def _build_rank28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def schoolbook(i0, i1, j0, j1, k0, k1):
            nonlocal col
            for ii in range(i0, i1 + 1):
                for jj in range(j0, j1 + 1):
                    for kk in range(k0, k1 + 1):
                        U[ii * m + jj, col] = 1
                        V[jj * p + kk, col] = 1
                        W[kk * n + ii, col] = 1
                        col += 1

        def strassen(i0, j0, k0):
            nonlocal col
            # M1 = (A00+A11)*(B00+B11)
            U[i0*m+j0, col] = 1
            U[(i0+1)*m+(j0+1), col] = 1
            V[j0*p+k0, col] = 1
            V[(j0+1)*p+(k0+1), col] = 1
            W[k0*n+i0, col] = 1
            W[(k0+1)*n+(i0+1), col] = 1
            col += 1
            # M2 = (A10+A11)*B00
            U[(i0+1)*m+j0, col] = 1
            U[(i0+1)*m+(j0+1), col] = 1
            V[j0*p+k0, col] = 1
            W[k0*n+(i0+1), col] = 1
            W[(k0+1)*n+(i0+1), col] = -1
            col += 1
            # M3 = A00*(B11-B01)
            U[i0*m+j0, col] = 1
            V[(j0+1)*p+(k0+1), col] = 1
            V[(j0+1)*p+k0, col] = -1
            W[(k0+1)*n+i0, col] = 1
            W[(k0+1)*n+(i0+1), col] = 1
            col += 1
            # M4 = A11*(B00-B10)
            U[(i0+1)*m+(j0+1), col] = 1
            V[(j0+1)*p+k0, col] = 1
            V[j0*p+k0, col] = -1
            W[k0*n+i0, col] = 1
            W[k0*n+(i0+1), col] = 1
            col += 1
            # M5 = (A00+A01)*B11
            U[i0*m+j0, col] = 1
            U[i0*m+(j0+1), col] = 1
            V[(j0+1)*p+(k0+1), col] = 1
            W[k0*n+i0, col] = -1
            W[(k0+1)*n+i0, col] = 1
            col += 1
            # M6 = (A10-A00)*(B00+B01)
            U[(i0+1)*m+j0, col] = 1
            U[i0*m+j0, col] = -1
            V[j0*p+k0, col] = 1
            V[j0*p+(k0+1), col] = 1
            W[(k0+1)*n+(i0+1), col] = 1
            col += 1
            # M7 = (A01-A11)*(B10+B11)
            U[i0*m+(j0+1), col] = 1
            U[(i0+1)*m+(j0+1), col] = -1
            V[(j0+1)*p+k0, col] = 1
            V[(j0+1)*p+(k0+1), col] = 1
            W[k0*n+i0, col] = 1
            col += 1

        # Block A: j=0, all k -> 2x1x5 schoolbook = rank 10
        schoolbook(0, 1, 0, 0, 0, 4)
        # Block B: j in {1,2}, k in {0,1} -> 2x2x2 Strassen = rank 7
        strassen(0, 1, 0)
        # Block C: j in {1,2}, k=2 -> 2x2x1 schoolbook = rank 4
        schoolbook(0, 1, 1, 2, 2, 2)
        # Block D: j in {1,2}, k in {3,4} -> 2x2x2 Strassen = rank 7
        strassen(0, 1, 3)

        assert col == rank, f"col={col}, expected {rank}"
        return U, V, W

    def _verify(U, V, W):
        n, m, p = 2, 3, 5
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    U, V, W = _build_rank28()
    assert _verify(U, V, W), "Rank-28 decomposition failed verification!"

    return {"U": U, "V": V, "W": W}
