import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 140

    def get_rank50():
        U = np.zeros((n * m, 50))
        V = np.zeros((m * p, 50))
        W = np.zeros((p * n, 50))
        col = 0

        def add_strassen_block(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0*m+j0, col] += 1
            U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M2: (c+d)e
            U[(i0+1)*m+j0, col] += 1
            U[(i0+1)*m+(j0+1), col] += 1
            V[j0*p+k0, col] += 1
            W[k0*n+(i0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] -= 1
            col += 1
            # M3: a(f-h)
            U[i0*m+j0, col] += 1
            V[j0*p+(k0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] -= 1
            W[(k0+1)*n+i0, col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M4: d(g-e)
            U[(i0+1)*m+(j0+1), col] += 1
            V[(j0+1)*p+k0, col] += 1
            V[j0*p+k0, col] -= 1
            W[k0*n+i0, col] += 1
            W[k0*n+(i0+1), col] += 1
            col += 1
            # M5: (a+b)h
            U[i0*m+j0, col] += 1
            U[i0*m+(j0+1), col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] -= 1
            W[(k0+1)*n+i0, col] += 1
            col += 1
            # M6: (c-a)(e+f)
            U[(i0+1)*m+j0, col] += 1
            U[i0*m+j0, col] -= 1
            V[j0*p+k0, col] += 1
            V[j0*p+(k0+1), col] += 1
            W[(k0+1)*n+(i0+1), col] += 1
            col += 1
            # M7: (b-d)(g+h)
            U[i0*m+(j0+1), col] += 1
            U[(i0+1)*m+(j0+1), col] -= 1
            V[(j0+1)*p+k0, col] += 1
            V[(j0+1)*p+(k0+1), col] += 1
            W[k0*n+i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_block(0, j0, k0)

        for j in range(m):
            for i in range(n):
                U[i*m+j, col] = 1
                V[j*p+6, col] = 1
                W[6*n+i, col] = 1
                col += 1

        return U[:, :col], V[:, :col], W[:, :col]

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    U, V, W = get_rank50()
    if time.time() < deadline and verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
