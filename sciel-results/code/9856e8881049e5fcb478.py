import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    c = 0

    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1 = (a00+a11)*(b00+b11)
        U[0*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1

        # m2 = (a10+a11)*b00
        U[1*m + j_off, c] = 1; U[1*m + (j_off+1), c] = 1
        V[j_off*p + k_off, c] = 1
        W[k_off*n + 1, c] = 1; W[(k_off+1)*n + 1, c] = -1
        c += 1

        # m3 = a00*(b01-b11)
        U[0*m + j_off, c] = 1
        V[j_off*p + (k_off+1), c] = 1; V[(j_off+1)*p + (k_off+1), c] = -1
        W[(k_off+1)*n + 0, c] = 1; W[(k_off+1)*n + 1, c] = 1
        c += 1

        # m4 = a11*(b10-b00)
        U[1*m + (j_off+1), c] = 1
        V[(j_off+1)*p + k_off, c] = 1; V[j_off*p + k_off, c] = -1
        W[k_off*n + 0, c] = 1; W[k_off*n + 1, c] = 1
        c += 1

        # m5 = (a00+a01)*b11
        U[0*m + j_off, c] = 1; U[0*m + (j_off+1), c] = 1
        V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = -1; W[(k_off+1)*n + 0, c] = 1
        c += 1

        # m6 = (a10-a00)*(b00+b01)
        U[1*m + j_off, c] = 1; U[0*m + j_off, c] = -1
        V[j_off*p + k_off, c] = 1; V[j_off*p + (k_off+1), c] = 1
        W[(k_off+1)*n + 1, c] = 1
        c += 1

        # m7 = (a01-a11)*(b10+b11)
        U[0*m + (j_off+1), c] = 1; U[1*m + (j_off+1), c] = -1
        V[(j_off+1)*p + k_off, c] = 1; V[(j_off+1)*p + (k_off+1), c] = 1
        W[k_off*n + 0, c] = 1
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)

    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)

    # Remaining 14 schoolbook terms (j=2, all k; plus j=0,1 with k=4)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1

    assert c == rank, f"Expected rank {rank}, got {c}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
