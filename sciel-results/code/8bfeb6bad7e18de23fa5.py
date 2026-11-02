import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x11 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks (rank 7 each) for 5 blocks,
    and schoolbook for the remaining 2x1 block (rank 4).
    Total rank = 5*7 + 4 = 39.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    n, m, p = 2, 2, 11
    rank = 39

    # Initialize factor matrices as lists of lists for speed
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39

    r = 0

    # Process 5 blocks of 2 columns each (k=0..9) using Strassen
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n     # C[0, k0]
        w_c10 = k0 * n + 1 # C[1, k0]
        w_c01 = k1 * n     # C[0, k1]
        w_c11 = k1 * n + 1 # C[1, k1]

        # U indices: A[i, j] -> U[i*m + j]
        # U[0]=A[0,0], U[1]=A[0,1], U[2]=A[1,0], U[3]=A[1,1]

        # Strassen's 7 products for 2x2 * 2x2:
        # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]

        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook for 2x2 * 2x1
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = p + k2  # B[1, 10]
        w_c02 = k2 * n      # C[0, 10]
        w_c12 = k2 * n + 1  # C[1, 10]

        # C[0,10] = a*B[0,10] + b*B[1,10]
        # C[1,10] = c*B[0,10] + d*B[1,10]

        # a * B[0, 10]
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # b * B[1, 10]
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # c * B[0, 10]
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1

        # d * B[1, 10]
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1

    # Convert to numpy arrays
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    return {"U": U, "V": V, "W": W}
