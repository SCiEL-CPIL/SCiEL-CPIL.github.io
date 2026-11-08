import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros(n * m, dtype=float)
        V = np.zeros(m * p, dtype=float)
        W = np.zeros(p * n, dtype=float)
        for (i, j), val in u_vals.items():
            U[i * m + j] = val
        for (j, k), val in v_vals.items():
            V[j * p + k] = val
        for (k, i), val in w_vals.items():
            W[k * n + i] = val
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        add_term({(i0, j0): 1, (i1, j1): 1}, {(j0, k0): 1, (j1, k1): 1}, {(k0, i0): 1, (k1, i1): 1})
        add_term({(i1, j0): 1, (i1, j1): 1}, {(j0, k0): 1}, {(k0, i1): 1, (k1, i1): -1})
        add_term({(i0, j0): 1}, {(j0, k1): 1, (j1, k1): -1}, {(k1, i0): 1, (k1, i1): 1})
        add_term({(i1, j1): 1}, {(j1, k0): 1, (j0, k0): -1}, {(k0, i0): 1, (k0, i1): 1})
        add_term({(i0, j0): 1, (i0, j1): 1}, {(j1, k1): 1}, {(k0, i0): -1, (k1, i0): 1})
        add_term({(i1, j0): 1, (i0, j0): -1}, {(j0, k0): 1, (j0, k1): 1}, {(k1, i1): 1})
        add_term({(i0, j1): 1, (i1, j1): -1}, {(j1, k0): 1, (j1, k1): 1}, {(k0, i0): 1})

    def mm_2x1_1x2(i0, i1, j, k0, k1):
        # A is 2x1 (rows i0,i1; col j), B is 1x2 (row j; cols k0,k1)
        # C[i,k] = A[i,j]*B[j,k]
        for i in [i0, i1]:
            for k in [k0, k1]:
                add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mm_1x2_2x1(i, j0, j1, k):
        # A is 1x2 (row i; cols j0,j1), B is 2x1 (rows j0,j1; col k)
        # C[i,k] = sum_j A[i,j]*B[j,k]
        for j in [j0, j1]:
            add_term({(i, j): 1}, {(j, k): 1}, {(k, i): 1})

    def mm_2x3_3x2(i0, i1, j0, j1, j2, k0, k1):
        # A is 2x3 (rows i0,i1; cols j0,j1,j2), B is 3x2 (rows j0,j1,j2; cols k0,k1)
        # Split inner dim into 2+1:
        # A[:, j0:j1] * B[j0:j1, :] -> 2x2 * 2x2 (7 terms)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A[:, j2] * B[j2, :] -> 2x1 * 1x2 (4 terms)
        mm_2x1_1x2(i0, i1, j2, k0, k1)

    def mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2):
        # A is 2x2 (rows i0,i1; cols j0,j1), B is 2x3 (rows j0,j1; cols k0,k1,k2)
        # Split inner dim of B into 2+1:
        # A * B[:, k0:k1] -> 2x2 * 2x2 (7 terms)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A * B[:, k2] -> 2x2 * 2x1 (4 terms)
        for i in [i0, i1]:
            for j in [j0, j1]:
                add_term({(i, j): 1}, {(j, k2): 1}, {(k2, i): 1})

    def mm_2x3_3x3(i0, i1, j0, j1, j2, k0, k1, k2):
        # A is 2x3 (rows i0,i1; cols j0,j1,j2), B is 3x3 (rows j0,j1,j2; cols k0,k1,k2)
        # Split inner dim into 2+1:
        # A[:, j0:j1] * B[j0:j1, :] -> 2x2 * 2x3 (11 terms)
        mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        # A[:, j2] * B[j2, :] -> 2x1 * 1x3 (6 terms)
        for i in [i0, i1]:
            for k in [k0, k1, k2]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1):
        # A is 3x2 (rows i0,i1,i2; cols j0,j1), B is 2x2 (rows j0,j1; cols k0,k1)
        # Split rows of A into 2+1:
        # A[i0:i1, :] * B -> 2x2 * 2x2 (7 terms)
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        # A[i2, :] * B -> 1x2 * 2x2 (4 terms)
        for j in [j0, j1]:
            for k in [k0, k1]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    def mm_3x3_3x2(i0, i1, i2, j0, j1, j2, k0, k1):
        # A is 3x3 (rows i0,i1,i2; cols j0,j1,j2), B is 3x2 (rows j0,j1,j2; cols k0,k1)
        # Split inner dim into 2+1:
        # A[:, j0:j1] * B[j0:j1, :] -> 3x2 * 2x2 (11 terms)
        mm_3x2_2x2(i0, i1, i2, j0, j1, k0, k1)
        # A[:, j2] * B[j2, :] -> 3x1 * 1x2 (6 terms)
        for i in [i0, i1, i2]:
            for k in [k0, k1]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x3_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # A is 3x3, B is 3x3
        # Split inner dim into 2+1:
        # A[:, j0:j1] * B[j0:j1, :] -> 3x2 * 2x3 (11 terms)
        mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2)
        # A[:, j2] * B[j2, :] -> 3x1 * 1x3 (9 terms)
        for i in [i0, i1, i2]:
            for k in [k0, k1, k2]:
                add_term({(i, j2): 1}, {(j2, k): 1}, {(k, i): 1})

    def mm_3x2_2x3(i0, i1, i2, j0, j1, k0, k1, k2):
        # A is 3x2 (rows i0,i1,i2; cols j0,j1), B is 2x3 (rows j0,j1; cols k0,k1,k2)
        # Split rows of A into 2+1:
        # A[i0:i1, :] * B -> 2x2 * 2x3 (11 terms)
        mm_2x2_2x3(i0, i1, j0, j1, k0, k1, k2)
        # A[i2, :] * B -> 1x2 * 2x3 (6 terms)
        for j in [j0, j1]:
            for k in [k0, k1, k2]:
                add_term({(i2, j): 1}, {(j, k): 1}, {(k, i2): 1})

    # Block decomposition: split indices into {0,1} and {2,3,4}
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # C[0:2,2:5] = A[0:2,0:2]*B[0:2,2:5] + A[0:2,2:5]*B[2:5,2:5]
    # C[2:5,0:2] = A[2:5,0:2]*B[0:2,0:2] + A[2:5,2:5]*B[2:5,0:2]
    # C[2:5,2:5] = A[2:5,0:2]*B[0:2,2:5] + A[2:5,2:5]*B[2:5,2:5]

    # C[0:2,0:2]
    strassen_2x2(0, 1, 0, 1, 0, 1)  # A[0:2,0:2]*B[0:2,0:2]
    mm_2x3_3x2(0, 1, 2, 3, 4, 0, 1)  # A[0:2,2:5]*B[2:5,0:2]

    # C[0:2,2:5]
    mm_2x2_2x3(0, 1, 0, 1, 2, 3, 4)  # A[0:2,0:2]*B[0:2,2:5]
    mm_2x3_3x3(0, 1, 2, 3, 4, 2, 3, 4)  # A[0:2,2:5]*B[2:5,2:5]

    # C[2:5,0:2]
    mm_3x2_2x2(2, 3, 4, 0, 1, 0, 1)  # A[2:5,0:2]*B[0:2,0:2]
    mm_3x3_3x2(2, 3, 4, 2, 3, 4, 0, 1)  # A[2:5,2:5]*B[2:5,0:2]

    # C[2:5,2:5]
    mm_3x2_2x3(2, 3, 4, 0, 1, 2, 3, 4)  # A[2:5,0:2]*B[0:2,2:5]
    mm_3x3_3x3(2, 3, 4, 2, 3, 4, 2, 3, 4)  # A[2:5,2:5]*B[2:5,2:5]

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]

    # Verify
    if time.time() - t0 < deadline - 5:
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            U = np.zeros((n*m, n*m*p))
            V = np.zeros((m*p, n*m*p))
            W = np.zeros((p*n, n*m*p))
            for i2 in range(n):
                for j2 in range(m):
                    for k2 in range(p):
                        idx = i2*m*p + j2*p + k2
                        U[i2*m+j2, idx] = 1
                        V[j2*p+k2, idx] = 1
                        W[k2*n+i2, idx] = 1
            rank = n*m*p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
