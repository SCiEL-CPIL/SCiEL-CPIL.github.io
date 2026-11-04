import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Build a rank-28 decomposition using Strassen on 2x2 blocks.
    # The multiplication tensor T(i,j,k) = 1 for all i,j,k.
    # We split the 3 intermediate columns into [0:2] and [2:3].
    # Split the 5 output columns into [0:2], [2:4], [4:5].
    # Use Strassen (rank 7) for 2x2*2x2 blocks, schoolbook for the rest.

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 0, 1)

    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 2, 3)

    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [0, 1], [4])

    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [0, 1])

    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4]  (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [2, 3])

    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5]  (Schoolbook, rank 2)
    schoolbook_block([0, 1], [2], [4])

    # Verify: reconstruct the tensor and check all entries equal 1
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        # contribution to C[i,k] from this rank-1 term:
        # sum_j uc[i,j] * vc[j,k] * wc[k,i] ... but W is indexed [k*n+i, c]
        # so W_c[k, i] = wc[k, i]
        # C[i,k] += sum_j U_c[i,j] V_c[j,k] * W_c[k,i]
        # We need: sum_c U_c[i,j] V_c[j,k] W_c[k,i] = 1 for all i,j,k
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += uc[i, j] * vc[j, k] * wc[k, i]
                tensor[i, j, k] = tensor[i, j, k]  # placeholder, restructure below
    # Redo verification properly
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback: full schoolbook rank 30
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
