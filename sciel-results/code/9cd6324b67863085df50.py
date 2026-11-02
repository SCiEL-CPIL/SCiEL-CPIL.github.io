import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start = time.time()
    deadline = start + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def fallback():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Strassen rank-7 on each independent 2x2 block
    # A is 2x4: 2 column-blocks of 2x2
    # B is 4x8: 4 row-blocks of 4x8, each with 4 column-blocks of 2x2
    # But A's columns map to B's rows. Since A has 4 columns, it produces 4
    # independent 2x2 block multiplications per B-row-block? No.
    # Actually: A[i,j] for i in 0,1 and j in 0..3. B[j,k] for j in 0..3, k in 0..7.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Group B into 4 row-groups of 2 rows each. For each row-group (u), 
    # A's columns 2u, 2u+1 pair with B's rows 2u, 2u+1.
    # This gives 2 u-groups, each producing a 2x8 sub-product using a 2x2 A-block and a 2x8 B-block.
    # The 2x8 B-block has 4 sub-columns of 2, so 4 independent 2x2 Strassen per u.
    # Total: 2 u's * 4 col-blocks * 7 = 56.

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen formulas for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    col_idx = 0
    for u in range(m // 2):   # 2 A-row-blocks (but A has 4 columns -> 2 col-blocks of 2)
        for v in range(p // 2):  # 4 B col-blocks
            base = col_idx
            col_idx += 7

            # A block (2x2): rows 0,1; cols 2u, 2u+1
            a00 = 0 * m + 2 * u
            a01 = 0 * m + 2 * u + 1
            a10 = 1 * m + 2 * u
            a11 = 1 * m + 2 * u + 1

            # B block (2x2): rows 2u, 2u+1; cols 2v, 2v+1
            b00 = (2 * u) * p + 2 * v
            b01 = (2 * u) * p + 2 * v + 1
            b10 = (2 * u + 1) * p + 2 * v
            b11 = (2 * u + 1) * p + 2 * v + 1

            # C block (2x2): rows 0,1 (output rows); cols 2v, 2v+1
            # W index = k*n + i
            c00 = (2 * v) * n + 0
            c01 = (2 * v + 1) * n + 0
            c10 = (2 * v) * n + 1
            c11 = (2 * v + 1) * n + 1

            # M1 = (A00+A11)*(B00+B11)
            U[a00, base] += 1; U[a11, base] += 1
            V[b00, base] += 1; V[b11, base] += 1
            W[c00, base] += 1; W[c11, base] += 1

            # M2 = (A10+A11)*B00
            U[a10, base+1] += 1; U[a11, base+1] += 1
            V[b00, base+1] += 1
            W[c10, base+1] += 1; W[c11, base+1] -= 1

            # M3 = A00*(B01-B11)
            U[a00, base+2] += 1
            V[b01, base+2] += 1; V[b11, base+2] -= 1
            W[c01, base+2] += 1; W[c11, base+2] += 1

            # M4 = A11*(B10-B00)
            U[a11, base+3] += 1
            V[b10, base+3] += 1; V[b00, base+3] -= 1
            W[c00, base+3] += 1; W[c10, base+3] += 1

            # M5 = (A00+A01)*B11
            U[a00, base+4] += 1; U[a01, base+4] += 1
            V[b11, base+4] += 1
            W[c00, base+4] -= 1; W[c01, base+4] += 1

            # M6 = (A10-A00)*(B00+B01)
            U[a10, base+5] += 1; U[a00, base+5] -= 1
            V[b00, base+5] += 1; V[b01, base+5] += 1
            W[c11, base+5] += 1

            # M7 = (A01-A11)*(B10+B11)
            U[a01, base+6] += 1; U[a11, base+6] -= 1
            V[b10, base+6] += 1; V[b11, base+6] += 1
            W[c00, base+6] += 1

    # Verify correctness
    valid = True
    if time.time() < deadline:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(U[i*m+j, :] * W[k*n+i, :], V[j*p+k, :]))
                    if abs(s - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    else:
        valid = False

    if not valid:
        U, V, W = fallback()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
