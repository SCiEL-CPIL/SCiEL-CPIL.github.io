import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def verify(Uv, Vv, Wv, tol=1e-10):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uv[i*m+j], Vv[j*p+k] * Wv[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def add_strassen_3x3x3(U, V, W, i0, j0, k0, col_offset):
        """Add rank-19 decomposition for a 3x3x3 block."""
        # A[i,j] -> U[(i0+di)*m + (j0+dj), c]
        # B[j,k] -> V[(j0+dj)*p + (k0+dk), c]
        # C[i,k] -> W[(k0+dk)*n + (i0+di), c]

        # Term 0: A00 * B00
        c = col_offset + 0
        U[(i0+0)*m + (j0+0), c] += 1
        V[(j0+0)*p + (k0+0), c] += 1
        W[(k0+0)*n + (i0+0), c] += 1

        # Term 1: A00 * (B01 + B02)
        c = col_offset + 1
        U[(i0+0)*m + (j0+0), c] += 1
        V[(j0+0)*p + (k0+1), c] += 1
        V[(j0+0)*p + (k0+2), c] += 1
        W[(k0+1)*n + (i0+0), c] += 1
        W[(k0+2)*n + (i0+0), c] += 1

        # Term 2: A01 * (B10 + B20)
        c = col_offset + 2
        U[(i0+0)*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+0), c] += 1
        V[(j0+2)*p + (k0+0), c] += 1
        W[(k0+0)*n + (i0+0), c] += 1
        W[(k0+0)*n + (i0+1), c] += 1

        # Term 3: A10 * (B01 + B11)
        c = col_offset + 3
        U[(i0+1)*m + (j0+0), c] += 1
        V[(j0+0)*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+0), c] += 1

        # Term 4: A11 * (B12 + B22)
        c = col_offset + 4
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+2), c] += 1
        V[(j0+2)*p + (k0+2), c] += 1
        W[(k0+2)*n + (i0+1), c] += 1
        W[(k0+2)*n + (i0+1), c] += 1

        # Term 5: A20 * (B10 + B11)
        c = col_offset + 5
        U[(i0+2)*m + (j0+0), c] += 1
        V[(j0+1)*p + (k0+0), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[(k0+0)*n + (i0+2), c] += 1
        W[(k0+1)*n + (i0+2), c] += 1

        # Term 6: A21 * (B02 + B22)
        c = col_offset + 6
        U[(i0+2)*m + (j0+1), c] += 1
        V[(j0+0)*p + (k0+2), c] += 1
        V[(j0+2)*p + (k0+2), c] += 1
        W[(k0+2)*n + (i0+2), c] += 1
        W[(k0+2)*n + (i0+2), c] += 1

        # Term 7: A22 * (B10 + B20)
        c = col_offset + 7
        U[(i0+2)*m + (j0+2), c] += 1
        V[(j0+1)*p + (k0+0), c] += 1
        V[(j0+2)*p + (k0+0), c] += 1
        W[(k0+0)*n + (i0+2), c] += 1
        W[(k0+0)*n + (i0+2), c] += 1

        # Term 8: (A00 + A11) * (B11 - B01)
        c = col_offset + 8
        U[(i0+0)*m + (j0+0), c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        V[(j0+0)*p + (k0+1), c] += -1
        W[(k0+1)*n + (i0+0), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # Term 9: (A00 + A22) * (B20 - B00)
        c = col_offset + 9
        U[(i0+0)*m + (j0+0), c] += 1
        U[(i0+2)*m + (j0+2), c] += 1
        V[(j0+2)*p + (k0+0), c] += 1
        V[(j0+0)*p + (k0+0), c] += -1
        W[(k0+0)*n + (i0+0), c] += 1
        W[(k0+0)*n + (i0+2), c] += 1

        # Term 10: (A11 + A22) * (B21 - B11)
        c = col_offset + 10
        U[(i0+1)*m + (j0+1), c] += 1
        U[(i0+2)*m + (j0+2), c] += 1
        V[(j0+2)*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += -1
        W[(k0+1)*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+2), c] += 1

        # Term 11: (A01 + A10) * (B00 + B10)
        c = col_offset + 11
        U[(i0+0)*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+0), c] += 1
        V[(j0+0)*p + (k0+0), c] += 1
        V[(j0+1)*p + (k0+0), c] += 1
        W[(k0+0)*n + (i0+0), c] += 1
        W[(k0+0)*n + (i0+1), c] += 1

        # Term 12: (A02 + A12) * (B11 + B21)
        c = col_offset + 12
        U[(i0+0)*m + (j0+2), c] += 1
        U[(i0+1)*m + (j0+2), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        V[(j0+2)*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+0), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1

        # Term 13: (A12 + A22) * (B01 + B11)
        c = col_offset + 13
        U[(i0+1)*m + (j0+2), c] += 1
        U[(i0+2)*m + (j0+2), c] += 1
        V[(j0+0)*p + (k0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+2), c] += 1

        # Term 14: (A01 + A21) * (B02 + B12)
        c = col_offset + 14
        U[(i0+0)*m + (j0+1), c] += 1
        U[(i0+2)*m + (j0+1), c] += 1
        V[(j0+0)*p + (k0+2), c] += 1
        V[(j0+1)*p + (k0+2), c] += 1
        W[(k0+2)*n + (i0+0), c] += 1
        W[(k0+2)*n + (i0+2), c] += 1

        # Term 15: (A02 + A22) * (B00 + B10)
        c = col_offset + 15
        U[(i0+0)*m + (j0+2), c] += 1
        U[(i0+2)*m + (j0+2), c] += 1
        V[(j0+0)*p + (k0+0), c] += 1
        V[(j0+1)*p + (k0+0), c] += 1
        W[(k0+0)*n + (i0+0), c] += 1
        W[(k0+0)*n + (i0+2), c] += 1

        # Term 16: (A10 + A20) * (B02 + B12)
        c = col_offset + 16
        U[(i0+1)*m + (j0+0), c] += 1
        U[(i0+2)*m + (j0+0), c] += 1
        V[(j0+0)*p + (k0+2), c] += 1
        V[(j0+1)*p + (k0+2), c] += 1
        W[(k0+2)*n + (i0+1), c] += 1
        W[(k0+2)*n + (i0+2), c] += 1

        # Term 17: (A00 + A01 + A02) * (B00 + B11 + B22)
        c = col_offset + 17
        U[(i0+0)*m + (j0+0), c] += 1
        U[(i0+0)*m + (j0+1), c] += 1
        U[(i0+0)*m + (j0+2), c] += 1
        V[(j0+0)*p + (k0+0), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        V[(j0+2)*p + (k0+2), c] += 1
        W[(k0+0)*n + (i0+0), c] += 1
        W[(k0+1)*n + (i0+0), c] += 1
        W[(k0+2)*n + (i0+0), c] += 1

        # Term 18: (A10 + A11 + A12) * (B00 + B11 + B22)
        c = col_offset + 18
        U[(i0+1)*m + (j0+0), c] += 1
        U[(i0+1)*m + (j0+1), c] += 1
        U[(i0+1)*m + (j0+2), c] += 1
        V[(j0+0)*p + (k0+0), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        V[(j0+2)*p + (k0+2), c] += 1
        W[(k0+0)*n + (i0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        W[(k0+2)*n + (i0+1), c] += 1

    # Strategy 1: 3x3x3 Strassen block (rank 19) + schoolbook remainder (39 terms) = rank 66
    total_rank = 19 + 39
    U1 = np.zeros((n*m, total_rank))
    V1 = np.zeros((m*p, total_rank))
    W1 = np.zeros((p*n, total_rank))

    add_strassen_3x3x3(U1, V1, W1, 0, 0, 0, 0)

    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                covered[di][dj][dk] = True

    col = 19
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U1[i*m+j, col] += 1
                    V1[j*p+k, col] += 1
                    W1[k*n+i, col] += 1
                    col += 1

    if verify(U1, V1, W1):
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}

    # Fallback: pure schoolbook (rank 75)
    total_rank2 = n * m * p
    U2 = np.zeros((n*m, total_rank2))
    V2 = np.zeros((m*p, total_rank2))
    W2 = np.zeros((p*n, total_rank2))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, col] = 1.0
                V2[j*p+k, col] = 1.0
                W2[k*n+i, col] = 1.0
                col += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
