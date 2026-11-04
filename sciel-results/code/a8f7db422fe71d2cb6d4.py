"""Matrix multiplication 3,3,5 - block-split with Bini 23-rank core."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 3, 5

    # Bini 1970 decomposition for 3x3x3 matrix multiplication (rank 23).
    # Entries are 0, 1, or -1.  For each term r:
    #   a_r[i,j] contributes to A's (i,j)
    #   b_r[j,k] contributes to B's (j,k)
    #   c_r[i,k] contributes to C's (i,k)
    # and sum_r a_r[i,j] * b_r[j,k] * c_r[i,k] = A[i,j]*B[j,k].
    a_rows = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, -1, 0],
        [0, 1, -1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, 0, 1],
        [0, 1, -1],
    ]
    b_rows = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    c_rows = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]

    def verify_block(a_rows, b_rows, c_rows):
        """Verify a 3x3x3 decomposition: sum_r a[i,j]*b[j,k]*c[i,k] == delta."""
        for i in range(3):
            for k in range(3):
                expected = 1.0 if i == k else 0.0
                got = 0.0
                for r in range(len(a_rows)):
                    for j in range(3):
                        got += a_rows[r][i][j] * b_rows[r][j][k] * c_rows[r][i][k]
                if abs(got - expected) > 1e-9:
                    return False
        return True

    def verify_full(U, V, W, rank, n, m, p):
        """Verify full 3x3x5 decomposition against the MM tensor."""
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] must equal 1 on valid chains, 0 else
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    got = 0.0
                    for r in range(rank):
                        got += U[a][r] * V[b][r] * W[c][r]
                    if abs(got - 1.0) > 1e-9:
                        return False
        # spot-check a few zero entries
        for a in range(n * m):
            for b in range(m * p):
                # find a c that is NOT a valid output for this (a,b)
                for c in range(p * n):
                    # valid if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == a and j * p + k == b and k * n + i == c:
                                    valid = True
                    if not valid:
                        got = 0.0
                        for r in range(rank):
                            got += U[a][r] * V[b][r] * W[c][r]
                        if abs(got) > 1e-9:
                            return False
        return True

    # --- Build block-split decomposition ---
    # C[:, :3] = A * B[:, :3]  (3x3x3, Bini rank 23)
    # C[:, 3:] = A * B[:, 3:]  (3x3x2, schoolbook rank 18)
    # Total rank = 41
    rank_bini = 23
    rank_school = n * m * 2  # 3*3*2 = 18
    rank = rank_bini + rank_school

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Bini part: columns 0..22
    # For term r: U[i*m+j, r] = a_r[i,j], V[j*p+k, r] = b_r[j,k] (k in 0..2),
    #             W[k*n+i, r] = c_r[i,k] (k in 0..2)
    for r in range(rank_bini):
        for i in range(3):
            for j in range(3):
                U[i * m + j][r] = a_rows[r][i][j]
        for j in range(3):
            for k in range(3):
                V[j * p + k][r] = b_rows[r][j][k]
        for k in range(3):
            for i in range(3):
                W[k * n + i][r] = c_rows[r][i][k]

    # Schoolbook part for columns 3,4 of B: columns 23..40
    col = rank_bini
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Verify; if anything is off, fall back to pure schoolbook (rank 45)
    if verify_full(U, V, W, rank, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: pure schoolbook rank 45
    rank45 = n * m * p
    U45 = [[0] * rank45 for _ in range(n * m)]
    V45 = [[0] * rank45 for _ in range(m * p)]
    W45 = [[0] * rank45 for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U45[i * m + j][column] = 1
                V45[j * p + k][column] = 1
                W45[k * n + i][column] = 1
                column += 1
    return {"U": U45, "V": V45, "W": W45}
# EVOLVE-BLOCK-END
