"""Matrix multiplication 2x3x10 tensor decomposition using block-Kronecker structure."""
import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Total rank: 11 (first 2 cols via Strassen+schoolbook) + 5*10 (remaining 8 cols) = 61
    rank = 61
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    # ============================================================
    # Part 1: First 2 columns of B (k=0,1)
    # A is 2x3, B1 is 3x2
    # Split A into A1 (2x2) and A2 (2x1), B1 into B11 (2x2) and B12 (1x2)
    # Strassen for A1*B11: 7 terms
    # Schoolbook for A2*B12: 4 terms
    # Total: 11 terms
    # ============================================================

    # Index conventions:
    # U[i*m+j] where i in [0,n), j in [0,m)  -> U index = i*m + j
    # V[j*p+k] where j in [0,m), k in [0,p)  -> V index = j*p + k
    # W[k*n+i] where k in [0,p), i in [0,n)  -> W index = k*n + i

    # Strassen for A1 (2x2) * B11 (2x2):
    # A1 = [[a00, a01], [a10, a11]]  (rows i=0,1; cols j=0,1)
    # B11 = [[b00, b01], [b10, b11]] (rows j=0,1; cols k=0,1)
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C11 (i=0,k=0) = M1 - M5 + M7
    # C12 (i=0,k=1) = M3 + M5
    # C21 (i=1,k=0) = M2 + M4
    # C22 (i=1,k=1) = M1 - M2 + M6

    # M1 = (a00+a11)*(b00+b11)
    U[0*m+0, col] = 1  # a00
    U[1*m+1, col] = 1  # a11
    V[0*p+0, col] = 1  # b00
    V[1*p+1, col] = 1  # b11
    W[0*n+0, col] = 1  # C11
    W[1*n+1, col] = 1  # C22
    col += 1

    # M2 = (a10+a11)*b00
    U[1*m+0, col] = 1  # a10
    U[1*m+1, col] = 1  # a11
    V[0*p+0, col] = 1  # b00
    W[1*n+0, col] = 1  # C21
    col += 1

    # M3 = a00*(b01-b11)
    U[0*m+0, col] = 1  # a00
    V[0*p+1, col] = 1  # b01
    V[1*p+1, col] = -1  # -b11
    W[0*n+1, col] = 1  # C12
    col += 1

    # M4 = a11*(b10-b00)
    U[1*m+1, col] = 1  # a11
    V[1*p+0, col] = 1  # b10
    V[0*p+0, col] = -1  # -b00
    W[1*n+0, col] = 1  # C21
    col += 1

    # M5 = (a00+a01)*b11
    U[0*m+0, col] = 1  # a00
    U[0*m+1, col] = 1  # a01
    V[1*p+1, col] = 1  # b11
    W[0*n+0, col] = -1  # -C11
    W[0*n+1, col] = 1  # C12
    col += 1

    # M6 = (a10-a00)*(b00+b01)
    U[1*m+0, col] = 1  # a10
    U[0*m+0, col] = -1  # -a00
    V[0*p+0, col] = 1  # b00
    V[0*p+1, col] = 1  # b01
    W[1*n+1, col] = 1  # C22
    col += 1

    # M7 = (a01-a11)*(b10+b11)
    U[0*m+1, col] = 1  # a01
    U[1*m+1, col] = -1  # -a11
    V[1*p+0, col] = 1  # b10
    V[1*p+1, col] = 1  # b11
    W[0*n+0, col] = 1  # C11
    col += 1

    # Schoolbook for A2 (2x1) * B12 (1x2):
    # A2 = [[a02], [a12]]  (rows i=0,1; col j=2)
    # B12 = [[b20, b21]]   (row j=2; cols k=0,1)
    # C11 += a02*b20
    # C12 += a02*b21
    # C21 += a12*b20
    # C22 += a12*b21

    # a02*b20
    U[0*m+2, col] = 1  # a02
    V[2*p+0, col] = 1  # b20
    W[0*n+0, col] = 1  # C11
    col += 1

    # a02*b21
    U[0*m+2, col] = 1  # a02
    V[2*p+1, col] = 1  # b21
    W[0*n+1, col] = 1  # C12
    col += 1

    # a12*b20
    U[1*m+2, col] = 1  # a12
    V[2*p+0, col] = 1  # b20
    W[1*n+0, col] = 1  # C21
    col += 1

    # a12*b21
    U[1*m+2, col] = 1  # a12
    V[2*p+1, col] = 1  # b21
    W[1*n+1, col] = 1  # C22
    col += 1

    # ============================================================
    # Part 2: Remaining 8 columns of B (k=2..9)
    # Use 5 rank-10 schoolbook tensors
    # Each tensor handles 2 output columns
    # For each pair of k values (k1, k2), we need terms for all i,j
    # That's n*m*2 = 12 terms per pair... but the directive says rank-10.
    # Let me re-read: "five rank-10 (1x2x10 schoolbook) tensors"
    # 
    # Actually, let me think about this differently.
    # We have 8 columns remaining. 5 tensors of rank 10 = 50 terms.
    # 8 columns * n*m = 8*6 = 48 schoolbook terms minimum.
    # So 50 > 48, there's some redundancy or different grouping.
    #
    # Let me just do schoolbook for the remaining 8 columns: 8*6 = 48 terms.
    # Total = 11 + 48 = 59.
    #
    # But the directive says rank 61 with five rank-10 tensors.
    # Let me follow the directive: 5 groups, each with 10 terms.
    # 8 columns / 5 groups... doesn't divide evenly.
    #
    # Let me re-interpret: perhaps the "1x2x10" means we group by rows of A?
    # A has 2 rows. For each row i, we have 1*3*10 = 30 terms schoolbook.
    # With 2 rows that's 60 total schoolbook.
    #
    # I'll just do schoolbook for the remaining 8 columns to get rank 59,
    # which is better than 61.
    # ============================================================

    for k in range(2, p):  # k = 2,3,...,9
        for i in range(n):
            for j in range(m):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    # Adjust rank to actual number of terms used
    rank = col
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]

    return {"U": U, "V": V, "W": W}
