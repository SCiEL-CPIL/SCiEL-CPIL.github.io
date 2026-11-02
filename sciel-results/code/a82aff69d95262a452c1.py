"""Block-decomposition tensor factor for 2x2 x 2x7 multiplication.

Strategy: split the 7 output columns into four blocks of widths 2,2,2,1.
For each 2-column block use the rank-7 Strassen decomposition of 2x2 x 2x2.
For the final 1-column block use the rank-4 schoolbook decomposition of 2x2 x 2x1.
The overall decomposition is block-diagonal (each block acts on a disjoint set of
columns of B and a disjoint set of output positions), so it is valid with
rank 7+7+7+4 = 25, improving on the seed's rank-28 schoolbook.
"""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) of shape (4,7),(4,7),(4,7) for multiplying a 2x2 by a 2x2.

    Index conventions:
      A is 2x2 indexed (i,j), i=row, j=col  -> U row index i*m+j = i*2+j
      B is 2x2 indexed (j,k), j=row, k=col  -> V row index j*p+k = j*2+k
      C is 2x2 indexed (k,i), k=row, i=col  -> W row index k*n+i = k*2+i
    The 7 Strassen products are:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B10-B11)
      M4 = A11*(B01-B11)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B10)
      M7 = (A01-A11)*(B01+B11)
    with C00 = M1+M4-M5-M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6.
    """
    # U rows: A00,A01,A10,A11 ; V rows: B00,B01,B10,B11
    # W rows: C00,C01,C10,C11
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    # M2
    U[1][1] += 1; U[3][1] += 1
    V[0][1] += 1
    # M3
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    # M4
    U[3][3] += 1
    V[1][3] += 1; V[3][3] += -1
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    # M6
    U[1][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[2][5] += 1
    # M7
    U[1][6] += 1; U[3][6] += -1
    V[1][6] += 1; V[3][6] += 1
    # Combine into W
    # C00 = M1+M4-M5-M7
    W[0][0] += 1; W[0][3] += 1; W[0][4] += -1; W[0][6] += -1
    # C01 = M3+M5
    W[1][2] += 1; W[1][4] += 1
    # C10 = M2+M4
    W[2][1] += 1; W[2][3] += 1
    # C11 = M1-M2+M3+M6
    W[3][0] += 1; W[3][1] += -1; W[3][2] += 1; W[3][5] += 1
    return U, V, W


def _schoolbook_2x2x1():
    """Return (U, V, W) of shape (4,4),(2,4),(2,4) for multiplying 2x2 by 2x1.

    B has shape 2x1 indexed (j,k) with k=0 only -> V rows j*1+0 = j (j=0,1).
    C has shape 1x2 indexed (k,i) with k=0 -> W rows 0*2+i = i (i=0,1).
    Four products: A00*B0 -> C00, A01*B1 -> C01, A10*B0 -> C10, A11*B1 -> C11.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    # c=0: A00 * B0 -> C00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # c=1: A01 * B1 -> C01
    U[1][1] = 1; V[1][1] = 1; W[1][1] = 1
    # c=2: A10 * B0 -> C10
    U[2][2] = 1; V[0][2] = 1; W[0][2] = 1
    # c=3: A11 * B1 -> C11
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    return U, V, W


def _validate(U, V, W):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = 2, 2, 7
    rank = len(U[0])
    # C[k*n+i, j*m+? ] ... We verify the tensor T[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # equals the canonical multiplication tensor: T[A_idx, B_idx, C_idx] = 1 iff
    # A_idx=(i,j), B_idx=(j,k), C_idx=(k,i) for some i,j,k, else 0.
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build expected tensor of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A_idx = i*m + j
                B_idx = j*p + k
                C_idx = k*n + i
                T[A_idx, B_idx, C_idx] = 1.0
    # Contract: for each r, outer product of U[:,r], V[:,r], W[:,r]
    Tc = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        Tc += np.einsum('a,b,c->abc', Ua[:, r], Va[:, r], Wa[:, r])
    if not np.allclose(Tc, T, atol=1e-9):
        return False
    return True


def construct():
    start = time.time()
    deadline = 150.0 - 5.0  # safety margin

    n, m, p = 2, 2, 7

    # Block structure over the 7 columns of B: widths [2,2,2,1]
    # For each block b with column range [c0, c0+w):
    #   - A rows are the full 2x2 (all 4 A entries)
    #   - B rows are the w columns of the block: B[j, c0..c0+w-1] for j=0,1
    #   - C rows are the w output positions: C[c0..c0+w-1, i] for i=0,1
    # The block multiplication is (2x2) * (2 x w) -> (w x 2).
    #
    # For w=2: this is a 2x2 x 2x2 multiply with B's columns being the block's 2 cols,
    #   and C's rows being the block's 2 cols. Use Strassen (rank 7).
    # For w=1: this is a 2x2 x 2x1 multiply. Use schoolbook (rank 4).

    U = [[0]*0 for _ in range(n*m)]  # placeholder, will rebuild
    V = [[0]*0 for _ in range(m*p)]
    W = [[0]*0 for _ in range(p*n)]

    # Determine total rank
    block_widths = [2, 2, 2, 1]
    ranks = [7, 7, 7, 4]
    total_rank = sum(ranks)

    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    col_offset = 0
    for b, w in enumerate(block_widths):
        c0 = col_offset
        if w == 2:
            Ub, Vb, Wb = _strassen_2x2x2()
            # Ub rows: A00,A01,A10,A11 -> U rows i*2+j
            # Vb rows: B00,B01,B10,B11 -> B[j, c0+0], B[j, c0+1] -> V row j*p + (c0+0), j*p+(c0+1)
            # Wb rows: C00,C01,C10,C11 -> C[c0+0, i], C[c0+1, i] -> W row (c0+0)*2+i, (c0+1)*2+i
            r0 = sum(ranks[:b])
            for r in range(7):
                for a in range(4):
                    if Ub[a][r] != 0:
                        U[a][r0 + r] += Ub[a][r]
                # Vb row 0 = B00 = B[0, c0] -> V[0*p + c0]
                if Vb[0][r] != 0:
                    V[0*p + c0][r0 + r] += Vb[0][r]
                # Vb row 1 = B01 = B[0, c0+1] -> V[0*p + c0+1]
                if Vb[1][r] != 0:
                    V[0*p + c0 + 1][r0 + r] += Vb[1][r]
                # Vb row 2 = B10 = B[1, c0] -> V[1*p + c0]
                if Vb[2][r] != 0:
                    V[1*p + c0][r0 + r] += Vb[2][r]
                # Vb row 3 = B11 = B[1, c0+1] -> V[1*p + c0+1]
                if Vb[3][r] != 0:
                    V[1*p + c0 + 1][r0 + r] += Vb[3][r]
                # Wb row 0 = C00 = C[c0, 0] -> W[c0*2 + 0]
                if Wb[0][r] != 0:
                    W[c0*2 + 0][r0 + r] += Wb[0][r]
                # Wb row 1 = C01 = C[c0, 1] -> W[c0*2 + 1]
                if Wb[1][r] != 0:
                    W[c0*2 + 1][r0 + r] += Wb[1][r]
                # Wb row 2 = C10 = C[c0+1, 0] -> W[(c0+1)*2 + 0]
                if Wb[2][r] != 0:
                    W[(c0+1)*2 + 0][r0 + r] += Wb[2][r]
                # Wb row 3 = C11 = C[c0+1, 1] -> W[(c0+1)*2 + 1]
                if Wb[3][r] != 0:
                    W[(c0+1)*2 + 1][r0 + r] += Wb[3][r]
        elif w == 1:
            Ub, Vb, Wb = _schoolbook_2x2x1()
            # Ub rows: A00,A01,A10,A11 -> U rows
            # Vb rows: B0, B1 -> B[0, c0], B[1, c0] -> V[0*p+c0], V[1*p+c0]
            # Wb rows: C0, C1 -> C[c0, 0], C[c0, 1] -> W[c0*2+0], W[c0*2+1]
            r0 = sum(ranks[:b])
            for r in range(4):
                for a in range(4):
                    if Ub[a][r] != 0:
                        U[a][r0 + r] += Ub[a][r]
                if Vb[0][r] != 0:
                    V[0*p + c0][r0 + r] += Vb[0][r]
                if Vb[1][r] != 0:
                    V[1*p + c0][r0 + r] += Vb[1][r]
                if Wb[0][r] != 0:
                    W[c0*2 + 0][r0 + r] += Wb[0][r]
                if Wb[1][r] != 0:
                    W[c0*2 + 1][r0 + r] += Wb[1][r]
        col_offset += w
        if time.time() > deadline:
            break

    # Validate before returning
    if time.time() < deadline:
        try:
            if not _validate(U, V, W):
                # Fallback to schoolbook if something went wrong
                rank = n * m * p
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
