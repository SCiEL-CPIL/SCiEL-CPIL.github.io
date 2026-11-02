"""Matrix multiplication 2,2,13 — rank-minimal exact tensor decomposition.

We treat the 2x13 matrix B as a 2x2 matrix over the ring Q[x]/(x^13).
Strassen's algorithm for 2x2 matrix multiplication gives 7 multiplications
in the ring. Each ring multiplication (polynomial mult mod x^13) has tensor
rank 13 (it is the standard M_{1,1,13} convolution tensor, which for
truncated degree-12 polynomial multiplication has rank 13 over Q).

Total rank = 7 * 13 = 91.  But wait — schoolbook is 2*2*13 = 52, which is
better.  So Strassen over the ring is NOT an improvement here.

However, there is a better structural approach.  The key observation is that
the 2x2 * 2x13 multiplication can be seen as: for each of the 13 output
columns, we do a 2x2 * 2x1 multiply.  The 2x2 * 2x1 tensor has rank 4
(same as schoolbook for that sub-problem, since one dimension is 1).

But we can share work across columns.  Specifically, consider grouping
columns in pairs.  For each pair of consecutive columns (2k, 2k+1), we
have a 2x2 * 2x2 sub-problem, which has rank 7 (Strassen).  With 6 full
pairs (12 columns) and 1 remaining column, the total rank is 6*7 + 4 = 46.

This gives rank 46, which beats the schoolbook rank of 52.

Let me verify the Strassen decomposition for 2x2 * 2x2:
  A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]

  M1 = (a+d)(e+h)
  M2 = (c+d)e
  M3 = a(f-h)
  M4 = d(g-e)
  M5 = (a+b)h
  M6 = (c-a)(e+f)
  M7 = (b-d)(g+h)

  C00 = M1 + M4 - M5 + M7
  C10 = M2 + M4
  C01 = M3 + M5
  C11 = M1 - M2 + M3 + M6

This is the standard Strassen formula.  Each Mj is a product of a linear
combination of A-entries with a linear combination of B-entries.  In the
tensor decomposition, each Mj contributes one rank-1 term where:
  U[:,j] = the coefficients for the A linear combination
  V[:,j] = the coefficients for the B linear combination
  W[:,j] = the coefficients for how Mj contributes to C entries

For the 2x2 * 2x13 case, we group columns into pairs:
  Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)  — 6 pairs
  Remaining: column 12 — handled by schoolbook (rank 4)

Total rank = 6*7 + 4 = 46.

Let me build this decomposition.

For each pair of columns (2k, 2k+1), k=0..5:
  The B entries for this pair are:
    B[0, 2k], B[0, 2k+1], B[1, 2k], B[1, 2k+1]
  These are e, f, g, h in the Strassen notation.

  The 7 rank-1 terms for this pair:
  M1 = (a+d) * (e+h)
    U: a->1, d->1 (indices 0,1,2,3 for A entries a,b,c,d)
    V: e->1, h->1 (indices for B: 0*13+2k, 1*13+2k+1)
    W: C[0,2k] += 1, C[1,2k+1] += 1  (since C00=M1+..., C11=M1-...)
      W index: k*n+i where output C is p*n = 13*2 = 26
      C[0,2k] -> index 2k*2+0 = 4k
      C[1,2k+1] -> index (2k+1)*2+1 = 4k+3

  M2 = (c+d) * e
    U: c->1, d->1 (indices 2,3)
    V: e->1 (index 0*13+2k = 2k)
    W: C[1,2k] += 1, C[1,2k+1] += -1
      C[1,2k] -> index 2k*2+1 = 4k+1
      C[1,2k+1] -> index 4k+3

  M3 = a * (f-h)
    U: a->1 (index 0)
    V: f->1, h->-1 (indices 0*13+2k+1=2k+1, 1*13+2k+1=13+2k+1=14+2k)
    W: C[0,2k+1] += 1, C[1,2k+1] += 1
      C[0,2k+1] -> index (2k+1)*2+0 = 4k+2
      C[1,2k+1] -> index 4k+3

  M4 = d * (g-e)
    U: d->1 (index 3)
    V: g->1, e->-1 (indices 1*13+2k=13+2k, 0*13+2k=2k)
    W: C[0,2k] += 1, C[1,2k] += 1
      C[0,2k] -> index 4k
      C[1,2k] -> index 4k+1

  M5 = (a+b) * h
    U: a->1, b->1 (indices 0,1)
    V: h->1 (index 1*13+2k+1=14+2k)
    W: C[0,2k] += -1, C[0,2k+1] += 1
      C[0,2k] -> index 4k
      C[0,2k+1] -> index 4k+2

  M6 = (c-a) * (e+f)
    U: c->1, a->-1 (indices 2, 0)
    V: e->1, f->1 (indices 2k, 2k+1)
    W: C[1,2k+1] += 1
      C[1,2k+1] -> index 4k+3

  M7 = (b-d) * (g+h)
    U: b->1, d->-1 (indices 1, 3)
    V: g->1, h->1 (indices 13+2k, 14+2k)
    W: C[0,2k] += 1
      C[0,2k] -> index 4k

For the remaining column 12 (single column, schoolbook rank 4):
  C[0,12] = a*B[0,12] + b*B[1,12]
  C[1,12] = c*B[0,12] + d*B[1,12]

  Term 1: a * B[0,12] -> U: a->1, V: B[0,12]->1, W: C[0,12]->1
  Term 2: b * B[1,12] -> U: b->1, V: B[1,12]->1, W: C[0,12]->1
  Term 3: c * B[0,12] -> U: c->1, V: B[0,12]->1, W: C[1,12]->1
  Term 4: d * B[1,12] -> U: d->1, V: B[1,12]->1, W: C[1,12]->1

Total rank = 6*7 + 4 = 46.

Now I need to be careful about the index conventions.

U shape: (n*m, rank) = (4, 46)
  A entries: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
  Index in U: i*m+j, so a->0, b->1, c->2, d->3

V shape: (m*p, rank) = (26, 46)
  B entries: B[i,j] for i in 0..1, j in 0..12
  Index in V: i*p+j, so B[0,j]->j, B[1,j]->13+j

W shape: (p*n, rank) = (26, 46)
  C entries: C[i,j] for i in 0..1, j in 0..12
  Convention: W[k*n+i, r] where k is the column index, i is the row index
  So C[i,k] -> index k*n+i = 2k+i
  C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3, ..., C[0,12]->24, C[1,12]->25

Let me re-verify the W indices:
  C[0,2k] -> 2k*2+0 = 4k
  C[1,2k] -> 2k*2+1 = 4k+1
  C[0,2k+1] -> (2k+1)*2+0 = 4k+2
  C[1,2k+1] -> (2k+1)*2+1 = 4k+3
  C[0,12] -> 12*2+0 = 24
  C[1,12] -> 12*2+1 = 25

This all checks out.

Let me now implement this and verify with a test.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 13]


# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110
    n, m, p = 2, 2, 13
    rank = 6 * 7 + 4  # 46

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # 6 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for k in range(6):
        # Column indices for this pair
        c0 = 2 * k
        c1 = 2 * k + 1

        # B indices
        b_e = 0 * p + c0      # B[0, c0]
        b_f = 0 * p + c1      # B[0, c1]
        b_g = 1 * p + c0      # B[1, c0]
        b_h = 1 * p + c1      # B[1, c1]

        # C indices
        w_c00 = c0 * n + 0    # C[0, c0]
        w_c10 = c0 * n + 1    # C[1, c0]
        w_c01 = c1 * n + 0    # C[0, c1]
        w_c11 = c1 * n + 1    # C[1, c1]

        # M1 = (a+d)(e+h)
        U[0, col] = 1.0  # a
        U[3, col] = 1.0  # d
        V[b_e, col] = 1.0
        V[b_h, col] = 1.0
        W[w_c00, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        # M2 = (c+d)e
        U[2, col] = 1.0  # c
        U[3, col] = 1.0  # d
        V[b_e, col] = 1.0
        W[w_c10, col] = 1.0
        W[w_c11, col] = -1.0
        col += 1

        # M3 = a(f-h)
        U[0, col] = 1.0  # a
        V[b_f, col] = 1.0
        V[b_h, col] = -1.0
        W[w_c01, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        # M4 = d(g-e)
        U[3, col] = 1.0  # d
        V[b_g, col] = 1.0
        V[b_e, col] = -1.0
        W[w_c00, col] = 1.0
        W[w_c10, col] = 1.0
        col += 1

        # M5 = (a+b)h
        U[0, col] = 1.0  # a
        U[1, col] = 1.0  # b
        V[b_h, col] = 1.0
        W[w_c00, col] = -1.0
        W[w_c01, col] = 1.0
        col += 1

        # M6 = (c-a)(e+f)
        U[2, col] = 1.0  # c
        U[0, col] = -1.0  # -a
        V[b_e, col] = 1.0
        V[b_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        # M7 = (b-d)(g+h)
        U[1, col] = 1.0  # b
        U[3, col] = -1.0  # -d
        V[b_g, col] = 1.0
        V[b_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1

    # Remaining column 12: schoolbook (rank 4)
    c = 12
    b_0c = 0 * p + c  # B[0, 12]
    b_1c = 1 * p + c  # B[1, 12]
    w_c0c = c * n + 0  # C[0, 12]
    w_c1c = c * n + 1  # C[1, 12]

    # a * B[0,12]
    U[0, col] = 1.0
    V[b_0c, col] = 1.0
    W[w_c0c, col] = 1.0
    col += 1

    # b * B[1,12]
    U[1, col] = 1.0
    V[b_1c, col] = 1.0
    W[w_c0c, col] = 1.0
    col += 1

    # c * B[0,12]
    U[2, col] = 1.0
    V[b_0c, col] = 1.0
    W[w_c1c, col] = 1.0
    col += 1

    # d * B[1,12]
    U[3, col] = 1.0
    V[b_1c, col] = 1.0
    W[w_c1c, col] = 1.0
    col += 1

    assert col == rank, f"Column count {col} != rank {rank}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
