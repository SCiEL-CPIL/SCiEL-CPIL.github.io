"""
Rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.

Strategy:
  1. Start from the schoolbook rank-63 decomposition.
  2. Search for small-rank corrections (rank 1, 2, 3, ...) that can be added
     to the schoolbook factors to produce a valid decomposition of lower rank,
     OR find an alternative decomposition of rank < 63 directly.
  3. In practice, for 3x3*3x7 the known lower bound is 21 and the schoolbook
     is 63. We attempt to find a rank below 63 via structured search.

  Given the difficulty of finding a rank < 63 decomposition in the time budget,
  we use a hybrid approach:
    - First, try to find a rank-61 decomposition via block partitioning
      (Strassen for 2x2x2 blocks where applicable).
    - If that fails or times out, fall back to schoolbook rank 63.

  The key insight for rank reduction: partition the 3x3 matrix A into blocks
  and use Strassen-like tricks on 2x2 sub-blocks.

  Partition:
    A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B (3x7) = [B1; B2; B3] where B1 is 2x7, B2 is 1x7, B3 is 1x7
    C (3x7) = [C1; C2; C3] where C1 is 2x7, C2 is 1x7, C3 is 1x7

  C1 = A11*B1 + A12*B2
  C2 = A21*B1 + A22*B2
  C3 = A21*B3 + A22*B3  -- wait, let me re-derive.

  Actually, let's index properly:
  A rows 0,1,2; cols 0,1,2
  B rows 0,1,2; cols 0..6
  C rows 0,1,2; cols 0..6

  C[i][k] = sum_j A[i][j] * B[j][k]

  Partition A:
    Block AA = A[0:2, 0:2]  (2x2)
    Block AB = A[0:2, 2]    (2x1)
    Block BA = A[2, 0:2]    (1x2)
    Block BB = A[2, 2]      (1x1)

  Partition B:
    Block B1 = B[0:2, :]    (2x7)
    Block B2 = B[2, :]      (1x7)

  Then:
    C[0:2, :] = AA * B1 + AB * B2
    C[2, :]   = BA * B1 + BB * B2

  For C[0:2, :]:
    AA * B1 is 2x2 * 2x7 -> can use Strassen on 2x2*2x2 blocks for each pair of columns.
    But B1 is 2x7, so we split into 3 blocks of 2 columns + 1 block of 1 column.
    Strassen for 2x2*2x2 has rank 7.
    Schoolbook for 2x2*2x2 has rank 4.
    So Strassen saves 3 per 2-column block.

    For 7 columns: 3 full 2-col blocks (6 cols) + 1 single col.
    3 * Strassen(7) + schoolbook(2x2*2x1 = 4) = 21 + 4 = 25 for AA*B1.
    AB * B2: 2x1 * 1x7 = 14 (schoolbook, no reduction possible)
    Total for C[0:2,:]: 25 + 14 = 39

  For C[2,:]:
    BA * B1: 1x2 * 2x7 = 14 (schoolbook)
    BB * B2: 1x1 * 1x7 = 7 (schoolbook)
    Total: 21

  Grand total: 39 + 21 = 60.

  Wait, that's rank 60! Let me verify the Strassen decomposition for 2x2*2x2.

  Strassen 2x2*2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)

    C00 = M1 - M4 + M5 + M6
    C01 = M3 + M4
    C10 = M2 + M5 - M1 - M7
    C11 = M1 - M2 + M3 + M7

  This gives a rank-7 decomposition for 2x2*2x2 matrix multiplication.

  For 2x2 * 2x1 (single column), schoolbook rank is 4.

  So for AA * B1 where B1 is 2x7:
    Split B1 into B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)
    Strassen for B1a: rank 7
    Strassen for B1b: rank 7
    Strassen for B1c: rank 7
    Schoolbook for B1d: rank 4
    Total: 25

  This should work. Let me implement it.

  Actually, I need to be more careful. The Strassen decomposition operates on the
  tensor for 2x2*2x2. When I apply it to a 2x2*2x2 sub-problem, I get 7 terms.
  Each term corresponds to a product of a linear combination of A entries and a
  linear combination of B entries, producing a linear combination of C entries.

  Let me build the full decomposition explicitly.
"""

import time
from fractions import Fraction

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns list of (U_vec, V_vec, W_vec) where each vec is a 4-element list
    representing coefficients for the 4 entries of A, B, C respectively.
    A indexing: [A00, A01, A10, A11]
    B indexing: [B00, B01, B10, B11]
    C indexing: [C00, C01, C10, C11]
    """
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    # M3 = A00(B10-B11) -> C01 += 1, C10 += 1
    # M4 = A11(B01-B00) -> C00 -= 1, C01 += 1
    # M5 = (A00+A01)B11 -> C00 += 1, C10 -= 1
    # M6 = (A10-A00)(B00+B10) -> C00 += 1, C10 -= 1
    # M7 = (A01-A11)(B01+B11) -> C10 -= 1, C11 += 1

    terms = []
    # M1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3
    terms.append(([1,0,0,0], [0,0,1,-1], [0,1,1,0]))
    # M4
    terms.append(([0,0,0,1], [0,1,-1,0], [-1,1,0,0]))
    # M5
    terms.append(([1,1,0,0], [0,0,0,1], [1,0,-1,0]))
    # M6
    terms.append(([1,0,-1,0], [1,0,1,0], [1,0,-1,0]))
    # M7
    terms.append(([0,1,0,-1], [0,1,0,1], [0,0,-1,1]))

    return terms

def _schoolbook_2x2x2x1():
    """
    Schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    A: [A00, A01, A10, A11] (2x2)
    B: [B0, B1] (2x1)
    C: [C0, C1] (2x1)
    C0 = A00*B0 + A01*B1
    C1 = A10*B0 + A11*B1
    """
    terms = []
    terms.append(([1,0,0,0], [1,0], [1,0]))
    terms.append(([0,1,0,0], [0,1], [1,0]))
    terms.append(([0,0,1,0], [1,0], [0,1]))
    terms.append(([0,0,0,1], [0,1], [0,1]))
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = 3, 3, 7

    # Build the rank-60 decomposition using block partitioning
    # A (3x3) indexed as [A00, A01, A02, A10, A11, A12, A20, A21, A22]
    # B (3x7) indexed as [B00..B06, B10..B16, B20..B26]
    # C (3x7) indexed as [C00..C06, C10..C16, C20..C26]
    # W index: k*n + i for C[k][i] -> C[i][k] in our notation
    # Wait, the problem says W[k*n+i, r] which is trace convention.
    # C[i][k] = sum_j A[i][j] B[j][k]
    # W index for C[i][k] is k*n + i = k*3 + i

    # A indices: i*m + j = i*3 + j for A[i][j]
    # B indices: j*p + k = j*7 + k for B[j][k]
    # W indices: k*n + i = k*3 + i for C[i][k]

    # Block partition:
    # AA = A[0:2, 0:2] -> A indices: 0,1,3,4
    # AB = A[0:2, 2]   -> A indices: 2,5
    # BA = A[2, 0:2]   -> A indices: 6,7
    # BB = A[2, 2]     -> A index: 8

    # B1 = B[0:2, :] -> B indices: 0..6, 7..13
    # B2 = B[2, :]   -> B indices: 14..20

    # C[0:2, :] = AA*B1 + AB*B2
    # C[2, :]   = BA*B1 + BB*B2

    # We'll build U, V, W as lists of lists (rank x n*m, rank x m*p, rank x p*n)
    # Actually the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)

    U = []  # will be list of rank vectors, each of length 9
    V = []  # will be list of rank vectors, each of length 21
    W = []  # will be list of rank vectors, each of length 21

    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length 9, v_vec: length 21, w_vec: length 21"""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    def zero_u():
        return [0]*9
    def zero_v():
        return [0]*21
    def zero_w():
        return [0]*21

    # Process C[0:2, :] = AA*B1 + AB*B2

    # AA*B1: AA is 2x2, B1 is 2x7
    # Split B1 into B1a(2x2), B1b(2x2), B1c(2x2), B1d(2x1)
    # B1a: B[0,0:2], B[1,0:2] -> B indices 0,1,7,8
    # B1b: B[0,2:4], B[1,2:4] -> B indices 2,3,9,10
    # B1c: B[0,4:6], B[1,4:6] -> B indices 4,5,11,12
    # B1d: B[0,6], B[1,6]     -> B indices 6,13

    # For each Strassen block, we need to map local A, B, C indices to global.
    # Local A for AA: [A00, A01, A10, A11] -> global A indices [0, 1, 3, 4]
    # Local B for B1a: [B00, B01, B10, B11] -> global B indices [0, 1, 7, 8]
    # Local C for C[0:2, 0:2]: [C00, C01, C10, C11] -> global C indices
    #   C[0,0] -> W index 0*3+0 = 0
    #   C[0,1] -> W index 1*3+0 = 3
    #   C[1,0] -> W index 0*3+1 = 1
    #   C[1,1] -> W index 1*3+1 = 4
    # So local C [C00, C01, C10, C11] maps to global W [0, 3, 1, 4]

    # For B1b: B indices [2, 3, 9, 10]
    # C[0:2, 2:4]: C[0,2]->W 2*3+0=6, C[0,3]->W 3*3+0=9, C[1,2]->W 2*3+1=7, C[1,3]->W 3*3+1=10
    # Local C maps to global W [6, 9, 7, 10]

    # For B1c: B indices [4, 5, 11, 12]
    # C[0:2, 4:6]: C[0,4]->W 4*3+0=12, C[0,5]->W 5*3+0=15, C[1,4]->W 4*3+1=13, C[1,5]->W 5*3+1=16
    # Local C maps to global W [12, 15, 13, 16]

    # For B1d (schoolbook 2x2*2x1):
    # B indices [6, 13]
    # C[0:2, 6]: C[0,6]->W 6*3+0=18, C[1,6]->W 6*3+1=19
    # Local C [C0, C1] maps to global W [18, 19]

    strassen_terms = _strassen_2x2x2()

    # AA*B1a: Strassen
    a_map = [0, 1, 3, 4]  # local A00,A01,A10,A11 -> global
    for b_start, c_w_map in [
        ([0, 1, 7, 8], [0, 3, 1, 4]),    # B1a
        ([2, 3, 9, 10], [6, 9, 7, 10]),  # B1b
        ([4, 5, 11, 12], [12, 15, 13, 16])  # B1c
    ]:
        for lu, lv, lw in strassen_terms:
            gu = zero_u()
            gv = zero_v()
            gw = zero_w()
            for li, gi in enumerate(a_map):
                gu[gi] = lu[li]
            for li, gi in enumerate(b_start):
                gv[gi] = lv[li]
            for li, gi in enumerate(c_w_map):
                gw[gi] = lw[li]
            add_term(gu, gv, gw)

    # AA*B1d: Schoolbook 2x2*2x1
    sb_terms = _schoolbook_2x2x2x1()
    a_map = [0, 1, 3, 4]
    b_map = [6, 13]
    c_map = [18, 19]
    for lu, lv, lw in sb_terms:
        gu = zero_u()
        gv = zero_v()
        gw = zero_w()
        for li, gi in enumerate(a_map):
            gu[gi] = lu[li]
        for li, gi in enumerate(b_map):
            gv[gi] = lv[li]
        for li, gi in enumerate(c_map):
            gw[gi] = lw[li]
        add_term(gu, gv, gw)

    # AB*B2: AB is 2x1, B2 is 1x7 -> 2x7 result
    # AB: A[0,2]=A2, A[1,2]=A5 -> A indices [2, 5]
    # B2: B[2, 0:7] -> B indices [14, 15, 16, 17, 18, 19, 20]
    # C[0:2, 0:7]: C[0,k]->W 3k+0, C[1,k]->W 3k+1
    # This is schoolbook: C[0,k] = A2*B[2,k], C[1,k] = A5*B[2,k]
    # Rank = 2 * 7 = 14
    for k in range(7):
        b_idx = 14 + k
        c0_w = k * 3 + 0
        c1_w = k * 3 + 1
        # Term 1: A2 * B[2,k] -> C[0,k]
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        gu[2] = 1
        gv[b_idx] = 1
        gw[c0_w] = 1
        add_term(gu, gv, gw)
        # Term 2: A5 * B[2,k] -> C[1,k]
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        gu[5] = 1
        gv[b_idx] = 1
        gw[c1_w] = 1
        add_term(gu, gv, gw)

    # Process C[2, :] = BA*B1 + BB*B2
    # BA: A[2,0]=A6, A[2,1]=A7 -> A indices [6, 7]
    # B1: B[0,0:7], B[1,0:7] -> B indices [0..6, 7..13]
    # C[2, k] -> W index k*3+2
    # BA*B1: C[2,k] = A6*B[0,k] + A7*B[1,k]
    # Rank = 2 * 7 = 14
    for k in range(7):
        b0_idx = k
        b1_idx = 7 + k
        c_w = k * 3 + 2
        # Term 1: A6 * B[0,k] -> C[2,k]
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        gu[6] = 1
        gv[b0_idx] = 1
        gw[c_w] = 1
        add_term(gu, gv, gw)
        # Term 2: A7 * B[1,k] -> C[2,k]
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        gu[7] = 1
        gv[b1_idx] = 1
        gw[c_w] = 1
        add_term(gu, gv, gw)

    # BB*B2: BB is 1x1 (A[2,2]=A8), B2 is 1x7
    # C[2,k] += A8 * B[2,k]
    # Rank = 7
    for k in range(7):
        b_idx = 14 + k
        c_w = k * 3 + 2
        gu = zero_u(); gv = zero_v(); gw = zero_w()
        gu[8] = 1
        gv[b_idx] = 1
        gw[c_w] = 1
        add_term(gu, gv, gw)

    rank = len(U)

    # Verify the decomposition
    if time.time() < deadline:
        try:
            import numpy as np
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            # Check: sum_r U[:,r] * V[:,r] * W[:,r] should give the multiplication tensor
            # Tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
            # Actually, the tensor for matrix multiplication is:
            # T[i, j, k] where the term A[i,j]*B[j,k] contributes to C[i,k]
            # In our indexing:
            # A index: i*3+j
            # B index: j*7+k
            # W index: k*3+i
            # The tensor has entries T[a_idx, b_idx, w_idx] = 1 if a_idx=i*3+j, b_idx=j*7+k, w_idx=k*3+i for some i,j,k
            # Check: sum_r U[a,r]*V[b,r]*W[w,r] should equal T[a,b,w]
            max_err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        a_idx = i * 3 + j
                        b_idx = j * 7 + k
                        w_idx = k * 3 + i
                        val = np.sum(U_np[a_idx, :] * V_np[b_idx, :] * W_np[w_idx, :])
                        if abs(val - 1.0) > max_err:
                            max_err = abs(val - 1.0)
            # Also check that non-matching entries are zero
            for r in range(rank):
                pass
            # Spot check a few non-matching entries
            for a_idx in range(9):
                for b_idx in range(21):
                    for w_idx in range(21):
                        i = a_idx // 3
                        j = a_idx % 3
                        j2 = b_idx // 7
                        k = b_idx % 7
                        i2 = w_idx % 3
                        k2 = w_idx // 3
                        if i == i2 and j == j2 and k == k2:
                            continue
                        val = np.sum(U_np[a_idx, :] * V_np[b_idx, :] * W_np[w_idx, :])
                        if abs(val) > max_err:
                            max_err = abs(val)
            if max_err > 1e-10:
                print(f"Verification failed with max error {max_err}")
                # Fall back to schoolbook
                U = []
                V = []
                W = []
                col = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(7):
                            gu = zero_u(); gv = zero_v(); gw = zero_w()
                            gu[i*3+j] = 1
                            gv[j*7+k] = 1
                            gw[k*3+i] = 1
                            U.append(gu)
                            V.append(gv)
                            W.append(gw)
                rank = 63
        except Exception as e:
            print(f"Verification error: {e}")
            # Fall back to schoolbook
            U = []
            V = []
            W = []
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        gu = zero_u(); gv = zero_v(); gw = zero_w()
                        gu[i*3+j] = 1
                        gv[j*7+k] = 1
                        gw[k*3+i] = 1
                        U.append(gu)
                        V.append(gv)
                        W.append(gw)
            rank = 63

    return {"U": U, "V": V, "W": W}
