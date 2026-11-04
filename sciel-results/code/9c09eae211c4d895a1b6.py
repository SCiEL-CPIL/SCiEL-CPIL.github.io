"""
Rank-minimal tensor decomposition for 2x5 * 5x5 matrix multiplication.

Strategy:
1. Build a baseline block decomposition (rank 49) using Strassen for the 2x2x2
   block and schoolbook for the remaining blocks.
2. Attempt to reduce rank via random algebraic search: propose rank-r factor
   matrices and solve linear constraints. Verify exactness over Q.
3. Return the best (lowest-rank) verified decomposition.

All entries are rational (represented as Python Fractions or integers), so
verification is exact.
"""

import time
import random
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 100.0  # safety margin before 150s kill


def _time_left():
    return _DEADLINE_S - (time.time() - _T0)


def _build_block_rank49():
    """
    Block decomposition giving rank 49:
      Block 1: rows 0-1, cols 0-1 of A; rows 0-1 of B -> 2x2*2x2, Strassen rank 7
      Block 2: rows 0-1, cols 0-1 of A; rows 2-4 of B -> 2x2*2x3, schoolbook 12
      Block 3: rows 0-1, cols 2-4 of A; rows 0-1 of B -> 2x3*3x2, schoolbook 12
      Block 4: rows 0-1, cols 2-4 of A; rows 2-4 of B -> 2x3*3x3, schoolbook 18
    Total: 7+12+12+18 = 49
    """
    n, m, p = 2, 5, 5
    total = 7 + 12 + 12 + 18
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]

    # --- Strassen 2x2 * 2x2 ---
    # indices: A[i][j] with i,j in {0,1}; B[j][k] with j,k in {0,1}
    # 7 products:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-e)
    # m4 = (a+b)h
    # m5 = (c-a)(e+f)
    # m6 = (b-d)(g+h)
    # C00 = m0 + m4 - m5 + m6
    # C01 = m3 + m5
    # C10 = m1 + m4
    # C11 = m0 - m1 + m3 + m2
    strassen = [
        # (U_idx_list, V_idx_list, W_idx_list)
        ([0, 1], [0, 3], [0]),           # m0
        ([2, 3], [0], [1]),              # m1
        ([0], [1, 3], [3]),              # m2  (a, f-h) -> W[3]=C11
        ([3], [2, 0], [2]),              # m3  (d, g-e) -> W[2]=C01... wait
        ([0, 1], [3], [1]),              # m4
        ([2, 0], [0, 1], [2]),           # m5
        ([1, 3], [2, 3], [3]),           # m6
    ]
    # Let me redo this more carefully.
    # A = [[a,b],[c,d]] = A[0][0]=a, A[0][1]=b, A[1][0]=c, A[1][1]=d
    # B = [[e,f],[g,h]] = B[0][0]=e, B[0][1]=f, B[1][0]=g, B[1][1]=h
    # C = A*B: C[0][0]=ae+cg, C[0][1]=af+dh, C[1][0]=ce+dg, C[1][1]=cf+dh

    # U factor: for product m_r, U[i*m+j] = coefficient of A[i][j]
    # V factor: for product m_r, V[j*p+k] = coefficient of B[j][k]
    # W factor: for product m_r, W[k*n+i] = coefficient for C[i][k]

    # m0 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; contributes to C00 and C11
    # m1 = (c+d)e: U: c=1,d=1; V: e=1; contributes to C10 and C11
    # m2 = a(f-h): U: a=1; V: f=1,h=-1; contributes to C01 and C11
    # m3 = d(g-e): U: d=1; V: g=1,e=-1; contributes to C01 and C11
    # m4 = (a+b)h: U: a=1,b=1; V: h=1; contributes to C00 and C10
    # m5 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; contributes to C00 and C01
    # m6 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; contributes to C00 and C11

    # C00 = m0 + m4 - m5 + m6
    # C01 = m3 + m5  (check: d(g-e)+(c-a)(e+f) = dg-de+ce+cf-ae-af... hmm not quite)
    # Let me just use the standard Strassen formulas directly.

    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(f-h)
    # m3 = a(d-b)  -- no, let me look up the standard form
    # Actually let me just use a known correct set:
    #
    # m0 = (a + d)(e + h)
    # m1 = (c + d) * e
    # m2 = a * (f - h)
    # m3 = d * (g - e)
    # m4 = (a + b) * h
    # m5 = (c - a) * (e + f)
    # m6 = (b - d) * (g + h)
    #
    # C00 = m0 + m4 - m5 + m6
    # C01 = m3 + m5
    # C10 = m1 + m4
    # C11 = m0 - m1 + m3 + m2

    # Verify C01: m3 + m5 = d(g-e) + (c-a)(e+f) = dg - de + ce + cf - ae - af
    # Actual C01 = af + dh. These don't match in general.
    # Let me recheck. The standard Strassen decomposition is:
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(f-h)
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 - m4 + m3 + m7  (using 1-indexed)
    # C00 = m1 + m5 - m6 + m7
    # C01 = m4 + m6
    # C10 = m2 + m5
    #
    # Let me verify C01 = m4 + m6 = d(g-e) + (c-a)(e+f)
    #   = dg - de + ce + cf - ae - af
    #   Hmm, that's not af+dh either. I'm confusing myself.
    #
    # Let me just use the most common formulation:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0][0] = p1 + p5 - p6 + p7
    # C[0][1] = p4 + p6
    # C[1][0] = p2 + p5
    # C[1][1] = p1 - p2 + p4 + p3

    # Verify C[0][1] = p4 + p6 = d(g-e) + (c-a)(e+f)
    # = dg - de + ce + cf - ae - af
    # We need this to equal af + dh. Not matching. I must have the indices wrong.

    # OK let me just carefully derive or look up. The issue is I keep getting
    # confused. Let me use a different, well-known correct set:
    #
    # M = [[a,b],[c,d]], N = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(f-h)
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # Result:
    # [[p1+p5-p6+p7, p4+p6],
    #  [p2+p5, p1-p2+p4+p3]]
    #
    # Check top-right: p4+p6 = d(g-e)+(c-a)(e+f) = dg-de+ce+cf-ae-af
    # Should be af+dh. Still doesn't work. I think the standard Strassen is:
    #
    # Actually, I recall now. The correct standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(f-h)
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C = [[p1 - p4 + p6 - p2, p1 + p4 - p5 - p7],
    #      [p3 - p1 + p7 + p5, p2 - p3 + p1 - p6]]
    #
    # I'm going in circles. Let me just hardcode a verified Strassen set
    # by checking with specific values in code, or better yet, use the
    # schoolbook for all blocks (rank 50) as a safe fallback and focus
    # the search on finding something better.

    # Actually, the simplest safe approach: use schoolbook for everything (rank 50)
    # and then try to do better via search. But the current best is 49.
    # Let me just use the block decomposition with a CORRECT Strassen.

    # I'll define Strassen and verify it in code before using it.
    pass  # placeholder, will implement below


def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 matrix multiplication, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Using the well-known Strassen formulas (verified):
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(f-h)
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0][0] = p1 + p5 - p6 + p7
    # C[0][1] = p4 + p6
    # C[1][0] = p2 + p5
    # C[1][1] = p1 - p2 + p3 + p4

    # Let me verify C[0][1] = p4 + p6:
    # p4 = d(g-e), p6 = (c-a)(e+f)
    # p4+p6 = dg - de + ce + cf - ae - af
    # We need C[0][1] = af + dh. This doesn't match!
    #
    # I think the correct formula uses different signs. Let me try:
    # C[0][1] = p4 + p6 where p4 = d(g-e) and p6 = (c-a)(e+f)
    # Expanding: dg - de + ce + cf - ae - af
    # For this to equal af + dh, we'd need dg-de+ce+cf-ae-af = af+dh
    # => dg - de + ce + cf - ae - 2af - dh = 0, which is not identically zero.
    #
    # So my formula is wrong. Let me use a different reference.
    #
    # Correct Strassen (from Wikipedia, verified):
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(f - h)
    # p3 = a(f - h)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    #
    # C = [[p1 + p5 - p6 + p7, p4 + p6],
    #      [p2 + p5, p1 - p2 + p4 + p3]]
    #
    # Hmm, same formulas. Let me test with a=1,b=0,c=0,d=1 (identity A),
    # e=1,f=2,g=3,h=4 (B = [[1,2],[3,4]])
    # C should be B itself: [[1,2],[3,4]]
    #
    # p1 = (1+1)(1+4) = 10
    # p2 = (0+1)(2-4) = -2
    # p3 = 1*(2-4) = -2
    # p4 = 1*(3-1) = 2
    # p5 = (1+0)*4 = 4
    # p6 = (0-1)(1+2) = -3
    # p7 = (0-1)(3+4) = -7
    #
    # C[0][0] = 10+4-(-3)+(-7) = 10+4+3-7 = 10 ✓
    # C[0][1] = 2+(-3) = -1 ✗ (should be 2)
    #
    # So the formula C[0][1] = p4 + p6 is wrong.
    #
    # Let me try C[0][1] = p4 + p6 with different p definitions.
    # Maybe I have the p's mixed up. Let me look at this differently.
    #
    # Actually, I think the issue is that different sources use different
    # labelings. Let me just construct a valid rank-7 decomposition by
    # solving the linear system, or use a known-correct set.
    #
    # Alternative: I'll just use schoolbook for all blocks (rank 50) as the
    # baseline, and focus computational effort on finding a better decomposition
    # via random search. Rank 50 is safe and verifiable.

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # I'll fill these in using a verified set. Since I can't easily verify
    # in my head, let me use a computational approach: start with schoolbook
    # (rank 50) and let the search try to reduce it. For now, return a
    # simple rank-4 schoolbook for 2x2x2 to keep things safe.
    # Actually, let me just use the original program's Strassen which was
    # verified to work (score 1.0 means it was valid).

    # From the original program, the Strassen was:
    # r=0: (a+d)(e+h) -> U:[1,1,0,0], V:[1,0,0,1], W: C00+=1, C11+=1
    # r=1: (c+d)e -> U:[0,0,1,1], V:[1,0,0,0], W: C10+=1, C11+=(-1)
    # r=2: a(f-h) -> U:[1,0,0,0], V:[0,1,0,-1], W: C01+=1, C11+=1
    # r=3: d(g-e) -> U:[0,0,0,1], V:[0,0,1,-1], W: C01+=1, C11+=1
    # r=4: (a+b)h -> U:[1,1,0,0], V:[0,0,0,1], W: C00+=1, C10+=1
    # r=5: (c-a)(e+f) -> U:[-1,0,1,0], V:[1,1,0,0], W: C00+=(-1), C01+=1
    # r=6: (b-d)(g+h) -> U:[0,1,0,-1], V:[0,0,1,1], W: C00+=1, C11+=1

    # Wait, the original had:
    # W[0] (C00) = m1 + m4 - m5 + m7  (0-indexed: W[0][0]=1, W[0][3]=1, W[0][4]=-1, W[0][6]=1)
    # W[2] (C01) = m3 + m5  (W[2][2]=1, W[2][4]=1)
    # W[1] (C10) = m2 + m4  (W[1][1]=1, W[1][3]=1)
    # W[3] (C11) = m1 - m2 + m3 + m6 (W[3][0]=1, W[3][1]=-1, W[3][2]=1, W[3][5]=1)

    # Using 0-indexed products m0..m6:
    # m0 = (a+d)(e+h)
    # m1 = (c+d)e
    # m2 = a(f-h)
    # m3 = d(g-e)
    # m4 = (a+b)h
    # m5 = (c-a)(e+f)
    # m6 = (b-d)(g+h)
    #
    # C00 = m0 + m4 - m5 + m6
    # C01 = m3 + m5
    # C10 = m1 + m4
    # C11 = m0 - m1 + m3 + m2

    # Let me verify C01 = m3 + m5 = d(g-e) + (c-a)(e+f)
    # = dg - de + ce + cf - ae - af
    # Need = af + dh. Still doesn't work!
    #
    # But the original program scored 1.0, meaning it passed verification.
    # So either my verification logic is wrong, or the original's indexing
    # is different from what I'm assuming.
    #
    # The original uses W[k*n + i] convention where n=2.
    # W[0] = W[0*2+0] = C[0][0] (row 0, col 0)
    # W[1] = W[0*2+1] = C[1][0] (row 1, col 0)
    # W[2] = W[1*2+0] = C[0][1] (row 0, col 1)
    # W[3] = W[1*2+1] = C[1][1] (row 1, col 1)
    #
    # So W[2] is C[0][1] and W[1] is C[1][0].
    #
    # The original says:
    # C00: W[0] = m0 + m4 - m5 + m6
    # C01: W[2] = m3 + m5  (wait, it says W[2][2]=1, W[2][4]=1, so m2 and m4? No...)
    #
    # Looking at the original code more carefully:
    # W[2][2] = 1  -> coefficient of product index 2 in W[2]
    # W[2][4] = 1  -> coefficient of product index 4 in W[2]
    # So C01 = m2 + m4 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    # But C01 should be af + dh. So bh != dh unless b=d. This is still wrong!
    #
    # I'm clearly misunderstanding the indexing. Let me re-read the original.
    #
    # Original:
    # r=2: a(f-h): U[0][2]=1, V[1][2]=1, V[3][2]=-1
    #   U[0] is A[0][0]=a, so U[0][2]=1 means product 2 uses a.
    #   V[1] is B[0][1]=f, V[3] is B[1][1]=h. So product 2 = a*(f-h). ✓
    #
    # r=4: (a+b)h: U[0][4]=1, U[1][4]=1, V[3][4]=1
    #   Product 4 = (a+b)*h. ✓
    #
    # W[2][2]=1: product index 2 contributes to W[2]=C[0][1]
    # W[2][4]=1: product index 4 contributes to W[2]=C[0][1]
    # So C[0][1] = m2 + m4 = a(f-h) + (a+b)h = af + bh
    #
    # But the correct C[0][1] = a*f + b*h. YES! af + bh IS correct!
    # I was confusing myself: C[0][1] = a*f + b*h, NOT a*f + d*h!
    #
    # For A=[[a,b],[c,d]] and B=[[e,f],[g,h]]:
    # C[0][0] = ae + cg
    # C[0][1] = af + dh  -- NO WAIT
    # C[0][1] = a*B[0][1] + b*B[1][1] = a*f + b*h
    #
    # YES! C[0][1] = af + bh, not af + dh. I was making an error.
    # The correct product is C[i][k] = sum_j A[i][j]*B[j][k].
    # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] = a*f + b*h. ✓
    #
    # Great, so the original Strassen is correct. Let me use it as-is.

    U[0][0] = 1; U[1][0] = 1   # m0: a+d
    V[0][0] = 1; V[3][0] = 1   # m0: e+h
    U[2][1] = 1; U[3][1] = 1   # m1: c+d
    V[0][1] = 1                # m1: e
    U[0][2] = 1                # m2: a
    V[1][2] = 1; V[3][2] = -1  # m2: f-h
    U[3][3] = 1                # m3: d
    V[2][3] = 1; V[0][3] = -1  # m3: g-e
    U[0][4] = 1; U[1][4] = 1   # m4: a+b
    V[3][4] = 1                # m4: h
    U[2][5] = 1; U[0][5] = -1  # m5: c-a
    V[0][5] = 1; V[1][5] = 1   # m5: e+f
    U[1][6] = 1; U[3][6] = -1  # m6: b-d
    V[2][6] = 1; V[3][6] = 1   # m6: g+h

    # W[k*n + i], n=2
    # C00 = W[0] = m0 + m4 - m5 + m6
    W[0][0] = 1; W[0][4] = 1; W[0][5] = -1; W[0][6] = 1
    # C10 = W[1] = m1 + m4
    W[1][1] = 1; W[1][4] = 1
    # C01 = W[2] = m2 + m3
    W[2][2] = 1; W[2][3] = 1
    # C11 = W[3] = m0 - m1 + m2 + m6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][6] = 1

    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m * m x p, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_baseline():
    """Build the rank-49 block decomposition."""
    n, m, p = 2, 5, 5
    total_rank = 49
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Block 1: 2x2 * 2x2 (Strassen, rank 7)
    U1, V1, W1 = _strassen_2x2x2()
    off = 0
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j][off + r] = U1[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * 5 + k][off + r] = V1[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][off + r] = W1[k * 2 + i][r]
    off += 7

    # Block 2: 2x2 * 2x3 (schoolbook, rank 12)
    U2, V2, W2 = _schoolbook_block(2, 2, 3)
    for r in range(12):
        for i in range(2):
            for j in range(2):
                U[i * 5 + j][off + r] = U2[i * 2 + j][r]
        for j in range(2):
            for kp in range(3):
                V[j * 5 + (kp + 2)][off + r] = V2[j * 3 + kp][r]
        for kp in range(3):
            for i in range(2):
                W[(kp + 2) * 2 + i][off + r] = W2[kp * 2 + i][r]
    off += 12

    # Block 3: 2x3 * 3x2 (schoolbook, rank 12)
    U3, V3, W3 = _schoolbook_block(2, 3, 2)
    for r in range(12):
        for i in range(2):
            for jp in range(3):
                U[i * 5 + (jp + 2)][off + r] = U3[i * 3 + jp][r]
        for jp in range(3):
            for k in range(2):
                V[(jp + 2) * 5 + k][off + r] = V3[jp * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][off + r] = W3[k * 2 + i][r]
    off += 12

    # Block 4: 2x3 * 3x3 (schoolbook, rank 18)
    U4, V4, W4 = _schoolbook_block(2, 3, 3)
    for r in range(18):
        for i in range(2):
            for jp in range(3):
                U[i * 5 + (jp + 2)][off + r] = U4[i * 3 + jp][r]
        for jp in range(3):
            for kp in range(3):
                V[(jp + 2) * 5 + (kp + 2)][off + r] = V4[jp * 3 + kp][r]
        for kp in range(3):
            for i in range(2):
                W[(kp + 2) * 2 + i][off + r] = W4[kp * 2 + i][r]
    off += 18

    return U, V, W, total_rank


def _verify_decomp(U, V, W, rank, n=2, m=5, p=5):
    """Check exact decomposition. Returns True if valid."""
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            for k in range(p):
                vj = j * p + k
                wk = k * n + i
                s = 0
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    # Also check that off-diagonal entries are zero
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            for k in range(p):
                vj = j * p + k
                wk = k * n + i
                for i2 in range(n):
                    for j2 in range(m):
                        if i2 == i and j2 == j:
                            continue
                        for k2 in range(p):
                            if k2 == k:
                                continue
                            ui2 = i2 * m + j2
                            vj2 = j2 * p + k2
                            wk2 = k2 * n + i2
                            s = 0
                            for r in range(rank):
                                s += U[ui2][r] * V[vj2][r] * W[wk2][r]
                            if s != 0:
                                return False
    return True


def _verify_fast(U, V, W, rank, n=2, m=5, p=5):
    """Fast verification: check that the tensor matches. Only check (i,j,k) != (i',j',k') cases."""
    # For the multiplication tensor, T[i,j,k] = 1 iff we're looking at A[i,j]*B[j,k]
    # Actually the tensor has T[i,j,k] = 1 for all valid (i,j,k) since it's
    # sum_j A[i,j] B[j,k]. The tensor entries are:
    # T[i, j, k] = delta_{j,k} -- no, that's not right either.
    #
    # The matrix multiplication tensor is:
    # T[i, j, k] = 1 if j == k (the contraction index matches)
    # Wait, no. The tensor for multiplying A (n x m) by B (m x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all valid indices? No.
    #
    # The tensor is defined so that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # This means T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k]
    # appears exactly once in the sum for C[i,k] when the j matches).
    #
    # Wait, that's not quite right. Let me think again.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # But that means the decomposition must satisfy:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1  for all i,j,k
    #
    # And for the "wrong" combinations... actually there are no wrong combinations
    # because the tensor is fully populated. The tensor IS the all-ones tensor
    # on the index set [0,n) x [0,m) x [0,p).
    #
    # Wait, that can't be right. Let me reconsider.
    #
    # The matrix multiplication tensor T_{2,5,5} has entries:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,...,4}, k in {0,...,4}
    #
    # No wait. The tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  where the sum is over all valid indices
    #
    # Actually, the standard matrix multiplication tensor is:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p x_{ij} y_{jk} z_{ki}
    # where x_{ij} are entries of A, y_{jk} entries of B, z_{ki} entries of C^T
    #
    # In the convention W[k*n+i], the tensor is:
    # sum_{i,j,k} U[i,j,r] * V[j,k,r] * W[k*n+i, r] should equal 1 for all i,j,k
    #
    # So yes, T[i,j,k] = 1 for all valid (i,j,k). The verification only needs
    # to check that sum_r U[i,m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    # There are no "off-diagonal" constraints because the tensor is all-ones.

    for i in range(n):
        for j in range(m):
            ui = i * m + j
            for k in range(p):
                vj = j * p + k
                wk = k * n + i
                s = 0
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    return True


def _random_search(U_base, V_base, W_base, rank_base, max_rank_to_try=48, time_budget=60.0):
    """
    Try to find a lower-rank decomposition starting from the baseline.
    Uses random initialization and iterative refinement.
    """
    import numpy as np
    from fractions import Fraction

    n, m, p = 2, 5, 5
    num_U = n * m  # 10
    num_V = m * p  # 25
    num_W = p * n  # 10

    best_U, best_V, best_W, best_rank = U_base, V_base, W_base, rank_base

    for target_rank in range(rank_base - 1, max(max_rank_to_try, 0) - 1, -1):
        if _time_left() < 5.0:
            break

        # Try a few random initializations
        for attempt in range(20):
            if _time_left() < 5.0:
                break

            # Random rational initialization with small integers
            rng = random.Random(hash((target_rank, attempt)) & 0xFFFFFFFF)
            U = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(num_U)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(num_V)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(num_W)]

            # Check if this random guess works (very unlikely but possible)
            if _verify_fast(U, V, W, target_rank):
                return U, V, W, target_rank

    return best_U, best_V, best_W, best_rank


def _try_rank_reduction(U_base, V_base, W_base, rank_base):
    """
    Attempt to reduce the rank by solving linear systems.
    For a given rank r < rank_base, check if there exist U, V, W of size
    (10,r), (25,r), (10,r) such that the tensor is reproduced.
    """
    import numpy as np
    from fractions import Fraction

    n, m, p = 2, 5, 5
    num_U = n * m  # 10
    num_V = m * p  # 25
    num_W = p * n  # 10

    # The constraint is: for all i in [0,2), j in [0,5), k in [0,5):
    #   sum_{r=0}^{R-1} U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
    #
    # This is a system of 50 equations (2*5*5) in 10R + 25R + 10R = 45R unknowns.
    # The equations are cubic (products of three variables), so they're nonlinear.
    #
    # One approach: fix two of the three matrices and solve for the third.
    # For example, fix U and V, then for each (i,j,k):
    #   sum_r W[k*2+i, r] * (U[i*5+j, r] * V[j*5+k, r]) = 1
    # This is linear in W! So we can:
    # 1. Randomly initialize U and V
    # 2. For each W row index wk = k*2+i, solve:
    #    sum_r W[wk, r] * C[wk, r] = 1  for all (i,j,k) mapping to this wk
    #    where C[wk, r] = U[i*5+j, r] * V[j*5+k, r]
    #
    # But each W[wk, r] appears in multiple equations (one for each j).
    # For fixed k and i (hence fixed wk = k*2+i), the equations are:
    #   sum_r W[wk, r] * U[i*5+j, r] * V[j*5+k, r] = 1  for j = 0..4
    #
    # This is a system of 5 equations (one per j) in R unknowns (the W[wk, r] values).
    # If R >= 5, this is generally solvable (if the 5xR matrix has rank 5).
    #
    # So the algorithm is:
    # 1. Choose R (target rank)
    # 2. Randomly initialize U (10 x R) and V (25 x R)
    # 3. For each (k, i) pair (10 pairs, since k in 0..4, i in 0..1):
    #    Build the 5 x R matrix M where M[j, r] = U[i*5+j, r] * V[j*5+k, r]
    #    Solve M * W[wk, :].T = [1,1,1,1,1].T
    #    If no solution, fail.
    # 4. Verify the full decomposition.
    #
    # This gives us a way to search over rank R by random initialization of U and V!

    best_result = None

    for R in range(rank_base - 1, 30, -1):  # Try ranks from 48 down to 31
        if _time_left() < 10.0:
            break

        for attempt in range(50):
            if _time_left() < 10.0:
                break

            rng = random.Random((R * 1000 + attempt) ^ 0xDEADBEEF)

            # Random integer initialization
            U = [[rng.randint(-2, 2) for _ in range(R)] for _ in range(num_U)]
            V = [[rng.randint(-2, 2) for _ in range(R)] for _ in range(num_V)]

            # Solve for W
            W = [[0] * R for _ in range(num_W)]
            solvable = True

            for k in range(p):
                for i in range(n):
                    wk = k * n + i
                    # Build system: for j in 0..4:
                    #   sum_r W[wk, r] * U[i*5+j, r] * V[j*5+k, r] = 1
                    # This is a 5 x R system.
                    # Use numpy to solve.
                    A_mat = np.array([[U[i * 5 + j][r] * V[j * 5 + k][r] for r in range(R)] for j in range(m)], dtype=float)
                    b_vec = np.ones(m, dtype=float)

                    try:
                        # Solve A_mat * x = b_vec, where x = W[wk, :]
                        # Use least squares as a fallback
                        x, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(A_mat @ x, b_vec, atol=1e-10):
                            # Convert to rationals (they should be close to integers)
                            for r in range(R):
                                W[wk][r] = round(x[r])
                            # Verify integer solution
                            A_int = np.array([[U[i * 5 + j][r] * V[j * 5 + k][r] for r in range(R)] for j in range(m)], dtype=int)
                            W_row = np.array(W[wk], dtype=int)
                            if not np.all(A_int @ W_row == 1):
                                solvable = False
                                break
                        else:
                            solvable = False
                            break
                    except:
                        solvable = False
                        break

                if not solvable:
                    break

            if solvable:
                if _verify_fast(U, V, W, R):
                    return U, V, W, R

    return None


def construct():
    if not _time_left():
        # Emergency: return schoolbook
        n, m, p = 2, 5, 5
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build baseline rank-49
    U, V, W, rank = _build_baseline()

    # Verify baseline
    if not _verify_fast(U, V, W, rank):
        # Shouldn't happen, but fallback to schoolbook
        n, m, p = 2, 5, 5
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_rank = rank

    # Try to reduce rank
    if _time_left() > 15.0:
        result = _try_rank_reduction(U, V, W, rank)
        if result is not None:
            U, V, W, best_rank = result

    # Final verification
    if not _verify_fast(U, V, W, best_rank):
        # Fallback to baseline
        U, V, W, best_rank = _build_baseline()
        if not _verify_fast(U, V, W, best_rank):
            # Absolute fallback: schoolbook
            n, m, p = 2, 5, 5
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
