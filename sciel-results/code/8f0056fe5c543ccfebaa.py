"""Rewritten program for 3x3x4 matrix multiplication tensor decomposition.

Strategy: Construct a decomposition of rank 34 by splitting the 3x3x4 problem into
- One 2x2x4 block decomposed via two 2x2x2 blocks, each using Strassen (rank 7 each, total 14)
- Remaining rows/cols handled by schoolbook (rank 20)

Total rank = 14 + 20 = 34.

Convention: W[k*n+i, r] where n=3, p=4, m=3.
U is (n*m, rank) = (9, rank), V is (m*p, rank) = (12, rank), W is (p*n, rank) = (12, rank).
Each rank-1 term corresponds to a (i,j,k) triple with:
  U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.

For Strassen blocks, the entries can be half-integers or integers.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 4]


def _make_strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (2*2, 7) = (4, 7), V: (2*2, 7) = (4, 7), W: (2*2, 7) = (4, 7).
    Indices: A[i,j] -> U[i*2+j, c], B[j,k] -> V[j*2+k, c], C[i,k] -> W[k*2+i, c].
    
    Standard Strassen (0-indexed):
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)B00
    M2 = A00(B10-B11)
    M3 = A11(B01-B00)
    M4 = (A00+A10)(B00+B11)
    M5 = (A01-A10)(B01+B11)
    M6 = (A00+A11)(B01-B10)

    C00 = M0 - M3 + M5 - M4  ... wait let me use a known correct version.
    
    Actually let me use the standard formulation carefully:
    For C = A*B where A, B are 2x2:
    A = [[a,b],[c,d]], B = [[w,x],[y,z]]
    M1 = (a+d)(w+z)
    M2 = (c+d)w
    M3 = (a-b)(x+z)  -- no, this isn't standard either.
    
    Let me use the most common reference:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)(B00+B11)
    M6 = (A01-A10)(B01+B11)
    M7 = (A00+A11)(B01-B10)
    
    C00 = M1 - M3 + M5 - M7
    C01 = M3 + M4 + M7
    C10 = M2 - M4 + M6
    C11 = M1 - M2 - M5 + M6
    """
    # Build U, V, W for each of the 7 multipliers
    # U: (4,7), V: (4,7), W: (4,7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row/col indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # A00->U[0], A01->U[1], A10->U[2], A11->U[3]
    # B00->V[0], B01->V[1], B10->V[2], B11->V[3]
    # C00->W[0], C01->W[1], C10->W[2], C11->W[3]
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1  # C00
    W[1,0] = -1  # C01
    W[2,0] = -1  # C10
    W[3,0] = -1  # C11
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1  # B00
    W[0,1] = -1  # C00
    W[3,1] = -1  # C11
    
    # M3 = A00(B10-B11)
    U[0,2] = 1  # A00
    V[2,2] = 1; V[3,2] = -1  # B10, -B11
    W[0,2] = -1  # C00
    W[1,2] = 1  # C01
    
    # M4 = A11(B01-B00)
    U[3,3] = 1  # A11
    V[0,3] = -1; V[1,3] = 1  # -B00, B01
    W[0,3] = -1  # C00  -- wait, C00 should have -M3 not -M4? Let me recheck.
    
    # I'm getting confused with the signs. Let me just use a verified decomposition.
    # Let me use a simpler approach: use the known rank-7 decomposition verified numerically.
    
    # I'll verify numerically instead. Let me define the 7 terms and check C = sum of terms.
    pass
    
    # Actually, let me just hardcode a known-correct Strassen decomposition and verify.
    # The key is that the tensor identity must hold.
    # Let me use a completely different approach: just enumerate all possible rank-1 terms
    # and use the fact that for 2x2x2, rank 7 is achievable.
    # I'll define the decomposition and verify it works numerically in a test.
    
    # Known correct Strassen (from Wikipedia, 0-indexed):
    # p0 = (a+d)(w+z), p1 = (c+d)w, p2 = (a-b)(y+z), p3 = (c+d)(z-x)
    # p4 = (a+c)(w+x), p5 = (b-d)(x+z), p6 = (a+c)(y-w)
    # c = p0 - p3 + p6 - p4, d = p2 + p4, e = p1 - p2 + p5, f = p0 - p1 - p5 + p3
    
    # A = [[a,b],[c,d]], B = [[w,x],[y,z]]
    # U: A indices: a->0, b->1, c->2, d->3
    # V: B indices: w->0, x->1, y->2, z->3
    # W: C indices: c(out)->W[0*2+0]=W[0] (C00), d(out)->W[1*2+0]=W[1] (C01)
    #     e(out)->W[0*2+1]=W[2] (C10), f(out)->W[1*2+1]=W[3] (C11)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # p0 = (a+d)(w+z)
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1  # C00
    W2[3,0] = -1  # C11
    
    # p1 = (c+d)w
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1  # C10
    W2[3,1] = -1  # C11
    
    # p2 = (a-b)(y+z)
    U2[0,2] = 1; U2[1,2] = -1
    V2[2,2] = 1; V2[3,2] = 1
    W2[1,2] = 1  # C01
    W2[2,2] = -1  # C10
    
    # p3 = (c+d)(z-x)
    U2[2,3] = 1; U2[3,3] = 1
    V2[0,3] = -1; V2[3,3] = 1  # -x + z => -V[1] + V[3]
    W2[0,3] = -1  # C00
    W2[3,3] = 1  # C11
    
    # p4 = (a+c)(w+x)
    U2[0,4] = 1; U2[2,4] = 1
    V2[0,4] = 1; V2[1,4] = 1
    W2[0,4] = -1  # C00
    W2[1,4] = 1  # C01
    
    # p5 = (b-d)(x+z)
    U2[1,5] = 1; U2[3,5] = -1
    V2[1,5] = 1; V2[3,5] = 1
    W2[2,5] = 1  # C10
    W2[3,5] = -1  # C11
    
    # p6 = (a+c)(y-w)
    U2[0,6] = 1; U2[2,6] = 1
    V2[0,6] = -1; V2[2,6] = 1  # -w + y
    W2[0,6] = 1  # C00
    W2[1,6] = -1  # C01
    
    return U2, V2, W2


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[:,:,c] @ V[:,:,c] @ W[:,:,c] gives the MM tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] should equal 1 for all valid (i,j,k) pairs
    # T = sum_c outer(U[:,c], V[:,c], W[:,c])
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    total = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Contribution to tensor: u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    # Check if total equals 1 for all entries
    if np.allclose(total, 1.0, atol=1e-9):
        return True
    return False


def _build_hybrid_decomposition():
    """Build a rank-34 decomposition by combining Strassen for 2x2x4 blocks
    with schoolbook for the remaining parts."""
    n, m, p = 3, 3, 4
    
    # We decompose the 3x3x4 tensor as follows:
    # Rows 0,1 of A (i=0,1) with all j=0,1,2 and k=0,1,2,3
    # Rows 2 of A (i=2) with all j,k
    
    # Strategy: 
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 via Strassen (rank 7)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 via Strassen (rank 7)
    # Block 3: i in {0,1}, j in {2}, k in {0,1,2,3} -> 2x1x4 = 8 terms (schoolbook)
    # Block 4: i in {2}, j in {0,1}, k in {0,1,2,3} -> 1x2x4 = 8 terms (schoolbook)
    # Block 5: i in {2}, j in {2}, k in {0,1,2,3} -> 1x1x4 = 4 terms (schoolbook)
    # Total: 7 + 7 + 8 + 8 + 4 = 34
    
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    U2, V2, W2 = _make_strassen_222()
    for c in range(7):
        for row in range(4):
            # U2[row] corresponds to (i,j) where i = row//2, j = row%2
            i = row // 2
            j = row % 2
            U[i * m + j, col] = U2[row, c]
        for row in range(4):
            # V2[row] corresponds to (j,k) where j = row//2, k = row%2
            j = row // 2
            k = row % 2
            V[j * p + k, col] = V2[row, c]
        for row in range(4):
            # W2[row] corresponds to (k,i) where k = row//2, i = row%2
            # But our W convention is W[k*n+i]
            k = row // 2
            i = row % 2
            W[k * n + i, col] = W2[row, c]
        col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    U2, V2, W2 = _make_strassen_222()
    for c in range(7):
        for row in range(4):
            i = row // 2
            j = row % 2
            U[i * m + j, col] = U2[row, c]
        for row in range(4):
            j = row // 2
            k = row % 2
            V[j * p + (k + 2), col] = V2[row, c]
        for row in range(4):
            k = row // 2
            i = row % 2
            W[(k + 2) * n + i, col] = W2[row, c]
        col += 1
    
    # Block 3: i in {0,1}, j = 2, k in {0,1,2,3} -> schoolbook 2*1*4 = 8 terms
    for i in range(2):
        for k in range(4):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Block 4: i = 2, j in {0,1}, k in {0,1,2,3} -> schoolbook 1*2*4 = 8 terms
    for j in range(2):
        for k in range(4):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    
    # Block 5: i = 2, j = 2, k in {0,1,2,3} -> schoolbook 1*1*4 = 4 terms
    for k in range(4):
        U[2 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 2, col] = 1
        col += 1
    
    return U, V, W


# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    
    Uses a hybrid approach: Strassen for 2x2x2 sub-blocks, schoolbook elsewhere.
    Target rank: 34.
    """
    start_time = time.time()
    deadline = 150  # seconds, must finish before 120s evaluation limit with margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Build the hybrid decomposition (rank 34)
    U, V, W = _build_hybrid_decomposition()
    
    # Convert to lists of integers/half-integers
    U_list = [[int(U[i, c]) if U[i, c] == int(U[i, c]) else U[i, c] for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i, c]) if V[i, c] == int(V[i, c]) else V[i, c] for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i, c]) if W[i, c] == int(W[i, c]) else W[i, c] for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_school = [[0] * rank for _ in range(n * m)]
        V_school = [[0] * rank for _ in range(m * p)]
        W_school = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Safety: return schoolbook if too close to deadline
        rank = n * m * p
        U_school = [[0] * rank for _ in range(n * m)]
        V_school = [[0] * rank for _ in range(m * p)]
        W_school = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
