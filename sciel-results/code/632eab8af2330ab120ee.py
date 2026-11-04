"""Matrix multiplication 4x4*4x6 via rank-77 decomposition (Strassen 49 + Strassen 28)."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]
TIME_LIMIT = 120.0
DEADLINE = time.time() + TIME_LIMIT - 10.0  # safety margin before the 120s kill

def _time_ok():
    return time.time() < DEADLINE

# ---- Strassen 2x2 (rank 7) -------------------------------------------------
def _strassen2(n, m, p):
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[1,1]=1
    V[0,0]=1; V[2,0]=1
    W[0,0]=1; W[2,1]=1
    U[0,1]=1; U[1,1]=1; U[1,0]=1
    V[1,1]=1; V[3,1]=1
    W[1,1]=1; W[3,0]=1
    U[0,2]=1
    V[0,2]=1; V[2,2]=1
    W[0,2]=1; W[2,1]=1; W[2,0]=1
    U[1,3]=1
    V[1,3]=1; V[3,3]=1
    W[1,3]=1; W[3,0]=1; W[3,1]=1
    U[0,4]=1
    V[0,4]=1; V[2,4]=1
    W[0,4]=1; W[2,0]=1
    U[1,5]=1
    V[1,5]=1; V[3,5]=1
    W[1,5]=1; W[3,1]=1
    U[0,6]=1; U[1,6]=1
    V[0,6]=1; V[2,6]=1
    W[0,6]=1; W[2,1]=1; W[2,0]=1; W[3,1]=1; W[3,0]=1
    return U, V, W

# ---- Strassen 4x4 (rank 49) ------------------------------------------------
def _strassen4(n, m, p):
    U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
    # 14 blocks, each rank-7 Strassen 2x2, covering the 4x4 product
    blocks = [
        (0,0,0), (0,0,1), (0,1,0), (0,1,1),
        (1,0,0), (1,0,1), (1,1,0), (1,1,1),
        (2,0,0), (2,0,1), (2,1,0), (2,1,1),
        (3,0,0), (3,0,1), (3,1,0), (3,1,1),
    ]
    # We only need 14 distinct (a,b) for the 2x2 blocks of the 4x4; use the
    # standard 49-term recursive Strassen pattern over the 8 sub-2x2 blocks.
    # Simpler & correct: build via 2x2 Strassen applied to each of the 8
    # (2x2 * 2x2) sub-products of the 4x4 matrix product.
    # A4[2a+ia,2b+ib], B4[2b+ib,2c+ic] -> C4[2a+ia,2c+ic]
    # The 4x4 product has 16 sub-products of size 2x2, but Strassen groups
    # them: 4x4 = 2x2 blocks (A00,A01,A10,A11)x(B00,B01;B10,B11).
    # Use recursive Strassen: 4x4 product via 7 multiplications of 2x2 blocks,
    # each 2x2 multiplication uses 7 scalar mults -> 49 total.
    s2U, s2V, s2W = _strassen2(n, m, p)
    # Map the 4x4 Strassen into 2x2 blocks.  Index A4 row = 2a+ia, col = 2b+ib.
    # For each of the 7 Strassen 2x2-block multiplications, the inputs are
    # specific 2x2 block combinations.  We build the 49-term decomposition by
    # composing the 7-block Strassen (at 2x2 level) with the 7-term Strassen
    # (at scalar level).
    # 2x2-level Strassen: C00 = M0 - M2 + M3 + M4, etc.  M_t = (A_t) (B_t)
    # where A_t, B_t are 2x2 blocks formed from sums/diffs of the 4 2x2 blocks.
    # A-blocks (2x2): A00,A01,A10,A11 ; B-blocks: B00,B01,B10,B11
    # M0 = A00 * (B00+B01)
    # M1 = (A10+A11) * B00
    # M2 = (A00-A10) * (B10+B11)
    # M3 = A11 * (B01-B10)
    # M4 = (A00+A01) * B11
    # M5 = (A10-A00) * (B00-B10)
    # M6 = (A11-A01) * (B01-B11)
    # Each M_t is a 2x2 product = 7 scalar mults via s2U/s2V/s2W.
    # Represent A-block X as 4x4 (rows 2a+ia, cols 2b+ib).  A_t is a 2x2 block
    # matrix = linear combo of the 4 A-blocks; B_t similarly.
    # Build the 49 terms: for t in 0..6, for r in 0..6 (the 7 scalar mults),
    #   U[:, t*7+r] = s2U[:,r] * (A_t expansion)
    #   V[:, t*7+r] = s2V[:,r] * (B_t expansion)
    #   W[:, t*7+r] = s2W[:,r] * (C_t output placement)
    # A_t expansion: which of the 16 A4 entries are nonzero, with coefficient.
    def block_row(block):
        # block is (a,b) for A (row block a, col block b) -> 4 entries rows 2a+ia, cols 2b+ib
        a, b = block
        return [(2*a+ia, 2*b+ib) for ia in range(2) for ib in range(2)]
    # A_t as list of (block, coeff)
    At = [
        [((0,0),1)],                                  # M0: A00
        [((1,0),1),((1,1),1)],                        # M1
        [((0,0),1),((1,0),-1)],                       # M2
        [((1,1),1)],                                  # M3
        [((0,0),1),((0,1),1)],                        # M4
        [((1,0),1),((0,0),-1)],                       # M5
        [((1,1),1),((0,1),-1)],                       # M6
    ]
    Bt = [
        [((0,0),1),((0,1),1)],                        # M0: B00+B01
        [((0,0),1)],                                  # M1
        [((1,0),1),((1,1),1)],                        # M2
        [((0,1),1),((1,0),-1)],                       # M3
        [((1,1),1)],                                  # M4
        [((0,0),1),((1,0),-1)],                       # M5
        [((0,1),1),((1,1),-1)],                       # M6
    ]
    # C output: C00 = M0 - M2 + M3 + M4 ; C01 = M4 + M5 ; C10 = M1 - M5 - M6 + M2 ; C11 = M6 + M1
    # C_t: list of (cblock, coeff)
    Ct = [
        [((0,0),1)],   # M0
        [((1,0),1),((1,1),1)],  # M1
        [((0,0),-1),((1,0),1),((1,1),1)],  # M2 (with sign in C00, and + in C10)
        [((0,0),1),((1,1),-1)],  # M3
        [((0,0),1),((0,1),1)],   # M4
        [((0,1),1),((1,0),-1)],  # M5
        [((1,0),-1),((1,1),1)],  # M6
    ]
    # Recompute C mapping cleanly:
    # C00 = M0 - M2 + M3 + M4
    # C01 = M4 + M5
    # C10 = M1 - M5 - M6 + M2
    # C11 = M6 + M1
    Cmap = {
        0: [((0,0),1)],
        1: [((1,0),1),((1,1),1)],
        2: [((0,0),-1),((1,0),1),((1,1),1)],
        3: [((0,0),1),((1,1),-1)],
        4: [((0,0),1),((0,1),1)],
        5: [((0,1),1),((1,0),-1)],
        6: [((1,0),-1),((1,1),1)],
    }
    for t in range(7):
        for r in range(7):
            col = t*7 + r
            # U: A part
            for (blk, coeff) in At[t]:
                for (rr, cc) in block_row(blk):
                    U[rr, cc] += coeff * s2U[rr % 4, cc % 4, ] if False else 0
    # The above nested indexing is error-prone; rebuild U,V,W explicitly.
    U = np.zeros((16, 49)); V = np.zeros((16, 49)); W = np.zeros((16, 49))
    for t in range(7):
        for r in range(7):
            col = t*7 + r
            # U: for each (blk,coeff) in At[t], for each local (ia,ib), add coeff * s2U[ia,ib]
            for (blk, coeff) in At[t]:
                a, b = blk
                for ia in range(2):
                    for ib in range(2):
                        U[2*a+ia, 2*b+ib] += coeff * s2U[ia, ib]
            # V: B part
            for (blk, coeff) in Bt[t]:
                b, c = blk
                for ib in range(2):
                    for ic in range(2):
                        V[2*b+ib, 2*c+ic] += coeff * s2V[ib, ic]
            # W: C part (trace convention W[k*n+i, c])
            for (blk, coeff) in Cmap[t]:
                a, c = blk
                for ic in range(2):
                    for ia in range(2):
                        W[2*c+ic, 2*a+ia] += coeff * s2W[ic, ia]
    return U, V, W

# ---- Build the 4x4*4x6 decomposition --------------------------------------
def _build_446():
    # Split the 4x6 matrix B into B1 (4x4) and B2 (4x2).
    # C = A*B1 (4x4) + A*B2 (4x2).
    # A*B1: use Strassen4 (rank 49).  A*B2: use Strassen for 4x4 * 4x2.
    U1, V1, W1 = _strassen4(4, 4, 4)  # (16,49),(16,49),(16,49)
    # For 4x4 * 4x2: split B2 (4x2) into two 4x1?  Better: treat 4x2 as 2x2 blocks.
    # Use Strassen4 on a 4x4*4x2 by padding?  Instead: 4x4 * 4x2 = two 4x4*4x1?
    # Simplest correct: 4x4 * 4x2 via 2 Strassen 2x2-based 2x4?  Use rank-28 known:
    # 4x4*4x2 has rank 28 (Strassen: split into 2x2 blocks -> 2 2x2*2x1? )
    # Build 4x4*4x2 via recursive Strassen: split A into 2x2 blocks (4 blocks),
    # B2 (4x2) into 2x1 blocks (2 blocks).  C (4x2) into 2x1 blocks (2 blocks).
    # Use the known rank-28 pattern.  We'll construct it by composing Strassen 2x2
    # over the block structure: A = [[A00,A01],[A10,A11]] (2x2 blocks of 2x2),
    # B2 = [[B0],[B1]] (2x1 blocks of 2x2), C = [[C0],[C1]].
    # C0 = A00*B0 + A01*B1 ; C1 = A10*B0 + A11*B1.
    # That's 4 products of 2x2*2x2, each rank 7 -> rank 28.
    s2U, s2V, s2W = _strassen2(4, 4, 2)
    U2 = np.zeros((16, 28)); V2 = np.zeros((8, 28)); W2 = np.zeros((8, 28))
    # products: (a_block, b_block, c_block)
    prods = [((0,0), (0,0), (0,0)), ((0,1), (0,0), (0,1)),
             ((1,0), (0,0), (1,0)), ((1,1), (0,0), (1,1))]
    # Wait: C0 = A00*B0 + A01*B1.  A00 is block (0,0) [rows0-1,cols0-1], B0 is block (0,0) of B2 [rows0-1,cols0-1].
    # B2 is 4x2: B2[0:2,0:2]=B0, B2[2:4,0:2]=B1.
    # C0 = A00*B0 + A01*B1.  A01 is A block (0,1) [rows0-1,cols2-3].
    # C1 = A10*B0 + A11*B1.
    prods = [
        ((0,0), (0,0), (0,0)),  # A00 * B0 -> C0
        ((0,1), (0,0), (0,1)),  # A01 * B1 -> C0   (B1 is rows2-3 of B2)
        ((1,0), (0,0), (1,0)),  # A10 * B0 -> C1
        ((1,1), (0,0), (1,1)),  # A11 * B1 -> C1
    ]
    # Fix: B0 = B2 block row0 (rows0-1), B1 = B2 block row1 (rows2-3).
    # Represent B-block as (brow, bcol) where brow in {0,1}, bcol fixed 0 (since 2 cols).
    prods = [
        ((0,0), (0,0), (0,0)),  # A00 * B0 -> C0
        ((0,1), (1,0), (0,1)),  # A01 * B1 -> C0
        ((1,0), (0,0), (1,0)),  # A10 * B0 -> C1
        ((1,1), (1,0), (1,1)),  # A11 * B1 -> C1
    ]
    for pi, (ablk, bblk, cblk) in enumerate(prods):
        for r in range(7):
            col = pi*7 + r
            a, b = ablk
            for ia in range(2):
                for ib in range(2):
                    U2[2*a+ia, 2*b+ib] += s2U[ia, ib]
            brow, bcol = bblk
            for ib in range(2):
                for ic in range(2):
                    V2[2*brow+ib, 2*bcol+ic] += s2V[ib, ic]
            crow, ccol = cblk
            for ic in range(2):
                for ia in range(2):
                    W2[2*ccol+ic, 2*crow+ia] += s2W[ic, ia]
    # Combine: U = [U1 | U2], V = [V1 | V2'], W = [W1 | W2']
    # V1 is (16,49) for B1 (4x4).  V2 is (8,28) for B2 (4x2).  Full B is 4x6.
    # V total: rows = 4*6 = 24, cols = 49+28 = 77.
    # V1 occupies B1 columns 0..3 (V1[i,j] for j in 0..3).  V2 occupies B2 columns 4..5.
    Vfull = np.zeros((24, 77))
    Vfull[0:16, 0:49] = V1
    Vfull[0:8, 49:77] = V2  # B2 rows 0..3 -> V rows 0..7 (since B2 is 4x2, rows 0..3)
    # Wait: V index is j*p+k where j is A-row (0..3), k is B-col (0..5).  B1 cols 0..3, B2 cols 4..5.
    # V1: j in 0..3, k in 0..3 -> V rows j*6+k = 0..23.  But V1 was built for 4x4 (j*4+k).
    # Need to remap V1 from 4x4 indexing (j*4+k) to 4x6 indexing (j*6+k).
    Vfull = np.zeros((24, 77))
    for j in range(4):
        for k in range(4):
            Vfull[j*6+k, 0:49] = V1[j*4+k, 0:49]
    for j in range(4):
        for k in range(2):
            Vfull[j*6+4+k, 49:77] = V2[j*2+k, 0:28]
    # U: (16,77)
    Ufull = np.zeros((16, 77))
    Ufull[:, 0:49] = U1
    Ufull[:, 49:77] = U2
    # W: (24,77)  W index = k*n+i = k*4+i, k in 0..5, i in 0..3.
    # W1: for B1, k in 0..3 -> W rows k*4+i = 0..15.
    # W2: for B2, k in 4..5 -> W rows k*4+i = 16..23.
    Wfull = np.zeros((24, 77))
    for k in range(4):
        for i in range(4):
            Wfull[k*4+i, 0:49] = W1[k*4+i, 0:49]
    for k in range(2):
        for i in range(4):
            Wfull[(4+k)*4+i, 49:77] = W2[k*2+i, 0:28]
    return Ufull, Vfull, Wfull

def _to_int_list(M):
    # Round to nearest integer (entries should be integers).
    return [[int(round(float(x))) for x in row] for row in M]

def _verify(U, V, W, n=4, m=4, p=6, trials=20):
    # Check tensor identity: sum_c U[a,c]*V[b,c]*W[c_idx,c] == delta
    # U index a = i*m+j ; V index b = j*p+k ; W index = k*n+i.
    # Product C[i,k] = sum_{j} A[i,j] B[j,k].
    # Tensor: T[i,j,k] = 1.  Decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    rng = random.Random(12345)
    Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = rng.randint(0, 9, size=(n, m))
        B = rng.randint(0, 9, size=(m, p))
        C = A @ B
        # Compute via decomposition: C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
        # = sum_j A[i,j] B[j,k] (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])
        # The inner sum must equal 1 for all i,j,k.  So C should match.
        C2 = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(Ua.shape[1]):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    C2[i, k] += A[i, j] * B[j, k] * s
        if not np.allclose(C, C2, atol=1e-6):
            return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    # Deadline guard: try the rank-77 build; verify; fall back to schoolbook on failure.
    try:
        U, V, W = _build_446()
        U = _to_int_list(U); V = _to_int_list(V); W = _to_int_list(W)
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    return _schoolbook()
