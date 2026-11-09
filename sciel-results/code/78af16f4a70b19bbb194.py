"""Matrix multiplication 3,4,7 -- block/Strassen construction with verification + fallback."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_TOL = 1e-8


def _verify(U, V, W, n, m, p, tol=_TOL):
    """Exhaustively check the tensor identity.

    Required: for all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            u = U[i * m + j]
            for k in range(p):
                s = np.dot(u, V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_schoolbook(n, m, p):
    """Guaranteed-exact rank n*m*p decomposition (identity tensors)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _build_block(n, m, p, U, V, W, c0, i0, j0, k0, bi, bj, bk, sub):
    """Add one block's factor columns to U,V,W starting at column c0.

    bi,bj,bk are the (i,j,k) block sizes.
    sub is the tensor identity for this block shape (bi, bj, bk, rank).
    Returns the next free column index.
    """
    for a in range(bi):
        for b in range(bj):
            for d in range(bk):
                ui = i0 + a
                jj = j0 + b
                kk = k0 + d
                for r in range(sub.shape[3]):
                    u = sub[a, b, d, r]
                    if u:
                        U[ui * m + jj, c0 + r] = u
                    v = sub[b, d, 0, r] # Note: V index is (j, k) in block, so (b, d)
                    # Wait, V is (m, p). Block V is (bj, bk).
                    # The tensor identity is T[i, j, k, r] = U[i,j,r] * V[j,k,r] * W[k,i,r].
                    # So V[j,k,r] is the 2nd index of U, 1st index of V?
                    # Standard: T = sum U_i,r V_j,r W_k,r? No.
                    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                    # So for a block, we need:
                    # U_block[a,b,r], V_block[b,d,r], W_block[d,a,r].
                    # My previous _build_block used:
                    # u = sub[ui-i0, jj-j0, kk-k0, r] -> U[a,b,d,r]? No, U depends on i,j.
                    # The sub tensor should be defined as T[a,b,d,r] such that
                    # T[a,b,d,r] = U[a,b,r] * V[b,d,r] * W[d,a,r].
                    # But I am storing U, V, W separately.
                    # So I need to extract U, V, W from the identity.
                    # The identity is just the set of triples (u,v,w).
                    # Let's re-define how we store the sub-tensor.
                    # It should be 4D: [i, j, k, r] is the tensor value.
                    # But U[i,j,r], V[j,k,r], W[k,i,r] are the factors.
                    # We can't just index sub[i,j,k,r] to get U.
                    # We need to store the factors explicitly or reconstruct them.
                    # The previous code did:
                    # u = sub[ui - i0, jj - j0, kk - k0, r]
                    # This implies sub IS the U factor? No, that doesn't make sense.
                    # Let's look at the previous _build_block:
                    # u = sub[ui - i0, jj - j0, kk - k0, r]
                    # v = sub[jj - j0, kk - k0, 0, r]  <-- This is weird.
                    # w = sub[0, 0, kk - k0, r]        <-- This is weird.
                    
                    # The previous code was likely buggy or I misinterpreted it.
                    # Let's create a proper helper that takes U_block, V_block, W_block.
                    pass
    return c0 + sub.shape[3]

# We need to rewrite the block construction logic to be correct.
# Let's define the sub-tensors as (U, V, W) tuples.

def _make_strassen_222():
    """Returns (U, V, W) for 2x2x2 Strassen. Shapes: (4,7), (4,7), (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    # M2 = (A11+A22)(B21-B11)
    U[0,1] = 1; U[3,1] = 1
    V[1,1] = 1; V[0,1] = -1
    W[1,1] = 1; W[0,1] = 1 # W[k,i] -> W[1,0]? No.
    # Let's use the standard Strassen formulas for C = AB.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C11 = M1 + M4 - M5 + M6
    # C12 = M2 + M3
    # C21 = M2 + M5
    # C22 = M1 - M3 + M4 + M7
    
    # M1 = (A11+A22)(B11+B22)
    # A factors: A11, A22. B factors: B11, B22.
    # U[0,0]=1 (A11), U[3,0]=1 (A22).
    # V[0,0]=1 (B11), V[3,0]=1 (B22).
    # W[0,0]=1 (C11), W[3,0]=1 (C22).
    
    # M2 = (A11+A22)(B21-B11)
    # U[0,1]=1, U[3,1]=1.
    # V[1,1]=1 (B21), V[0,1]=-1 (B11).
    # C12 += M2, C21 += M2.
    # W[1,1]=1 (C12), W[1,0]=1 (C21). Note: W index is (k,i).
    # C12 is k=1, i=0? No.
    # Matrix C indices: C[i,k]? Or C[k,i]?
    # Task says: W[k*n+i, r]. So W index is (k,i).
    # C12 is row 0, col 1. So i=0, k=1. W[1*2+0] = W[2].
    # C21 is row 1, col 0. So i=1, k=0. W[0*2+1] = W[1].
    
    # Let's stick to 0-based indexing for blocks.
    # A[0,0], A[0,1], A[1,0], A[1,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[0,0], C[0,1], C[1,0], C[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1 # C00, C11
    
    # M2 = (A00+A11)(B10-B00)
    U[0,1]=1; U[3,1]=1
    V[1,1]=1; V[0,1]=-1 # B10, B00
    W[1,0]=1; W[0,1]=1 # C10 (i=1,k=0? No, C10 is row1,col0 -> i=1,k=0 -> W[0*2+1]=W[1]), C01 (i=0,k=1 -> W[1*2+0]=W[2])
    # Wait. W index is k*n+i.
    # C01: i=0, k=1. Index 1*2+0 = 2.
    # C10: i=1, k=0. Index 0*2+1 = 1.
    # So W[2,1] += 1, W[1,1] += 1.
    
    # M3 = (A00-A11)(B11+B01)
    U[0,2]=1; U[3,2]=-1
    V[3,2]=1; V[1,2]=1 # B11, B01
    W[2,2]=1; W[1,2]=1 # C01 (W[2]), C11 (W[3])?
    # C01: i=0, k=1 -> W[2].
    # C11: i=1, k=1 -> W[3].
    
    # M4 = (A11-A01)(B00+B10)
    U[3,3]=1; U[1,3]=-1
    V[0,3]=1; V[1,3]=1
    W[0,3]=1; W[1,3]=1 # C00 (W[0]), C10 (W[1])
    
    # M5 = (A10-A00)(B01+B11)
    U[2,4]=1; U[0,4]=-1
    V[1,4]=1; V[3,4]=1
    W[2,4]=1; W[3,4]=1 # C01 (W[2]), C11 (W[3])
    
    # M6 = (A01-A11)(B10+B11)
    U[1,5]=1; U[3,5]=-1
    V[1,5]=1; V[3,5]=1
    W[0,5]=1; W[2,5]=1 # C00 (W[0]), C01 (W[2])
    
    # M7 = (A00-A01)(B01-B10)
    U[0,6]=1; U[1,6]=-1
    V[1,6]=1; V[0,6]=-1
    W[3,6]=1 # C11 (W[3])?
    # C11 = M1 - M3 + M4 + M7.
    # M1: W[3]=1.
    # -M3: W[3]=-1.
    # M4: W[3]=0? M4 contributes to C00, C10.
    # M7: W[3]=1.
    # Sum: 1 - 1 + 0 + 1 = 1. Correct.
    
    return U, V, W

def _make_schoolbook_221():
    """2x2x1. Rank 4."""
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    # C00 = A00 B00
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # C01 = A01 B01? No, B is 2x1.
    # B = [[b0], [b1]].
    # C00 = A00 b0 + A01 b1? No.
    # C = A B.
    # C00 = A00 B00 + A01 B10. (B10 is b1).
    # C10 = A10 B00 + A11 B10.
    # So 4 mults: A00B00, A01B10, A10B00, A11B10.
    U[0,0]=1; V[0,0]=1; W[0,0]=1 # A00, B00, C00
    U[1,1]=1; V[1,1]=1; W[0,1]=1 # A01, B10, C00
    U[2,2]=1; V[0,2]=1; W[1,2]=1 # A10, B00, C10
    U[3,3]=1; V[1,3]=1; W[1,3]=1 # A11, B10, C10
    return U, V, W

def _make_schoolbook_122():
    """1x2x2. Rank 2."""
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((4, 2))
    # A = [a0, a1]. B = [[b00, b01], [b10, b11]].
    # C00 = a0 b00 + a1 b10.
    # C01 = a0 b01 + a1 b11.
    # Mult 1: a0 * (b00, b01) -> C00, C01.
    U[0,0]=1
    V[0,0]=1; V[1,0]=1 # B00, B01
    W[0,0]=1; W[2,0]=1 # C00 (k=0,i=0), C01 (k=1,i=0)
    
    # Mult 2: a1 * (b10, b11) -> C00, C01.
    U[1,1]=1
    V[2,1]=1; V[3,1]=1 # B10, B11
    W[0,1]=1; W[2,1]=1 # C00, C01
    return U, V, W

def _make_schoolbook_121():
    """1x2x1. Rank 2."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    # A = [a0, a1]. B = [b0, b1]. C = [c0].
    # c0 = a0 b0 + a1 b1.
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _build_block(n, m, p, U, V, W, c0, i0, j0, k0, bi, bj, bk, U_sub, V_sub, W_sub):
    """Add one block's factor columns to U,V,W starting at column c0."""
    for a in range(bi):
        for b in range(bj):
            ui = i0 + a
            jj = j0 + b
            for r in range(U_sub.shape[1]):
                u = U_sub[a * bj + b, r]
                if u:
                    U[ui * m + jj, c0 + r] = u
    for b in range(bj):
        for d in range(bk):
            jj = j0 + b
            kk = k0 + d
            for r in range(V_sub.shape[1]):
                v = V_sub[b * bk + d, r]
                if v:
                    V[jj * p + kk, c0 + r] = v
    for d in range(bk):
        for a in range(bi):
            kk = k0 + d
            ui = i0 + a
            for r in range(W_sub.shape[1]):
                w = W_sub[d * bi + a, r]
                if w:
                    W[kk * n + ui, c0 + r] = w
    return c0 + U_sub.shape[1]


def _construct_strassen_blocks(n, m, p):
    """Block decomposition: split rows 3=2+1, inner 4=2+2, cols 7=2+2+2+1."""
    U_222, V_222, W_222 = _make_strassen_222()
    U_221, V_221, W_221 = _make_schoolbook_221()
    U_122, V_122, W_122 = _make_schoolbook_122()
    U_121, V_121, W_121 = _make_schoolbook_121()
    
    # Calculate total rank
    # n=2 block:
    # 2 m-blocks * (3 p-blocks of size 2 + 1 p-block of size 1)
    # m-block 0: 3*7 + 1*4 = 25
    # m-block 1: 3*7 + 1*4 = 25
    # Total n=2: 50
    # n=1 block:
    # 2 m-blocks * (3 p-blocks of size 2 + 1 p-block of size 1)
    # m-block 0: 3*2 + 1*2 = 8
    # m-block 1: 3*2 + 1*2 = 8
    # Total n=1: 16
    total_rank = 50 + 16
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    # Row block 0 (rows 0,1)
    for jb, j0 in ((0, 0), (1, 2)):
        # p-blocks: 0-1, 2-3, 4-5, 6
        for kb, k0, bk in ((0, 0, 2), (1, 2, 2), (2, 4, 2), (3, 6, 1)):
            if bk == 2:
                col = _build_block(n, m, p, U, V, W, col, 0, j0, k0, 2, 2, 2, U_222, V_222, W_222)
            else:
                col = _build_block(n, m, p, U, V, W, col, 0, j0, k0, 2, 2, 1, U_221, V_221, W_221)
    
    # Row block 1 (row 2)
    for jb, j0 in ((0, 0), (1, 2)):
        for kb, k0, bk in ((0, 0, 2), (1, 2, 2), (2, 4, 2), (3, 6, 1)):
            if bk == 2:
                col = _build_block(n, m, p, U, V, W, col, 2, j0, k0, 1, 2, 2, U_122, V_122, W_122)
            else:
                col = _build_block(n, m, p, U, V, W, col, 2, j0, k0, 1, 2, 1, U_121, V_121, W_121)

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = 145.0

    U, V, W = _construct_schoolbook(n, m, p)
    best = (U, V, W)
    best_rank = U.shape[1]

    try:
        if time.time() < deadline:
            Us, Vs, Ws = _construct_strassen_blocks(n, m, p)
            if _verify(Us, Vs, Ws, n, m, p):
                best = (Us, Vs, Ws)
                best_rank = Us.shape[1]
    except Exception:
        pass

    U, V, W = best
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
