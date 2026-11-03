"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 140

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[2,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1
    return U, V, W

def _rank1_221():
    """Rank-3 decomposition for 2x2 * 2x1 -> 2x1. Returns (U,V,W) each (4,3),(2,3),(2,3)."""
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Use: P1=A00*B0, P2=A10*B0, P3=(A01+A11)*B1
    U = np.zeros((4, 3))
    V = np.zeros((2, 3))
    W = np.zeros((2, 3))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[2,1]=1; V[0,1]=1; W[1,1]=1
    U[1,2]=1; U[3,2]=1; V[1,2]=1
    W[0,2]=1; W[1,2]=1
    return U, V, W

def _rank1_121():
    """Rank-2 decomposition for 1x2 * 2x1 -> 1x1. Returns (U,V,W) each (2,2),(2,2),(1,2)."""
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W

def _build_tensor(n, m, p):
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                tensor[a,b,c] = 1.0
    return tensor

def _verify(U, V, W, tensor, tol=1e-9):
    r = U.shape[1]
    for a in range(tensor.shape[0]):
        for b in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                val = float(np.dot(U[a,:], V[b,:]*W[c,:]))
                if abs(val - tensor[a,b,c]) > tol:
                    return False
    return True

def _to_lists(U, V, W):
    def conv(mat):
        return [[int(x) if abs(x-round(x))<1e-9 else float(x) for x in row] for row in mat]
    return conv(U), conv(V), conv(W)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook fallback (rank 84)
    rank_sb = n*m*p
    U_sb = [[0]*rank_sb for _ in range(n*m)]
    V_sb = [[0]*rank_sb for _ in range(m*p)]
    W_sb = [[0]*rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Block decomposition: rank 49
    try:
        Uss, Vss, Wss = _strassen_222()
        Ur1, Vr1, Wr1 = _rank1_221()
        Ur2, Vr2, Wr2 = _rank1_121()
    except Exception:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Rank components: 
    # Part 1: 4 * 7 = 28
    # Part 2: 2 * 7 + 2 * 3 = 20
    # Part 3: 4 * 4 = 16 (each 1x2x2 is rank 4)
    # Part 4: 2 * 4 + 2 * 2 = 12 (2 blocks 1x2x2, 2 blocks 1x2x1)
    # Total: 28 + 20 + 16 + 12 = 76
    # Use 84 for safety (schoolbook)
    total_rank = 84
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))

    def map_U_2x2(i, j):  # A is 3x4, rows i in {0,1}, cols j in {0,1,2,3}
        return i*4 + j
    def map_V_2x2(j, k):  # B is 4x7, rows j in {0,1,2,3}, cols k in {0,1,2,3,4,5,6}
        return j*7 + k
    def map_W_2x2(k, i):  # C is 3x7, cols k, rows i
        return k*3 + i

    # --- 6 Strassen blocks for A[0:2,:4] * B[:4, 0:4] -> C[0:2, 0:4] ---
    # Blocks: (i_block, k_block) where i_block in {0,1} (A rows 0-1,2-3... wait)
    # A is 3x4, we use rows 0,1 for the 2x2 A-blocks. A cols 0,1,2,3.
    # A-block (i,j): A[i:i+2, j:j+2] for i in {0} (only one 2-row block since we use 2 rows), j in {0,1}
    # Wait: A has 3 rows, we use rows 0,1. So i_block=0 covers A rows 0,1.
    # j_blocks: cols 0-1 (j_block=0), cols 2-3 (j_block=1)
    # B blocks: B rows 0-1 (j_block=0), B rows 2-3 (j_block=1); B cols 0-1 (k_block=0), cols 2-3 (k_block=1)
    # C output: C rows 0,1; C cols 0-1 (k_block=0), C cols 2-3 (k_block=1)

    # For 2x4 * 4x4: split A into A[:, 0:2] and A[:, 2:4] (two 2x2), B into B[0:2, :] and B[2:4, :] (two 2x2 vertically)
    # C[0:2, 0:4] = A[0:2, 0:2]*B[0:2, 0:4] + A[0:2, 2:4]*B[2:4, 0:4]
    # But B[0:2, 0:4] is 2x4, not 2x2. Need to split further.
    # Actually: C[0:2, k] for k in 0..3 = sum_{j=0}^{3} A[0:2,j]*B[j,k]
    # = A[0:2,0:2]*B[0:2,k] + A[0:2,2:4]*B[2:4,k]
    # For k in 0,1: B[0:2,k] is 2x1... this doesn't fit 2x2 Strassen directly.

    # Let me reconsider. For 2x4 * 4x4 -> 2x4:
    # Split into 4 blocks of 2x2 * 2x2 -> 2x2:
    # A = [A00 A01] where A00=A[0:2,0:2], A01=A[0:2,2:4]
    # B = [B0; B1] where B0=B[0:2,0:4], B1=B[2:4,0:4]  -- these are 2x4
    # This doesn't work cleanly. Let me split B columns too.

    # Better: split C[0:2, 0:4] into 4 sub-blocks C[0:2, k:k+2] for k in {0,2}
    # C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
    #   = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]  (each 2x2*2x2)
    # C[0:2, 2:4] = A[0:2, 0:4] * B[0:4, 2:4]
    #   = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]  (each 2x2*2x2)
    # Total: 4 Strassen blocks, rank 4*7=28 for C[0:2, 0:4]

    # C[0:2, 4:7] = A[0:2, 0:4] * B[0:4, 4:7]
    #   = A[0:2,0:2]*B[0:2,4:7] + A[0:2,2:4]*B[2:4,4:7]
    #   Each is 2x2 * 2x3. Split B's 3 cols into 2+1:
    #   A[0:2,0:2]*B[0:2,4:6] (2x2*2x2, rank 7) + A[0:2,0:2]*B[0:2,6:7] (2x2*2x1, rank 3)
    #   A[0:2,2:4]*B[2:4,4:6] (2x2*2x2, rank 7) + A[0:2,2:4]*B[2:4,6:7] (2x2*2x1, rank 3)
    # Total for C[0:2,4:7]: 7+3+7+3 = 20

    # C[2, 0:4] = A[2, 0:4] * B[0:4, 0:4]  (1x4 * 4x4)
    #   = A[2,0:2]*B[0:2,0:4] + A[2,2:4]*B[2:4,0:4]
    #   Each 1x2 * 2x4. Split B cols:
    #   A[2,0:2]*B[0:2,0:2] (1x2*2x2) + A[2,0:2]*B[0:2,2:4] (1x2*2x2)
    #   A[2,2:4]*B[2:4,0:2] (1x2*2x2) + A[2,2:4]*B[2:4,2:4] (1x2*2x2)
    #   Each 1x2*2x2 -> 1x2: rank 2 each. Total: 8.

    # C[2, 4:7] = A[2, 0:4] * B[0:4, 4:7]
    #   = A[2,0:2]*B[0:2,4:7] + A[2,2:4]*B[2:4,4:7]
    #   1x2*2x3: split into 1x2*2x2 (rank 2) + 1x2*2x1 (rank 2)
    #   Two such: total 2*(2+2) = 8

    # Total: 28 + 20 + 8 + 8 = 64. Hmm, that's worse than my earlier estimate.

    # Let me redo with a cleaner partition. Actually let me just carefully build it.
    # I'll use explicit rank-1 tensor terms.

    # Reset
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0

    # Helper to add a block
    def add_block(Ub, Vb, Wb, u_idx, v_idx, w_idx):
        nonlocal col
        r = Ub.shape[1]
        for t in range(r):
            for (ri, val) in enumerate(Ub[:, t]):
                if val != 0:
                    U[u_idx[ri], col] += val
            for (ri, val) in enumerate(Vb[:, t]):
                if val != 0:
                    V[v_idx[ri], col] += val
            for (ri, val) in enumerate(Wb[:, t]):
                if val != 0:
                    W[w_idx[ri], col] += val
            col += 1

    # u_idx for 2x2 A block at A[i0, j0]: indices [i0*4+j0, (i0+1)*4+j0, i0*4+j0+1, (i0+1)*4+j0+1]
    # v_idx for 2x2 B block at B[j0, k0]: [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
    # w_idx for 2x2 C block at C[i0, k0]: [k0*3+i0, k0*3+i0+1, (k0+1)*3+i0, (k0+1)*3+i0+1]
    # Wait, W index = k*n+i = k*3+i. For 2x2 C block at row i0, col k0:
    # C[i0, k0] -> k0*3+i0
    # C[i0, k0+1] -> (k0+1)*3+i0
    # C[i0+1, k0] -> k0*3+(i0+1)
    # C[i0+1, k0+1] -> (k0+1)*3+(i0+1)

    def u22(i0, j0):
        return [i0*4+j0, (i0+1)*4+j0, i0*4+j0+1, (i0+1)*4+j0+1]
    def v22(j0, k0):
        return [j0*7+k0, j0*7+k0+1, (j0+1)*7+k0, (j0+1)*7+k0+1]
    def w22(i0, k0):
        return [k0*3+i0, (k0+1)*3+i0, k0*3+(i0+1), (k0+1)*3+(i0+1)]

    # For 2x2 * 2x1 (B is 2 rows, 1 col): v_idx = [j0*7+k0, (j0+1)*7+k0]
    def v21(j0, k0):
        return [j0*7+k0, (j0+1)*7+k0]
    def w21(i0, k0):
        return [k0*3+i0, k0*3+(i0+1)]

    # For 1x2 * 2x2: u_idx = [i*4+j0, i*4+j0+1], v_idx = v22, w_idx = [k0*3+i, (k0+1)*3+i]
    def u12(i, j0):
        return [i*4+j0, i*4+j0+1]
    def w12(i, k0):
        return [k0*3+i, (k0+1)*3+i]

    # For 1x2 * 2x1: u_idx = u12, v_idx = v21, w_idx = [k0*3+i]
    def w11(i, k0):
        return [k0*3+i]

    # --- Part 1: C[0:2, 0:4] = 4 Strassen blocks, rank 28 ---
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,0), w22(0,0))  # A00*B00
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,0), w22(0,0))  # A01*B10
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,2), w22(0,2))  # A00*B01
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,2), w22(0,2))  # A01*B11

    # --- Part 2: C[0:2, 4:7] ---
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    add_block(Uss, Vss, Wss, u22(0,0), v22(0,4), w22(0,4))
    add_block(Uss, Vss, Wss, u22(0,2), v22(2,4), w22(0,4))
    # C[0:2, 6:7] = A[0:2,0:2]*B[0:2,6:7] + A[0:2,2:4]*B[2:4,6:7]
    add_block(Ur1, Vr1, Wr1, u22(0,0), v21(0,6), w21(0,6))
    add_block(Ur1, Vr1, Wr1, u22(0,2), v21(2,6), w21(0,6))

    # Helper to add a 1x2 * 2x2 block (rank 4) using two 1x2 * 2x1 blocks (rank 2 each)
    def add_block_1x2x2(i, j0, k0):
        nonlocal col
        u_idx = u12(i, j0)
        v_full = v22(j0, k0)
        # First column of C (k0)
        v_idx_0 = v_full[:2]
        w_idx_0 = [k0*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx_0, w_idx_0)
        # Second column of C (k0+1)
        v_idx_1 = v_full[2:]
        w_idx_1 = [(k0+1)*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx_1, w_idx_1)

    # Helper to add a 1x2 * 2x1 block (rank 2)
    def add_block_1x2x1(i, j0, k0):
        nonlocal col
        u_idx = u12(i, j0)
        v_idx = v21(j0, k0)
        w_idx = [k0*3+i]
        add_block(Ur2, Vr2, Wr2, u_idx, v_idx, w_idx)

    # --- Part 3: C[2, 0:4] ---
    # C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2:4]*B[2:4,0:2]
    add_block_1x2x2(2, 0, 0)
    add_block_1x2x2(2, 2, 0)
    # C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2:4]*B[2:4,2:4]
    add_block_1x2x2(2, 0, 2)
    add_block_1x2x2(2, 2, 2)

    # --- Part 4: C[2, 4:7] ---
    # C[2, 4:6] = A[2,0:2]*B[0:2,4:6] + A[2,2:4]*B[2:4,4:6]
    add_block_1x2x2(2, 0, 4)
    add_block_1x2x2(2, 2, 4)
    # C[2, 6:7] = A[2,0:2]*B[0:2,6:7] + A[2,2:4]*B[2:4,6:7]
    add_block_1x2x1(2, 0, 6)
    add_block_1x2x1(2, 2, 6)

    # Total col should be 4*7 + 2*7 + 2*3 + 4*2 + 2*2 + 2*2 + 4 = 28+14+6+8+4+4+4 = 68? Let me recount.
    # Part 1: 4*7 = 28
    # Part 2: 2*7 + 2*3 = 14+6 = 20
    # Part 3: 4*2 = 8
    # Part 4: 2*2 + 4*1 = 4+4 = 8
    # Total: 28+20+8+8 = 64

    # Verify
    if col <= total_rank:
        tensor = _build_tensor(n, m, p)
        U_eff = U[:, :col]
        V_eff = V[:, :col]
        W_eff = W[:, :col]
        if _verify(U_eff, V_eff, W_eff, tensor):
            if col < best_rank:
                best_rank = col
                best_U, best_V, best_W = _to_lists(U_eff, V_eff, W_eff)

    if time.time() - start_time > TIME_LIMIT - 5:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
