import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2(n, m, p, i_off, j_off, k_off):
    """
    Returns U, V, W for a 2x2x2 block embedded in the full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i0 = i_off
    i1 = i_off + 1
    j0 = j_off
    j1 = j_off + 1
    k0 = k_off
    k1 = k_off + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Returns U, V, W for n x m x p schoolbook."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _embed_block(U_b, V_b, W_b, n, m, p, i_start, j_start, k_start, bn, bm, bp):
    """Embeds a local block decomposition into the full space."""
    rank_b = U_b.shape[1]
    U_full = np.zeros((n * m, rank_b))
    V_full = np.zeros((m * p, rank_b))
    W_full = np.zeros((p * n, rank_b))
    
    for i in range(bn):
        for j in range(bm):
            idx_u = (i_start + i) * m + (j_start + j)
            U_full[idx_u, :] = U_b[i * bm + j, :]
            
    for j in range(bm):
        for k in range(bp):
            idx_v = (j_start + j) * p + (k_start + k)
            V_full[idx_v, :] = V_b[j * bp + k, :]
            
    for k in range(bp):
        for i in range(bn):
            idx_w = (k_start + k) * n + (i_start + i)
            W_full[idx_w, :] = W_b[k * bn + i, :]
            
    return U_full, V_full, W_full

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = 150.0
    
    # Partition:
    # n=4: [2, 2]
    # m=5: [2, 3]
    # p=5: [2, 3]
    
    # Blocks:
    # (0:2, 0:2, 0:2): 2x2x2 Strassen (rank 7)
    # (0:2, 0:2, 2:5): 2x2x3 Schoolbook (rank 12)
    # (0:2, 2:5, 0:2): 2x3x2 Schoolbook (rank 12)
    # (0:2, 2:5, 2:5): 2x3x3 Schoolbook (rank 18)
    # (2:4, 0:2, 0:2): 2x2x2 Strassen (rank 7)
    # (2:4, 0:2, 2:5): 2x2x3 Schoolbook (rank 12)
    # (2:4, 2:5, 0:2): 2x3x2 Schoolbook (rank 12)
    # (2:4, 2:5, 2:5): 2x3x3 Schoolbook (rank 18)
    
    # Total rank: 7 + 12 + 12 + 18 + 7 + 12 + 12 + 18 = 98
    
    # Wait, the [2,2,1]x[2,2,1] partition yielded 92.
    # Let's re-evaluate [2,3]x[2,3] vs [2,2,1]x[2,2,1].
    # [2,2,1] partition:
    # n=[2,2], m=[2,2,1], p=[2,2,1]
    # Strassen blocks (2x2x2):
    # (0:2, 0:2, 0:2)
    # (0:2, 0:2, 2:4)
    # (0:2, 2:4, 0:2)
    # (0:2, 2:4, 2:4)
    # (2:4, 0:2, 0:2)
    # (2:4, 0:2, 2:4)
    # (2:4, 2:4, 0:2)
    # (2:4, 2:4, 2:4)
    # 8 blocks * 7 = 56
    
    # Schoolbook blocks:
    # (0:2, 0:2, 4:5): 2x2x1 (rank 4)
    # (0:2, 2:4, 4:5): 2x2x1 (rank 4)
    # (0:2, 4:5, 0:2): 2x1x2 (rank 4)
    # (0:2, 4:5, 2:4): 2x1x2 (rank 4)
    # (0:2, 4:5, 4:5): 2x1x1 (rank 2)
    # (2:4, 0:2, 4:5): 2x2x1 (rank 4)
    # (2:4, 2:4, 4:5): 2x2x1 (rank 4)
    # (2:4, 4:5, 0:2): 2x1x2 (rank 4)
    # (2:4, 4:5, 2:4): 2x1x2 (rank 4)
    # (2:4, 4:5, 4:5): 2x1x1 (rank 2)
    # Total schoolbook: 4+4+4+4+2+4+4+4+4+2 = 36
    # Total rank: 56 + 36 = 92
    
    # So [2,2,1] is better. Let's stick to the [2,2,1] partition which the current program uses.
    # The current program already implements this and scores 1.087.
    # Is there a better partition?
    # What about n=[4], m=[5], p=[5]? No Strassen.
    # What about n=[2,2], m=[5], p=[5]?
    #   (0:2, 0:5, 0:5): 2x5x5 schoolbook rank 50
    #   (2:4, 0:5, 0:5): 2x5x5 schoolbook rank 50
    #   Total 100.
    
    # The 92 rank is quite good.
    
    # Let's re-verify the 92 rank implementation.
    
    U_list = []
    V_list = []
    W_list = []
    
    blocks = [
        (0, 2, 0, 2, 0, 2, True),
        (0, 2, 0, 2, 2, 4, True),
        (0, 2, 0, 2, 4, 5, False),
        (0, 2, 2, 4, 0, 2, True),
        (0, 2, 2, 4, 2, 4, True),
        (0, 2, 2, 4, 4, 5, False),
        (0, 2, 4, 5, 0, 2, False),
        (0, 2, 4, 5, 2, 4, False),
        (0, 2, 4, 5, 4, 5, False),
        (2, 4, 0, 2, 0, 2, True),
        (2, 4, 0, 2, 2, 4, True),
        (2, 4, 0, 2, 4, 5, False),
        (2, 4, 2, 4, 0, 2, True),
        (2, 4, 2, 4, 2, 4, True),
        (2, 4, 2, 4, 4, 5, False),
        (2, 4, 4, 5, 0, 2, False),
        (2, 4, 4, 5, 2, 4, False),
        (2, 4, 4, 5, 4, 5, False),
    ]
    
    for i_start, i_end, j_start, j_end, k_start, k_end, use_strassen in blocks:
        bn = i_end - i_start
        bm = j_end - j_start
        bp = k_end - k_start
        
        if use_strassen and bn == 2 and bm == 2 and bp == 2:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_start, j_start, k_start)
            U_list.append(U_b)
            V_list.append(V_b)
            W_list.append(W_b)
        else:
            U_b, V_b, W_b = _schoolbook_block(bn, bm, bp)
            U_full, V_full, W_full = _embed_block(U_b, V_b, W_b, n, m, p, i_start, j_start, k_start, bn, bm, bp)
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
            
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    U_check = U_combined.tolist()
    V_check = V_combined.tolist()
    W_check = W_combined.tolist()
    
    if _verify(U_check, V_check, W_check, n, m, p):
        return {"U": U_check, "V": V_check, "W": W_check}
    else:
        # Fallback
        U, V, W = _schoolbook_block(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
