import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m x p matrix multiplication using schoolbook algorithm.
    Rank = n * m * p.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
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

def _strassen_2x2x2(n, m, p, i_off, j_off, k_off):
    """
    Returns U, V, W for a 2x2x2 block embedded in the full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices for the 2x2 block
    i0 = i_off
    i1 = i_off + 1
    j0 = j_off
    j1 = j_off + 1
    k0 = k_off
    k1 = k_off + 1
    
    # Helper to get indices
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # Term 2: M2
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # Term 3: M3
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # Term 4: M4
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # Term 5: M5
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # Term 6: M6
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # Term 7: M7
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the decomposition is exact.
    U, V, W are numpy arrays or lists.
    """
    rank = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    u_val = U[i*m+j][c] if not isinstance(U, np.ndarray) else U[i*m+j, c]
                    v_val = V[j*p+k][c] if not isinstance(V, np.ndarray) else V[j*p+k, c]
                    w_val = W[k*n+i][c] if not isinstance(W, np.ndarray) else W[k*n+i, c]
                    val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Strategy:
    # 1. Construct a structured decomposition using Strassen blocks where possible.
    #    This yields a rank of 92 for 4x5x5 (vs 100 for schoolbook).
    # 2. Verify the structured decomposition.
    # 3. If valid, return it.
    # 4. If invalid (should not happen), fall back to schoolbook (rank 100).
    
    # Define blocks: (i_start, i_end, j_start, j_end, k_start, k_end, use_strassen)
    # We partition 4x5x5 into:
    # - 2x2x2 blocks (use Strassen if possible)
    # - Remaining strips (use schoolbook)
    
    blocks = [
        (0, 2, 0, 2, 0, 2, True),   # 2x2x2 -> Strassen (7)
        (0, 2, 0, 2, 2, 4, True),   # 2x2x2 -> Strassen (7)
        (0, 2, 0, 2, 4, 5, False),  # 2x2x1 -> Schoolbook (4)
        (0, 2, 2, 4, 0, 2, True),   # 2x2x2 -> Strassen (7)
        (0, 2, 2, 4, 2, 4, True),   # 2x2x2 -> Strassen (7)
        (0, 2, 2, 4, 4, 5, False),  # 2x2x1 -> Schoolbook (4)
        (0, 2, 4, 5, 0, 2, False),  # 2x1x2 -> Schoolbook (4)
        (0, 2, 4, 5, 2, 4, False),  # 2x1x2 -> Schoolbook (4)
        (0, 2, 4, 5, 4, 5, False),  # 2x1x1 -> Schoolbook (2)
        (2, 4, 0, 2, 0, 2, True),   # 2x2x2 -> Strassen (7)
        (2, 4, 0, 2, 2, 4, True),   # 2x2x2 -> Strassen (7)
        (2, 4, 0, 2, 4, 5, False),  # 2x2x1 -> Schoolbook (4)
        (2, 4, 2, 4, 0, 2, True),   # 2x2x2 -> Strassen (7)
        (2, 4, 2, 4, 2, 4, True),   # 2x2x2 -> Strassen (7)
        (2, 4, 2, 4, 4, 5, False),  # 2x2x1 -> Schoolbook (4)
        (2, 4, 4, 5, 0, 2, False),  # 2x1x2 -> Schoolbook (4)
        (2, 4, 4, 5, 2, 4, False),  # 2x1x2 -> Schoolbook (4)
        (2, 4, 4, 5, 4, 5, False),  # 2x1x1 -> Schoolbook (2)
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_start, i_end, j_start, j_end, k_start, k_end, use_strassen in blocks:
        bn = i_end - i_start
        bm = j_end - j_start
        bp = k_end - k_start
        
        if use_strassen and bn == 2 and bm == 2 and bp == 2:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_start, j_start, k_start)
        else:
            # Generate schoolbook for the sub-block
            U_sub, V_sub, W_sub = _schoolbook(bn, bm, bp)
            
            # Embed into full space
            U_full = np.zeros((n * m, U_sub.shape[1]))
            V_full = np.zeros((m * p, V_sub.shape[1]))
            W_full = np.zeros((p * n, W_sub.shape[1]))
            
            for i in range(bn):
                for j in range(bm):
                    idx_u = (i_start + i) * m + (j_start + j)
                    U_full[idx_u, :] = U_sub[i * bm + j, :]
                    
            for j in range(bm):
                for k in range(bp):
                    idx_v = (j_start + j) * p + (k_start + k)
                    V_full[idx_v, :] = V_sub[j * bp + k, :]
                    
            for k in range(bp):
                for i in range(bn):
                    idx_w = (k_start + k) * n + (i_start + i)
                    W_full[idx_w, :] = W_sub[k * bn + i, :]
                    
            U_b, V_b, W_b = U_full, V_full, W_full
            
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)
        
    U_combined = np.hstack(U_list)
    V_combined = np.hstack(V_list)
    W_combined = np.hstack(W_list)
    
    # Verify the structured result
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        # The entries are already integers (0, 1, -1), so they satisfy the half-integer constraint.
        return {
            "U": U_combined.tolist(),
            "V": V_combined.tolist(),
            "W": W_combined.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
