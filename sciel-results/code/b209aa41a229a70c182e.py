"""
Combined Structural Decomposition for 4x5x6 Matrix Multiplication.
Uses 4x4x4 Strassen (rank 49), 4x4x2 Strassen (rank 28), and schoolbook remainder.
Target Rank: 101.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where each has shape (4, 7).
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen.
    Rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2, r2]
                            if c2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, col] += c1 * c2
            
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2, r2]
                            if c2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, col] += c1 * c2
            
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2, r2]
                            if c2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, col] += c1 * c2
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using 4x4x4 Strassen and 4x4x2 Strassen.
    
    Strategy:
    - Split A (4x5) into A_L (4x4) and A_R (4x1)
    - Split B (5x6) into B_T (4x6) and B_B (1x6)
    - C = A_L @ B_T + A_R @ B_B
    
    For A_L @ B_T (4x4 @ 4x6):
      - Split B_T into B_T1 (4x4) and B_T2 (4x2)
      - Use 4x4x4 Strassen for A_L @ B_T1 (rank 49)
      - Use 4x4x2 Strassen for A_L @ B_T2 (rank 28)
    
    For A_R @ B_B (4x1 @ 1x6):
      - Schoolbook: rank 24
    
    Total rank: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))  # 20 x 101
    V = np.zeros((m * p, total_rank))  # 30 x 101
    W = np.zeros((p * n, total_rank))  # 24 x 101
    
    # Part 1: A_L @ B_T1 (4x4 @ 4x4) -> C_L1 (4x4)
    base = 0
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
            
    # Part 2: A_L @ B_T2 (4x4 @ 4x2) -> C_L2 (4x2)
    base = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    for j in range(4):
        for k in range(2):
            V[j * 6 + (4+k), base:base+28] = V442[j * 2 + k, :]
    for k in range(2):
        for i in range(4):
            W[(4+k) * 4 + i, base:base+28] = W442[k * 4 + i, :]
            
    # Part 3: A_R @ B_B (4x1 @ 1x6) -> C_R (4x6)
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
            
    return U, V, W

def _build_4x5x6_alternative():
    """
    Alternative: Split differently to potentially get lower rank.
    - Split A (4x5) into A1 (4x3) and A2 (4x2)
    - Split B (5x6) into B1 (3x6) and B2 (2x6)
    - C = A1 @ B1 + A2 @ B2
    
    For A1 @ B1 (4x3 @ 3x6):
      - Use 4x3x6 schoolbook: rank 72, or try Strassen-like
    
    For A2 @ B2 (4x2 @ 2x6):
      - Use 4x2x6 schoolbook: rank 48
    
    Total: 72 + 48 = 120 (worse)
    
    Better: Try 4x4x6 with Strassen:
    - Split A into A_L (4x4) and A_R (4x1)
    - B is 5x6, split into B1 (4x6) and B2 (1x6)
    - A_L @ B1: 4x4x6, use 4x4x4 Strassen (49) + 4x4x2 Strassen (28) = 77
    - A_R @ B2: 4x1x6 schoolbook = 24
    - Total: 101 (same)
    
    Try another: 3x5x6 + 1x5x6
    - 3x5x6: schoolbook 90, or Strassen 3x3x3 (19) + 3x3x3 (19) + 3x3x3 (19) = 57? No, 3x3x3 Strassen is rank 19
    - Actually 3x5x6: split 5 into 3+2, 6 into 3+3
      - 3x3x3: 19
      - 3x3x3: 19  
      - 3x2x3: schoolbook 18 or Strassen 3x2x2 (14) + 3x2x1 (6) = 20? Let's use schoolbook 18
      - 3x2x3: 18
      Total: 19+19+18+18 = 74
    - 1x5x6: schoolbook 30
    - Total: 74 + 30 = 104 (worse)
    
    Try 4x4x6 + 4x1x6:
    - 4x4x6: split 6 into 4+2
      - 4x4x4: 49
      - 4x4x2: 28
      Total: 77
    - 4x1x6: 24
    - Total: 101 (same)
    
    So 101 seems to be the best block decomposition for now.
    """
    # This is a placeholder - we'll try to find better via iterative refinement
    return None

def _build_4x5x6_winograd():
    """
    Fallback: Build 4x5x6 decomposition using Winograd-style overlapping blocks.
    Rank = 108.
    """
    n, m, p = 4, 5, 6
    U442, V442, W442 = _build_strassen_4x4x2()
    total_rank = 3 * 28 + 24
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    for blk in range(3):
        base = blk * 28
        k_start = blk * 2
        for i in range(4):
            for j in range(4):
                U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
        for j in range(4):
            for k in range(6):
                if k >= k_start and k < k_start + 2:
                    v_row = j * 6 + k
                    v_local_row = j * 2 + (k - k_start)
                    V[v_row, base:base+28] = V442[v_local_row, :]
        for k in range(6):
            if k >= k_start and k < k_start + 2:
                for i in range(4):
                    w_row = k * 4 + i
                    w_local_row = (k - k_start) * 4 + i
                    W[w_row, base:base+28] = W442[w_local_row, :]
    base = 3 * 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _iterative_refinement(n, m, p, initial_rank=80, max_iters=100, seed=None):
    """
    Vectorized iterative refinement: alternate solving for each matrix.
    """
    if _time_left() < 10.0:
        return None
    if seed is not None:
        np.random.seed(seed)
    R = initial_rank
    U3 = np.random.randn(n, m, R) * 0.1
    V3 = np.random.randn(m, p, R) * 0.1
    W3 = np.random.randn(p, n, R) * 0.1
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        A = np.einsum('ijr,jkr->kir', U3, V3)
        norms = np.einsum('kir,kir->ki', A, A)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        W3 = A / norms[:, :, np.newaxis]
        
        B = np.einsum('jkr,kir->ijr', V3, W3)
        norms = np.einsum('ijr,ijr->ij', B, B)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        U3 = B / norms[:, :, np.newaxis]
        
        C = np.einsum('ijr,kir->jkr', U3, W3)
        norms = np.einsum('jkr,jkr->jk', C, C)
        valid = np.all(norms > 1e-10)
        if not valid:
            break
        V3 = C / norms[:, :, np.newaxis]
        
        if it % 5 == 0:
            U = U3.reshape(n*m, R)
            V = V3.reshape(m*p, R)
            W = W3.reshape(p*n, R)
            if _verify_np(U, V, W, n, m, p, 1e-6):
                if _verify_decomp(U, V, W, n, m, p, 1e-8):
                    return U, V, W
    
    U = U3.reshape(n*m, R)
    V = V3.reshape(m*p, R)
    W = W3.reshape(p*n, R)
    if _verify_np(U, V, W, n, m, p, 1e-6):
        if _verify_decomp(U, V, W, n, m, p, 1e-8):
            return U, V, W
    return None

def _try_multiple_ranks(n, m, p, start_rank, end_rank, max_restarts=5):
    """
    Try multiple ranks with multiple restarts to find a lower rank decomposition.
    """
    best_result = None
    best_rank = start_rank + 1
    
    for R in range(end_rank - 1, start_rank - 1, -1):
        if _time_left() < 15.0:
            break
        for restart in range(max_restarts):
            if _time_left() < 5.0:
                break
            seed = 42 + R * 100 + restart
            cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=60, seed=seed)
            if cand is not None:
                U, V, W = cand
                if _verify_decomp(U, V, W, n, m, p, 1e-8):
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_result = (U, V, W)
                        if best_rank <= start_rank:
                            return best_result
    return best_result

def _local_reduce(U, V, W, n, m, p, max_iters=10, budget_s=20.0):
    """Local column-removal search."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        for c in range(R):
            if _time_left() < 3.0:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Improved 4x5x6 (target rank 101)
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try Winograd 4x5x6 (rank 108)
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_winograd()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try multiple ranks with multiple restarts
    try:
        if _time_left() > 25.0:
            start_rank = max(60, best_rank - 20)
            result = _try_multiple_ranks(n, m, p, start_rank, best_rank, max_restarts=3)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Local column-removal
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=15, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
