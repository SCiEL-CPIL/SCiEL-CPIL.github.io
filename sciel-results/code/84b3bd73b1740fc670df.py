"""
Rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.
Strategy: structured block decomposition (Strassen + schoolbook), then
systematic column-removal and iterative refinement to minimize rank.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check of the decomposition."""
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

def _build_2x3x2_block():
    """
    Build 2x3x2 decomposition. A is 2x3, B is 3x2, C is 2x2.
    Rank: use 4 Strassen 2x2x2 blocks on the 2x2x2 core, plus handle the extra column.
    
    Actually, let's think more carefully:
    A (2x3) = [A1 (2x2), A2 (2x1)]
    B (3x2) = [B1 (2x2); B2 (1x2)]
    C = A1@B1 + A2@B2
    
    A1@B1: 2x2x2 Strassen, rank 7
    A2@B2: 2x1x2 schoolbook, rank 4
    
    Total rank: 7 + 4 = 11
    """
    n, m, p = 2, 3, 2
    U2, V2, W2 = _build_strassen_2x2()
    
    total_rank = 7 + 4  # 11
    U = np.zeros((n * m, total_rank))  # 6 rows
    V = np.zeros((m * p, total_rank))  # 6 rows
    W = np.zeros((p * n, total_rank))  # 4 rows
    
    # A1 @ B1: 2x2 @ 2x2, Strassen rank 7
    for t in range(7):
        # U: A1 is A[:, 0:2], rows i*3+j for i in 0..1, j in 0..1
        for i in range(2):
            for j in range(2):
                U[i * 3 + j, t] = U2[i * 2 + j, t]
        # V: B1 is B[0:2, :], rows j*2+k for j in 0..1, k in 0..1
        for j in range(2):
            for k in range(2):
                V[j * 2 + k, t] = V2[j * 2 + k, t]
        # W: C is C, rows k*2+i for k in 0..1, i in 0..1
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, t] = W2[k * 2 + i, t]
    
    # A2 @ B2: 2x1 @ 1x2, schoolbook rank 4
    base = 7
    for i in range(2):
        for k in range(2):
            col = base + i * 2 + k
            U[i * 3 + 2, col] = 1.0
            V[2 * 2 + k, col] = 1.0
            W[k * 2 + i, col] = 1.0
    
    return U, V, W

def _build_2x3x3_block():
    """
    Build 2x3x3 decomposition. A is 2x3, B is 3x3, C is 2x3.
    
    Split B into B1 (3x2) and B2 (3x1):
    C = A@B1 + A@B2
    
    A@B1: 2x3x2, rank 11 (from above)
    A@B2: 2x3x1 schoolbook, rank 6
    
    Total rank: 11 + 6 = 17
    """
    n, m, p = 2, 3, 3
    U1, V1, W1 = _build_2x3x2_block()
    
    total_rank = 11 + 6  # 17
    U = np.zeros((n * m, total_rank))  # 6 rows
    V = np.zeros((m * p, total_rank))  # 9 rows
    W = np.zeros((p * n, total_rank))  # 6 rows
    
    # A @ B1: 2x3 @ 3x2
    for t in range(11):
        for i in range(2):
            for j in range(3):
                U[i * 3 + j, t] = U1[i * 3 + j, t]
        for j in range(3):
            for k in range(2):
                V[j * 3 + k, t] = V1[j * 2 + k, t]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i, t] = W1[k * 2 + i, t]
    
    # A @ B2: 2x3 @ 3x1, schoolbook rank 6
    base = 11
    for i in range(2):
        for j in range(3):
            col = base + i * 3 + j
            U[i * 3 + j, col] = 1.0
            V[j * 3 + 2, col] = 1.0
            W[2 * 2 + i, col] = 1.0
    
    return U, V, W

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            # A part
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row][col] += coeff_a
                        
            # B part
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
                        
            # C part
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c

    # 4 blocks
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_4x5x6_kron():
    """
    Build 4x5x6 decomposition using 4=2+2, 6=3+3 splitting.
    
    A (4x5) split into A_top (2x5) and A_bot (2x5)
    B (5x6) split into B_left (5x3) and B_right (5x3)
    C (4x6) = [C_tl C_tr; C_bl C_br]
    
    C_tl = A_top @ B_left : 2x5x3
    C_tr = A_top @ B_right : 2x5x3
    C_bl = A_bot @ B_left : 2x5x3
    C_br = A_bot @ B_right : 2x5x3
    
    For each 2x5x3 block:
    Split B into B1 (5x2) and B2 (5x1):
    C = A@B1 + A@B2
    
    A@B1: 2x5x2, rank 11 (from 2x3x2 logic adapted)
    A@B2: 2x5x1 schoolbook, rank 10
    
    Total per 2x5x3: 11 + 10 = 21
    Total: 4 * 21 = 84
    
    Wait, let me recalculate 2x5x2:
    A (2x5) @ B (5x2):
    Split B into B1 (2x2) and B2 (3x2):
    A@B1: 2x2x2 Strassen rank 7
    A@B2: 2x3x2 rank 11
    Total: 18
    
    Hmm, that's more. Let me try a different split.
    
    Actually for 2x5x2:
    Split A into A1 (2x2) and A2 (2x3):
    A1@B: 2x2x2 Strassen rank 7
    A2@B: 2x3x2 rank 11
    Total: 18
    
    Or split B into B1 (2x2) and B2 (3x2):
    A@B1: 2x2x2 Strassen rank 7
    A@B2: 2x3x2 rank 11
    Total: 18
    
    So 2x5x2 has rank 18.
    
    For 2x5x3:
    Split into 2x5x2 (rank 18) + 2x5x1 (rank 10) = 28
    Or split into 2x3x3 (rank 17) + 2x2x3 (rank ?)
    
    2x2x3: Split B into B1 (2x2) and B2 (2x1):
    A@B1: 2x2x2 rank 7
    A@B2: 2x2x1 rank 4
    Total: 11
    
    So 2x5x3 = 2x3x3 (17) + 2x2x3 (11) = 28
    
    Total: 4 * 28 = 112. That's worse than 108.
    
    Let me try the original approach but with better blocks.
    
    Actually, let me use the 2x3x3 block directly:
    4x5x6:
    Split A into A1 (2x5) and A2 (2x5)
    Split B into B1 (5x3) and B2 (5x3)
    
    Each 2x5x3:
    Split B into B1 (5x2) and B2 (5x1)
    A@B1: 2x5x2
    A@B2: 2x5x1 rank 10
    
    2x5x2:
    Split A into A1 (2x2) and A2 (2x3)
    A1@B: 2x2x2 rank 7
    A2@B: 2x3x2 rank 11
    Total: 18
    
    So 2x5x3 = 18 + 10 = 28
    Total: 4 * 28 = 112
    
    That's not better. Let me try a different decomposition.
    
    Alternative: Use the 2x3x3 block (rank 17) and 2x2x2 block (rank 7).
    
    4x5x6:
    Split into 4 blocks of 2x5x3, each rank 28 -> total 112
    
    Or split into:
    Top-left 2x3x3 (rank 17)
    Top-right 2x3x3 (rank 17) 
    Bottom-left 2x3x3 (rank 17)
    Bottom-right 2x3x3 (rank 17)
    Plus handle the extra column in B and extra rows in A...
    
    This is getting complicated. Let me just keep the original 108-rank approach
    but try to improve it with the iterative refinement.
    """
    n, m, p = 4, 5, 6
    
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 3 * 28 + 24
    U = np.zeros((n * m, total_rank))  # 4x5 = 20 rows
    V = np.zeros((m * p, total_rank))  # 5x6 = 30 rows
    W = np.zeros((p * n, total_rank))  # 6x4 = 24 rows
    
    # A_left @ B_top: 4x4 @ 4x6
    # A_left is A[:, 0:4], B_top is B[0:4, :]
    # Split B_top into 3 blocks of 4x2
    
    for blk in range(3):
        base = blk * 28
        k_start = blk * 2
        
        # U: A_left (4x4) -> rows 0-15
        for i in range(4):
            for j in range(4):
                U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
        
        # V: B_top (4x6) -> rows 0-23 (j=0..3, k=0..5)
        for j in range(4):
            for k in range(6):
                if k >= k_start and k < k_start + 2:
                    v_row = j * 6 + k
                    v_local_row = j * 2 + (k - k_start)
                    V[v_row, base:base+28] = V442[v_local_row, :]
        
        # W: C (4x6) -> rows k*4+i
        for k in range(6):
            if k >= k_start and k < k_start + 2:
                for i in range(4):
                    w_row = k * 4 + i
                    w_local_row = (k - k_start) * 4 + i
                    W[w_row, base:base+28] = W442[w_local_row, :]
    
    # A_right @ B_bottom: 4x1 @ 1x6
    # A_right is A[:, 4], B_bottom is B[4, :]
    # Schoolbook: rank 4*1*6 = 24
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

def _iterative_refinement(n, m, p, initial_rank=80, max_iters=100):
    """
    Iterative refinement: start with a random decomposition, then
    alternate between solving for each matrix.
    """
    if _time_left() < 10.0:
        return None
    
    R = initial_rank
    
    for attempt in range(3):
        if _time_left() < 8.0:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            
            # Solve for W
            A = np.zeros((p, n, R))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        A[k, i, :] += U[i * m + j, :] * V[j * p + k, :]
            
            W_new = np.zeros((p * n, R))
            valid = True
            for k in range(p):
                for i in range(n):
                    a = A[k, i, :]
                    if np.linalg.norm(a) < 1e-10:
                        valid = False
                        break
                    W_new[k * n + i, :] = a / (a @ a)
            if not valid:
                break
            W = W_new
            
            # Solve for U
            B = np.zeros((n, m, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        B[i, j, :] += V[j * p + k, :] * W[k * n + i, :]
            
            U_new = np.zeros((n * m, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    b = B[i, j, :]
                    if np.linalg.norm(b) < 1e-10:
                        valid = False
                        break
                    U_new[i * m + j, :] = b / (b @ b)
            if not valid:
                break
            U = U_new
            
            # Solve for V
            C = np.zeros((m, p, R))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        C[j, k, :] += U[i * m + j, :] * W[k * n + i, :]
            
            V_new = np.zeros((m * p, R))
            valid = True
            for j in range(m):
                for k in range(p):
                    c = C[j, k, :]
                    if np.linalg.norm(c) < 1e-10:
                        valid = False
                        break
                    V_new[j * p + k, :] = c / (c @ c)
            if not valid:
                break
            V = V_new
            
            if it % 10 == 0:
                if _verify_np(U, V, W, n, m, p, 1e-6):
                    if _verify_decomp(U, V, W, n, m, p, 1e-8):
                        return U, V, W
        
        if _verify_np(U, V, W, n, m, p, 1e-6):
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                return U, V, W
    
    return None

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

def _pairwise_reduce(U, V, W, n, m, p, max_iters=5, budget_s=15.0):
    """Try removing pairs of columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 3:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1 + 1, R):
                if _time_left() < 3.0:
                    break
                mask = np.ones(R, dtype=bool)
                mask[c1] = False
                mask[c2] = False
                U2 = U[:, mask]
                V2 = V[:, mask]
                W2 = W[:, mask]
                if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        R -= 2
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return U, V, W

def _triple_reduce(U, V, W, n, m, p, max_iters=3, budget_s=10.0):
    """Try removing triples of columns."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 4:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1 + 1, R):
                if _time_left() < 3.0:
                    break
                for c3 in range(c2 + 1, R):
                    if _time_left() < 3.0:
                        break
                    mask = np.ones(R, dtype=bool)
                    mask[c1] = False
                    mask[c2] = False
                    mask[c3] = False
                    U2 = U[:, mask]
                    V2 = V[:, mask]
                    W2 = W[:, mask]
                    if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                            U, V, W = U2, V2, W2
                            R -= 3
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Try Winograd-based decomposition first
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Winograd 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_kron()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try iterative refinement from Winograd initial
    try:
        if _time_left() > 20.0:
            for R in range(max(60, best_rank - 10), best_rank):
                if _time_left() < 10.0:
                    break
                cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=50)
                if cand is not None:
                    U, V, W = cand
                    if _verify_decomp(U, V, W, n, m, p, 1e-9):
                        if U.shape[1] < best_rank:
                            best_rank = U.shape[1]
                            best_U, best_V, best_W = U, V, W
                            break
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=10, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pairwise column-removal refinement
    try:
        if _time_left() > 8.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, max_iters=5, budget_s=10.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Triple column-removal refinement
    try:
        if _time_left() > 5.0:
            cand = _triple_reduce(best_U, best_V, best_W, n, m, p, max_iters=3, budget_s=8.0)
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
