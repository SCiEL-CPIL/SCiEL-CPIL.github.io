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

def _build_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen.
    Rank = 7 * 7 = 49.
    U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U part: U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
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
            
            # V part: V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
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
            
            # W part: W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
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

def _build_2x2x3():
    """2x2 @ 2x3, rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)."""
    U2, V2, W2 = _build_strassen_2x2()
    rank = 11
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Part 1: 2x2x2 Strassen for B cols 0,1
    for i in range(2):
        for j in range(2):
            U[i*2+j, 0:7] = U2[i*2+j, :]
    for j in range(2):
        for k in range(2):
            V[j*3+k, 0:7] = V2[j*2+k, :]
    for k in range(2):
        for i in range(2):
            W[k*2+i, 0:7] = W2[k*2+i, :]
    
    # Part 2: 2x2x1 schoolbook for B col 2
    for i in range(2):
        for j in range(2):
            col = 7 + i*2 + j
            U[i*2+j, col] = 1.0
            V[j*3+2, col] = 1.0
            W[2*2+i, col] = 1.0
    
    return U, V, W

def _build_2x3x3():
    """2x3 @ 3x3, rank 17 (2x2x3 + schoolbook 2x1x3)."""
    rank = 17
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Part 1: 2x2x3 for A cols 0,1, B rows 0,1
    U1, V1, W1 = _build_2x2x3()
    for i in range(2):
        for j in range(2):
            U[i*3+j, 0:11] = U1[i*2+j, :]
    for j in range(2):
        for k in range(3):
            V[j*3+k, 0:11] = V1[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W[k*2+i, 0:11] = W1[k*2+i, :]
    
    # Part 2: 2x1x3 schoolbook for A col 2, B row 2
    for i in range(2):
        for k in range(3):
            col = 11 + i*3 + k
            U[i*3+2, col] = 1.0
            V[2*3+k, col] = 1.0
            W[k*2+i, col] = 1.0
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using 2x3x3 and 2x2x3 blocks.
    
    Split A (4x5) into rows {0,1} and {2,3}, cols {0,1,2} and {3,4}.
    Split B (5x6) into rows {0,1,2} and {3,4}, cols {0,1,2} and {3,4,5}.
    
    Subproblems:
    1. A[0:2,0:3] @ B[0:3,0:3] -> 2x3x3 (rank 17)
    2. A[0:2,3:5] @ B[3:5,0:3] -> 2x2x3 (rank 11)
    3. A[2:4,0:3] @ B[0:3,3:6] -> 2x3x3 (rank 17)
    4. A[2:4,3:5] @ B[3:5,3:6] -> 2x2x3 (rank 11)
    
    Total rank: 17 + 11 + 17 + 11 = 56
    """
    n, m, p = 4, 5, 6
    
    U233, V233, W233 = _build_2x3x3()
    U223, V223, W223 = _build_2x2x3()
    
    total_rank = 17 + 11 + 17 + 11
    U = np.zeros((n * m, total_rank))  # 4x5 = 20 rows
    V = np.zeros((m * p, total_rank))  # 5x6 = 30 rows
    W = np.zeros((p * n, total_rank))  # 6x4 = 24 rows
    
    # Block 1: A[0:2,0:3] @ B[0:3,0:3] -> C[0:2,0:3]
    base = 0
    for i in range(2):
        for j in range(3):
            U[i * 5 + j, base:base+17] = U233[i * 3 + j, :]
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, base:base+17] = V233[j * 3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k * 4 + i, base:base+17] = W233[k * 2 + i, :]
    
    # Block 2: A[0:2,3:5] @ B[3:5,0:3] -> C[0:2,0:3]
    base = 17
    for i in range(2):
        for j_local in range(2):
            j = 3 + j_local
            U[i * 5 + j, base:base+11] = U223[i * 2 + j_local, :]
    for j_local in range(2):
        j = 3 + j_local
        for k in range(3):
            V[j * 6 + k, base:base+11] = V223[j_local * 3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k * 4 + i, base:base+11] += W223[k * 2 + i, :]
    
    # Block 3: A[2:4,0:3] @ B[0:3,3:6] -> C[2:4,3:6]
    base = 17 + 11
    for i_local in range(2):
        i = 2 + i_local
        for j in range(3):
            U[i * 5 + j, base:base+17] = U233[i_local * 3 + j, :]
    for j in range(3):
        for k_local in range(3):
            k = 3 + k_local
            V[j * 6 + k, base:base+17] = V233[j * 3 + k_local, :]
    for k_local in range(3):
        k = 3 + k_local
        for i_local in range(2):
            i = 2 + i_local
            W[k * 4 + i, base:base+17] = W233[k_local * 2 + i_local, :]
    
    # Block 4: A[2:4,3:5] @ B[3:5,3:6] -> C[2:4,3:6]
    base = 17 + 11 + 17
    for i_local in range(2):
        i = 2 + i_local
        for j_local in range(2):
            j = 3 + j_local
            U[i * 5 + j, base:base+11] = U223[i_local * 2 + j_local, :]
    for j_local in range(2):
        j = 3 + j_local
        for k_local in range(3):
            k = 3 + k_local
            V[j * 6 + k, base:base+11] = V223[j_local * 3 + k_local, :]
    for k_local in range(3):
        k = 3 + k_local
        for i_local in range(2):
            i = 2 + i_local
            W[k * 4 + i, base:base+11] += W223[k_local * 2 + i_local, :]
    
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

def _local_reduce(U, V, W, n, m, p, max_iters=20, budget_s=60.0):
    """Local column-removal search with randomization and pairwise removal."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        # Randomly shuffle columns to try different orders
        cols = list(range(R))
        rng.shuffle(cols)
        
        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
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
        if not found and R >= 2:
            # Try pairwise removal
            found_pair = False
            for c1 in range(R):
                if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                    break
                for c2 in range(c1 + 1, R):
                    if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                        break
                    idx = [i for i in range(R) if i != c1 and i != c2]
                    U2 = U[:, idx]
                    V2 = V[:, idx]
                    W2 = W[:, idx]
                    if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                            U, V, W = U2, V2, W2
                            R -= 2
                            found_pair = True
                            break
                if found_pair:
                    break
            if not found_pair:
                break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Try Winograd-based decomposition first
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Improved 4x5x6
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 10.0:
            budget = min(100.0, _time_left() - 10.0)
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=50, budget_s=budget)
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
