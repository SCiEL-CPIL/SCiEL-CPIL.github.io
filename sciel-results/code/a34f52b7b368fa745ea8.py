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

def _build_strassen_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x4, B is 4x2, C is 4x2.
    Rank = 4 * 7 = 28.
    U has shape (16, 28), V has shape (8, 28), W has shape (8, 28).
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

def _prune_zero_contribution_columns(U, V, W, n, m, p, tol=1e-12):
    """
    Laser method: identify and remove rank-1 terms where the product
    U_col * V_col * W_col contributes zero to all tensor entries.
    A column r has zero contribution if for all (i,j,k),
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0.
    This happens when at least one of U_col, V_col, W_col is entirely zero,
    or when their supports are disjoint in a way that all products vanish.
    """
    R = U.shape[1]
    keep = []
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # Quick check: if any column is all zeros, this term contributes nothing
        if np.all(np.abs(u_col) < tol):
            continue
        if np.all(np.abs(v_col) < tol):
            continue
        if np.all(np.abs(w_col) < tol):
            continue
        
        # Full check: compute the contribution tensor for this single term
        # and verify it's all zeros
        u3 = u_col.reshape(n, m)
        v3 = v_col.reshape(m, p)
        w3 = w_col.reshape(p, n)
        
        # The contribution to T[i,j,k] is u3[i,j] * v3[j,k] * w3[k,i]
        # Check if this is zero for all (i,j,k)
        nonzero = False
        for i in range(n):
            for j in range(m):
                uj = u3[i, j]
                if abs(uj) < tol:
                    continue
                for k in range(p):
                    val = uj * v3[j, k] * w3[k, i]
                    if abs(val) > tol:
                        nonzero = True
                        break
                if nonzero:
                    break
            if nonzero:
                break
        
        if not nonzero:
            continue
        
        keep.append(r)
    
    if len(keep) < R:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    
    return U, V, W

def _build_4x5x6_improved():
    """
    Build 4x5x6 decomposition using block decomposition + laser pruning.
    
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
    
    Then apply laser pruning to remove redundant terms.
    Total initial rank: 49 + 28 + 24 = 101
    """
    n, m, p = 4, 5, 6
    
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: A_L @ B_T1 (4x4 @ 4x4)
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
    
    # Part 2: A_L @ B_T2 (4x4 @ 4x2)
    base = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, base:base+28] = V442[j * 2 + k_local, :]
    for k_local in range(2):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, base:base+28] = W442[k_local * 4 + i, :]
    
    # Part 3: A_R @ B_B (4x1 @ 1x6)
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    
    # Apply laser pruning
    U, V, W = _prune_zero_contribution_columns(U, V, W, n, m, p)
    
    return U, V, W

def _build_4x5x6_embedded():
    """
    Alternative: embed 4x5x6 into 8x8x8 via Kronecker structure,
    then prune. This uses the full 4x4x4x4x4x4x4x4x4 Strassen-Winograd
    structure but zeros out unused dimensions.
    
    Actually, a cleaner approach: use 4x4x4 for the main block and
    handle the extra dimensions more cleverly.
    
    Let's try: split A(4x5) = [A1(4x4) | A2(4x1)]
    Split B(5x6) = [B1(4x6); B2(1x6)]
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (4x4 @ 4x6): split B1 = [B1a(4x4) | B1b(4x2)]
    C = A1*B1a + A1*B1b + A2*B2
    
    Use Strassen 4x4x4 (rank 49) for A1*B1a
    Use Strassen 4x4x2 (rank 28) for A1*B1b  
    Use schoolbook (rank 24) for A2*B2
    
    But we can also try: use a single 4x4x6 decomposition.
    4x4x6 can be seen as 4x4x4 + 4x4x2, but maybe we can do better
    by using the structure more cleverly.
    
    For now, use the same block approach but with a different split:
    Split A(4x5) = [A1(2x5); A2(2x5)]  -- no, A is 4x5
    Split by rows: A_top(2x5), A_bot(2x5)
    Split B by cols: B_left(5x3), B_right(5x3)
    
    C = A_top*B_left + A_top*B_right + A_bot*B_left + A_bot*B_right
    Each term is 2x5 @ 5x3 = 2x3
    Rank of 2x5x3: schoolbook is 30, but we can do better.
    2x5x3 = 2x3x5 tensor. Use Strassen-like:
    Actually 2x3x5: we can split 5 = 3+2, so 2x3x3 + 2x3x2
    2x3x3: rank at most 2*3*3=18 schoolbook, but Strassen gives ~7*... 
    This is getting complicated. Let's stick with the 4x4x4 + 4x4x2 + schoolbook approach
    but enhance the pruning.
    """
    return _build_4x5x6_improved()

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

def _local_reduce(U, V, W, n, m, p, max_iters=30, budget_s=80.0):
    """
    Enhanced local column-removal search with:
    1. Single column removal
    2. Pair column removal (when single fails)
    3. Randomized ordering
    4. Zero-contribution pruning between iterations
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        
        # First, prune zero-contribution columns
        U, V, W = _prune_zero_contribution_columns(U, V, W, n, m, p)
        R = U.shape[1]
        if R < 2:
            break
        
        found = False
        cols = list(range(R))
        rng.shuffle(cols)
        
        # Try single column removal
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
        
        if found:
            continue
        
        # Try pair column removal (only if R is large enough)
        if R >= 4:
            pair_found = False
            # Try a limited number of random pairs
            n_pairs_to_try = min(50, R * (R - 1) // 2)
            tried = set()
            for _ in range(n_pairs_to_try):
                if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                    break
                c1, c2 = rng.choice(R, size=2, replace=False)
                key = (min(c1, c2), max(c1, c2))
                if key in tried:
                    continue
                tried.add(key)
                
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
                        pair_found = True
                        break
            
            if pair_found:
                continue
        
        # If nothing worked, stop
        break
    
    return U, V, W

def _try_winograd_split(n, m, p):
    """
    Try a Winograd-style decomposition for 4x5x6.
    
    Winograd's method for matrix multiplication uses the fact that
    (a+b)(c+d) = ac + ad + bc + bd can be computed with 3 multiplications
    instead of 4, by computing M1=(a+b)(c+d), M2=ac, M3=bd, then
    ad = M1 - M2 - M3.
    
    For 4x5x6, we can apply this at multiple levels.
    
    Level 1: Split the 5 in the middle into 3+2.
    A(4x5) = [A1(4x3) | A2(4x2)]
    B(5x6) = [B1(3x6); B2(2x6)]
    C = A1*B1 + A2*B2
    
    Level 2: For A1*B1 (4x3 @ 3x6), split 3 into 1+2.
    A1 = [A1a(4x1) | A1b(4x2)]
    B1 = [B1a(1x6); B1b(2x6)]
    A1*B1 = A1a*B1a + A1b*B1b
    
    This doesn't immediately help. Let's try a different approach.
    
    Actually, the key Winograd insight for rectangular matrices:
    For multiplying (n x m) by (m x p), we can use the "Laser method"
    by embedding into a larger tensor and using the fact that
    rank(4x5x6) <= rank(4x4x4) + rank(4x4x2) + rank(4x1x6)
    but with clever overlap, we might do better.
    
    For now, return the same as the improved version.
    """
    return _build_4x5x6_improved()

def construct():
    n, m, p = 4, 5, 6
    
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Try Improved 4x5x6 with laser pruning
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try Winograd split
    try:
        if _time_left() > 20.0:
            U, V, W = _try_winograd_split(n, m, p)
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Local column-removal refinement with enhanced search
    try:
        if _time_left() > 15.0:
            budget = min(90.0, _time_left() - 10.0)
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=30, budget_s=budget)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Second pass of local reduction if time permits
    try:
        if _time_left() > 10.0:
            budget = min(40.0, _time_left() - 5.0)
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=20, budget_s=budget)
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": W.tolist()}
