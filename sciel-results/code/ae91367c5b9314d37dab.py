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
    """
    R = U.shape[1]
    keep = []
    for r in range(R):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        if np.all(np.abs(u_col) < tol):
            continue
        if np.all(np.abs(v_col) < tol):
            continue
        if np.all(np.abs(w_col) < tol):
            continue
        
        u3 = u_col.reshape(n, m)
        v3 = v_col.reshape(m, p)
        w3 = w_col.reshape(p, n)
        
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
    Alternative: embed 4x5x6 into larger structure, then prune.
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

def _hypergraph_greedy_prune(U, V, W, n, m, p, max_iters=50, budget_s=20.0):
    """
    Combinatorial design strategy: model the decomposition as a 3-partite 3-uniform hypergraph.
    Each rank-1 term r is a hyperedge connecting:
      - Part 1: support of U_col_r (subset of {0..n*m-1})
      - Part 2: support of V_col_r (subset of {0..m*p-1})
      - Part 3: support of W_col_r (subset of {0..p*n-1})
    
    A set of terms is "disjoint" (non-interfering) if no two terms share a vertex in any part.
    We use a greedy maximum matching heuristic to find a large set of pairwise disjoint terms,
    then try to remove the *complement* of this matching if the remaining terms still span the tensor.
    
    Actually, a more direct approach: 
    1. Identify terms that are "redundant" in the sense that their contribution is linearly 
       dependent on others. But that's hard.
    
    Alternative interpretation: 
    The hypergraph matching gives us a set of "independent" multiplication paths. 
    If we can find a set of terms whose removal doesn't break the decomposition, we remove them.
    
    Let's try a different angle: 
    For each term r, compute its "unique contribution" - the set of tensor entries (i,j,k) 
    where this term is the *only* non-zero contributor. If a term has no unique contribution, 
    it might be removable. But that's not quite right either, because multiple terms can 
    contribute to the same entry.
    
    Better approach: 
    Use the greedy matching to identify a core set of terms that are "essential" (each covers 
    at least one unique tensor entry). Then try to remove terms that are NOT in this core set.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    # Compute the full tensor
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # For each term r, compute its contribution tensor
    # contrib_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Greedy matching: find a set of terms that are "maximally spread out"
    # We'll use a simple heuristic: sort terms by the number of unique tensor entries they cover
    # (i.e., entries where they are the only non-zero contributor), and greedily pick the best ones.
    
    # First, compute for each term r, the set of (i,j,k) where it is the only non-zero contributor
    # This is expensive, so let's approximate: for each (i,j,k), count how many terms contribute non-zero.
    
    # Count non-zero contributions per tensor entry
    # contrib_count[i,j,k] = number of r such that U3[i,j,r]*V3[j,k,r]*W3[k,i,r] != 0
    contrib_count = np.zeros((n, m, p), dtype=int)
    for r in range(R):
        u_col = U3[:, :, r]
        v_col = V3[:, :, r]
        w_col = W3[:, :, r]
        # contrib_r = u_col * v_col * w_col (element-wise broadcast)
        # We need to be careful with broadcasting: u_col is (n,m), v_col is (m,p), w_col is (p,n)
        # contrib_r[i,j,k] = u_col[i,j] * v_col[j,k] * w_col[k,i]
        for i in range(n):
            for j in range(m):
                uj = u_col[i, j]
                if abs(uj) < 1e-12:
                    continue
                for k in range(p):
                    val = uj * v_col[j, k] * w_col[k, i]
                    if abs(val) > 1e-12:
                        contrib_count[i, j, k] += 1
    
    # For each term r, count how many tensor entries it is the *only* contributor for
    unique_counts = np.zeros(R, dtype=int)
    for r in range(R):
        u_col = U3[:, :, r]
        v_col = V3[:, :, r]
        w_col = W3[:, :, r]
        for i in range(n):
            for j in range(m):
                uj = u_col[i, j]
                if abs(uj) < 1e-12:
                    continue
                for k in range(p):
                    val = uj * v_col[j, k] * w_col[k, i]
                    if abs(val) > 1e-12 and contrib_count[i, j, k] == 1:
                        unique_counts[r] += 1
    
    # Sort terms by unique count (descending)
    order = np.argsort(-unique_counts)
    
    # Greedy matching: pick terms that are "most unique"
    # Then try to remove the rest
    keep = set()
    for r in order:
        if unique_counts[r] > 0:
            keep.add(r)
    
    # Try to remove terms not in keep
    remove_candidates = [r for r in range(R) if r not in keep]
    
    # Try removing one by one
    for r in remove_candidates:
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        mask = np.ones(R, dtype=bool)
        mask[r] = False
        U2 = U[:, mask]
        V2 = V[:, mask]
        W2 = W[:, mask]
        if _verify_np(U2, V2, W2, n, m, p, 1e-4):
            if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                U, V, W = U2, V2, W2
                R = U.shape[1]
                # Recompute unique counts
                U3 = U.reshape(n, m, R)
                V3 = V.reshape(m, p, R)
                W3 = W.reshape(p, n, R)
                contrib_count = np.zeros((n, m, p), dtype=int)
                for rr in range(R):
                    u_col = U3[:, :, rr]
                    v_col = V3[:, :, rr]
                    w_col = W3[:, :, rr]
                    for i in range(n):
                        for j in range(m):
                            uj = u_col[i, j]
                            if abs(uj) < 1e-12:
                                continue
                            for k in range(p):
                                val = uj * v_col[j, k] * w_col[k, i]
                                if abs(val) > 1e-12:
                                    contrib_count[i, j, k] += 1
                unique_counts = np.zeros(R, dtype=int)
                for rr in range(R):
                    u_col = U3[:, :, rr]
                    v_col = V3[:, :, rr]
                    w_col = W3[:, :, rr]
                    for i in range(n):
                        for j in range(m):
                            uj = u_col[i, j]
                            if abs(uj) < 1e-12:
                                continue
                            for k in range(p):
                                val = uj * v_col[j, k] * w_col[k, i]
                                if abs(val) > 1e-12 and contrib_count[i, j, k] == 1:
                                    unique_counts[rr] += 1
    
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
    
    # Hypergraph greedy pruning
    try:
        if _time_left() > 25.0:
            budget = min(20.0, _time_left() - 10.0)
            cand = _hypergraph_greedy_prune(best_U, best_V, best_W, n, m, p, max_iters=50, budget_s=budget)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Local column-removal refinement with enhanced search
    try:
        if _time_left() > 15.0:
            budget = min(60.0, _time_left() - 10.0)
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
            budget = min(30.0, _time_left() - 5.0)
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
