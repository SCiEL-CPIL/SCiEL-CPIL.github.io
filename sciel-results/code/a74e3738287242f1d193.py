"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank_fast(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """Vectorized verification of tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank). True iff valid."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]

    # Diagonal entries T[i*m+j, j*p+k, k*n+i] == 1
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
    if not np.allclose(diag_vals, 1.0, atol=tol):
        return False

    # Off-diagonal: for each (i,j,k), sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k'*n+i',r] must be 0
    # unless (i',k') == (i,k). We check all (i',k') != (i,k).
    # Vectorize: for each j, build the constraint matrix.
    for j in range(m):
        # For fixed j, we have n*p (i,k) pairs and n*p (i',k') pairs
        # The constraint is: for each (i,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k'*n+i',r] = 0 for (i',k') != (i,k)
        # Let's form: for each (i,k), the vector prod_{i,k} = U[i*m+j,:] * V[j*p+k,:] (element-wise)
        # Then for each (i',k'), the dot product with W[k'*n+i',:] must be 0 unless (i',k')==(i,k)
        i_range = np.arange(n)
        k_range = np.arange(p)
        I, K = np.meshgrid(i_range, k_range, indexing='ij')
        I_flat = I.flatten()
        K_flat = K.flatten()
        # prod vectors: (n*p, rank)
        U_rows = U[I_flat * m + j]  # (n*p, rank)
        V_rows = V[j * p + K_flat]  # (n*p, rank)
        prods = U_rows * V_rows  # (n*p, rank)
        # W rows for all (i',k')
        W_rows = W[K_flat * n + I_flat]  # (n*p, rank)
        # Matrix M where M[a,b] = dot(prods[a], W_rows[b])
        # M should be identity (diagonal=1 already checked, off-diagonal=0)
        M = prods @ W_rows.T  # (n*p, n*p)
        # Check off-diagonal entries are ~0
        mask = ~np.eye(n * p, dtype=bool)
        if np.abs(M[mask]).max() > tol:
            return False
    return True


def _violation_vector(U, V, W, n=2, m=5, p=6):
    """Return vector of |residual| over all (diag + off-diag) tensor entries.
    Valid decomposition => all entries 0 (diag residual = |sum-1|, off = |sum|)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    n, m, p = 2, 5, 6
    i_vals = np.repeat(np.arange(n), m * p)
    j_vals = np.tile(np.repeat(np.arange(m), p), n)
    k_vals = np.tile(np.arange(p), n * m)
    u_idx = i_vals * m + j_vals
    v_idx = j_vals * p + k_vals
    w_idx = k_vals * n + i_vals
    diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
    out = [abs(x - 1.0) for x in diag_vals]
    for j in range(m):
        i_range = np.arange(n)
        k_range = np.arange(p)
        I, K = np.meshgrid(i_range, k_range, indexing='ij')
        I_flat = I.flatten()
        K_flat = K.flatten()
        U_rows = U[I_flat * m + j]  # (n*p, rank)
        V_rows = V[j * p + K_flat]  # (n*p, rank)
        prods = U_rows * V_rows  # (n*p, rank)
        W_rows = W[K_flat * n + I_flat]  # (n*p, rank)
        M = prods @ W_rows.T  # (n*p, n*p)
        mask = ~np.eye(n * p, dtype=bool)
        out.extend(np.abs(M[mask]).tolist())
    return np.array(out)


def _schoolbook_rank(n=2, m=5, p=6):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _schoolbook_arrays(n=2, m=5, p=6):
    """Return (U,V,W) numpy arrays of the schoolbook rank n*m*p decomposition,
    plus the list of (i,j,k) per column for merge bookkeeping."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    triples = []
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                triples.append((i, j, k))
                column += 1
    return U, V, W, triples


def construct():
    start_time = time.time()
    deadline = start_time + 132  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60

    if _verify_rank_fast(best["U"], best["V"], best["W"]):
        pass
    else:
        return best

    random.seed(42)
    possible_values = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 2.5, -2.5, 3, -3]
    nonzero_values = [v for v in possible_values if v != 0]

    def _check_ones_fast(U, V, W, tol=1e-6):
        """Fast check of just the 60 diagonal entries."""
        i_vals = np.repeat(np.arange(n), m * p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n * m)
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        diag_vals = np.einsum('ac,bc,ac->a', U[u_idx], V[v_idx], W[w_idx])
        return np.allclose(diag_vals, 1.0, atol=tol)

    def _smooth_objective(U, V, W, beta=30.0):
        """Soft-min surrogate: log-sum-exp of |residual|^2. Smooth, differentiable-ish
        (in expectation) proxy for the max-violation. Lower is better."""
        viol = _violation_vector(U, V, W)
        sq = viol * viol
        mx = sq.max()
        if mx < 1e-14:
            return 0.0
        # stable log-sum-exp
        shifted = sq - mx
        return mx + np.log(np.sum(np.exp(beta * (shifted - shifted.max()))) ) / beta + np.log(len(sq)) / beta
        # simpler: just max + small log-sum correction
    # Use a cleaner smooth objective
    def _objective(U, V, W, beta=40.0):
        viol = _violation_vector(U, V, W)
        sq = viol * viol
        mx = sq.max()
        if mx < 1e-12:
            return 0.0
        # log-sum-exp of sq, scaled: approximates max but smooth
        s = sq - mx
        lse = mx + np.log(np.sum(np.exp(beta * s))) / beta
        return lse

    def _local_search(U, V, W, rank, max_steps=400, beta=40.0):
        """Smooth-surrogate-guided local search with coordinate descent.
        Try changing one entry at a time; accept if objective decreases."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        cur_obj = _objective(U, V, W, beta)
        if cur_obj < 1e-7 and _verify_rank_fast(U, V, W):
            return U, V, W, True
        
        # Precompute indices for faster access
        u_indices = [(r, c) for r in range(n * m) for c in range(rank)]
        v_indices = [(r, c) for r in range(m * p) for c in range(rank)]
        w_indices = [(r, c) for r in range(p * n) for c in range(rank)]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try a few random entries per step
            for _ in range(3):
                if time.time() > deadline:
                    break
                mat = random.randint(0, 2)
                if mat == 0:
                    row, col = random.choice(u_indices)
                    old = U[row, col]
                    new_val = random.choice(possible_values)
                    if new_val == old:
                        continue
                    U[row, col] = new_val
                    new_obj = _objective(U, V, W, beta)
                    if new_obj < cur_obj - 1e-12:
                        cur_obj = new_obj
                        improved = True
                        if step % 20 == 0 and new_obj < 1e-7:
                            if _verify_rank_fast(U, V, W):
                                return U, V, W, True
                    else:
                        U[row, col] = old
                elif mat == 1:
                    row, col = random.choice(v_indices)
                    old = V[row, col]
                    new_val = random.choice(possible_values)
                    if new_val == old:
                        continue
                    V[row, col] = new_val
                    new_obj = _objective(U, V, W, beta)
                    if new_obj < cur_obj - 1e-12:
                        cur_obj = new_obj
                        improved = True
                        if step % 20 == 0 and new_obj < 1e-7:
                            if _verify_rank_fast(U, V, W):
                                return U, V, W, True
                    else:
                        V[row, col] = old
                else:
                    row, col = random.choice(w_indices)
                    old = W[row, col]
                    new_val = random.choice(possible_values)
                    if new_val == old:
                        continue
                    W[row, col] = new_val
                    new_obj = _objective(U, V, W, beta)
                    if new_obj < cur_obj - 1e-12:
                        cur_obj = new_obj
                        improved = True
                        if step % 20 == 0 and new_obj < 1e-7:
                            if _verify_rank_fast(U, V, W):
                                return U, V, W, True
                    else:
                        W[row, col] = old
            
            if not improved and step > 100:
                # Stuck, try a larger perturbation
                for _ in range(5):
                    mat = random.randint(0, 2)
                    if mat == 0:
                        row, col = random.choice(u_indices)
                        U[row, col] = random.choice(possible_values)
                    elif mat == 1:
                        row, col = random.choice(v_indices)
                        V[row, col] = random.choice(possible_values)
                    else:
                        row, col = random.choice(w_indices)
                        W[row, col] = random.choice(possible_values)
                new_obj = _objective(U, V, W, beta)
                if new_obj < cur_obj + 0.1:  # Allow some increase to escape local minima
                    cur_obj = new_obj
        
        if cur_obj < 1e-7 and _verify_rank_fast(U, V, W):
            return U, V, W, True
        return U, V, W, False

    def _structured_candidate(rank, seed=None):
        """Structured candidate exploiting the 2-row structure: share terms between
        i=0 and i=1 via the m*p (j,k) pairs, then add random extra terms."""
        if seed is not None:
            random.seed(seed)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        if rank >= m * p:
            for c in range(m * p):
                j = c // p
                k = c % p
                V[j * p + k, c] = 1.0
                u0 = random.choice(nonzero_values)
                u1 = random.choice(nonzero_values)
                w0 = round((1.0 / u0) * 2) / 2
                w1 = round((1.0 / u1) * 2) / 2
                U[0 * m + j, c] = u0
                U[1 * m + j, c] = u1
                W[k * n + 0, c] = w0
                W[k * n + 1, c] = w1
        for c in range(m * p, rank):
            for _ in range(random.randint(1, 4)):
                mat = random.randint(0, 2)
                if mat == 0:
                    U[random.randint(0, n * m - 1), c] = random.choice(possible_values)
                elif mat == 1:
                    V[random.randint(0, m * p - 1), c] = random.choice(possible_values)
                else:
                    W[random.randint(0, p * n - 1), c] = random.choice(possible_values)
        return U, V, W

    def _merge_search(seed=0, max_iters=20000):
        """Start from schoolbook rank 60, greedily merge/remove columns to reduce rank.
        A column can be removed if the residual it contributed can be absorbed by
        adjusting other columns (we test: zero it out, then try to fix via local
        search on the remaining). Returns (U,V,W,rank) or None."""
        random.seed(seed)
        U, V, W, triples = _schoolbook_arrays()
        rank = n * m * p
        active = list(range(rank))
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not active:
                break
            # pick a random active column to try removing
            idx = random.randint(0, len(active) - 1)
            col = active[idx]
            # save
            U_save, V_save, W_save = U[:, col].copy(), V[:, col].copy(), W[:, col].copy()
            # zero it out
            U[:, col] = 0
            V[:, col] = 0
            W[:, col] = 0
            # try to repair with a short local search
            U2, V2, W2, ok = _local_search(U, V, W, rank, max_steps=60, beta=60.0)
            if ok:
                # accept: drop this column
                active.pop(idx)
                # compact (optional) - keep as is with a zero column; final compact later
                U, V, W = U2, V2, W2
            else:
                # revert
                U[:, col] = U_save
                V[:, col] = V_save
                W[:, col] = W_save
        if not active:
            return None
        # compact to active columns
        U_c = U[:, active]
        V_c = V[:, active]
        W_c = W[:, active]
        if _verify_rank_fast(U_c, V_c, W_c):
            return U_c, V_c, W_c, len(active)
        return None

    # ---- Main search loop ----
    # Phase A: structured + random candidates for ranks 59..30
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        found = False

        # Strategy 1: structured candidates
        for attempt in range(40):
            if time.time() > deadline or found:
                break
            seed = 1000 + target_rank * 100 + attempt
            U, V, W = _structured_candidate(target_rank, seed=seed)
            if _check_ones_fast(U, V, W):
                if _verify_rank_fast(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    U2, V2, W2, success = _local_search(U, V, W, target_rank, max_steps=300)
                    if success:
                        best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        best_rank = target_rank
                        found = True
                        break

        if found:
            continue

        # Strategy 2: random diverse initialization
        for attempt in range(80):
            if time.time() > deadline or found:
                break
            U = np.array([[random.choice(possible_values) for _ in range(target_rank)]
                          for _ in range(n * m)], dtype=float)
            V = np.array([[random.choice(possible_values) for _ in range(target_rank)]
                          for _ in range(m * p)], dtype=float)
            W = np.array([[random.choice(possible_values) for _ in range(target_rank)]
                          for _ in range(p * n)], dtype=float)
            if _check_ones_fast(U, V, W):
                if _verify_rank_fast(U, V, W):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    U2, V2, W2, success = _local_search(U, V, W, target_rank, max_steps=300)
                    if success:
                        best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        best_rank = target_rank
                        found = True
                        break

    # Phase B: schoolbook-merge to push rank below 60 (only if we still have time)
    if time.time() < deadline - 30:
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            res = _merge_search(seed=seed, max_iters=15000)
            if res is not None:
                U_c, V_c, W_c, rk = res
                if rk < best_rank:
                    best = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                    best_rank = rk
    
    # Phase C: Polish - if we have a good candidate, try to reduce rank further
    if time.time() < deadline - 10 and best_rank < n * m * p:
        # Try to find a lower rank by starting from current best and merging
        current_U = np.array(best["U"], dtype=float)
        current_V = np.array(best["V"], dtype=float)
        current_W = np.array(best["W"], dtype=float)
        current_rank = best_rank
        
        # Try to reduce by 1, 2, 3...
        for reduction in range(1, 5):
            target = current_rank - reduction
            if target < 10:
                break
            if time.time() > deadline - 5:
                break
            
            # Try multiple random restarts from perturbed version of current best
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                # Create a perturbed version
                U_pert = current_U.copy()
                V_pert = current_V.copy()
                W_pert = current_W.copy()
                
                # Add some random noise to non-zero entries
                for _ in range(10):
                    mat = random.randint(0, 2)
                    if mat == 0:
                        r = random.randint(0, n * m - 1)
                        c = random.randint(0, current_rank - 1)
                        if U_pert[r, c] != 0:
                            U_pert[r, c] += random.choice([-0.5, 0.5])
                    elif mat == 1:
                        r = random.randint(0, m * p - 1)
                        c = random.randint(0, current_rank - 1)
                        if V_pert[r, c] != 0:
                            V_pert[r, c] += random.choice([-0.5, 0.5])
                    else:
                        r = random.randint(0, p * n - 1)
                        c = random.randint(0, current_rank - 1)
                        if W_pert[r, c] != 0:
                            W_pert[r, c] += random.choice([-0.5, 0.5])
                
                # Round to half-integers
                U_pert = np.round(U_pert * 2) / 2
                V_pert = np.round(V_pert * 2) / 2
                W_pert = np.round(W_pert * 2) / 2
                
                # Try local search to reduce rank
                U2, V2, W2, success = _local_search(U_pert, V_pert, W_pert, current_rank, max_steps=200)
                if success:
                    # Check if we can actually reduce the rank
                    # Try zeroing out some columns and see if it's still valid
                    for col_to_remove in range(current_rank):
                        if time.time() > deadline - 5:
                            break
                        U_test = U2.copy()
                        V_test = V2.copy()
                        W_test = W2.copy()
                        U_test[:, col_to_remove] = 0
                        V_test[:, col_to_remove] = 0
                        W_test[:, col_to_remove] = 0
                        
                        # Remove the column
                        mask = np.ones(current_rank, dtype=bool)
                        mask[col_to_remove] = False
                        U_compact = U_test[:, mask]
                        V_compact = V_test[:, mask]
                        W_compact = W_test[:, mask]
                        
                        if _verify_rank_fast(U_compact, V_compact, W_compact):
                            new_rank = current_rank - 1
                            if new_rank < best_rank:
                                best = {"U": U_compact.tolist(), "V": V_compact.tolist(), "W": W_compact.tolist()}
                                best_rank = new_rank
                                current_rank = new_rank
                                current_U = U_compact
                                current_V = V_compact
                                current_W = W_compact
                            break

    return best

# EVOLVE-BLOCK-END
