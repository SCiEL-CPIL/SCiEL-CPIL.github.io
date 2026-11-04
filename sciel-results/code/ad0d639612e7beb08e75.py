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

    # Off-diagonal entries must be 0. For fixed j, check (i,k) vs (i',k') mismatches.
    for j in range(m):
        for i in range(n):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                prod = U[a] * V[b]  # (rank,)
                for i_prime in range(n):
                    for k_prime in range(p):
                        if i == i_prime and k == k_prime:
                            continue
                        c = k_prime * n + i_prime
                        if abs(np.dot(prod, W[c])) > tol:
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
        for i in range(n):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                prod = U[a] * V[b]
                for i_prime in range(n):
                    for k_prime in range(p):
                        if i == i_prime and k == k_prime:
                            continue
                        c = k_prime * n + i_prime
                        out.append(abs(np.dot(prod, W[c])))
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
        lse = mx + np.log(np.sum(np.exp(beta * s)) ) / beta
        return lse

    def _local_search(U, V, W, rank, max_steps=400, beta=40.0):
        """Smooth-surrogate-guided local search. Try entry changes; accept if
        objective (log-sum-exp of squared residual) decreases. Periodic exact verify."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        cur_obj = _objective(U, V, W, beta)
        if cur_obj < 1e-7 and _verify_rank_fast(U, V, W):
            return U, V, W, True
        for step in range(max_steps):
            if time.time() > deadline:
                break
            # pick a random entry across the three matrices
            mat = random.randint(0, 2)
            if mat == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                old = U[row, col]
                U[row, col] = random.choice(possible_values)
            elif mat == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                old = V[row, col]
                V[row, col] = random.choice(possible_values)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                old = W[row, col]
                W[row, col] = random.choice(possible_values)
            new_obj = _objective(U, V, W, beta)
            if new_obj < cur_obj - 1e-12:
                cur_obj = new_obj
                if step % 20 == 0 and new_obj < 1e-7:
                    if _verify_rank_fast(U, V, W):
                        return U, V, W, True
            else:
                # revert
                if mat == 0:
                    U[row, col] = old
                elif mat == 1:
                    V[row, col] = old
                else:
                    W[row, col] = old
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

    return best

# EVOLVE-BLOCK-END
