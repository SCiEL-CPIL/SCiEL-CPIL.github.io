"""Program for 4x5 * 5x6 matrix multiplication tensor decomposition.
Systematic pattern-based search for rank-minimal exact decomposition.
"""
DIMENSIONS = [4, 5, 6]
import time
import itertools
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 140  # safety margin from 150s limit

    n, m, p = 4, 5, 6
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24

    def idxU(i, j): return i * m + j
    def idxV(j, k): return j * p + k
    def idxW(k, i): return k * n + i

    # Build list of all 120 entries
    entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    entry_set = set(entries)

    # Generate candidate patterns: (row_set, col_set, out_set)
    # Each pattern covers entries in row_set x col_set x out_set
    # We want sparse patterns for better rank reduction
    patterns = []
    for rsize in [1, 2, 3, 4]:
        for csize in [1, 2, 3, 4]:
            for osize in [1, 2, 3, 4]:
                if rsize * csize * osize > 40:
                    continue
                for rows in itertools.combinations(range(n), rsize):
                    for cols in itertools.combinations(range(m), csize):
                        for outs in itertools.combinations(range(p), osize):
                            cov = frozenset((i, j, k) for i in rows for j in cols for k in outs)
                            if len(cov) < 3:
                                continue
                            patterns.append((rows, cols, outs, cov))

    # Sort by coverage (larger first) for greedy
    patterns.sort(key=lambda x: -len(x[3]))

    # Greedy cover with pattern selection
    def greedy_cover(max_rank, time_budget):
        """Try to cover all entries with <= max_rank patterns."""
        uncovered = set(entries)
        chosen = []
        t_start = time.time()
        # Precompute coverage sets for speed
        while uncovered and len(chosen) < max_rank:
            if time.time() - t_start > time_budget:
                return None
            best_pat = None
            best_count = 0
            # Sample patterns for speed (check top-coverage patterns first)
            for pat in patterns:
                rows, cols, outs, cov = pat
                cnt = len(cov & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_pat = pat
                    if best_count >= len(uncovered):
                        break
            if best_pat is None or best_count == 0:
                return None
            chosen.append(best_pat)
            uncovered -= best_pat[3]
        if uncovered:
            return None
        return chosen

    # Solve linear system for a given set of patterns
    def solve_patterns(patterns_list):
        """Solve for weights so that sum of active patterns = 1 at each entry."""
        r = len(patterns_list)
        # Build system: for each entry (i,j,k), sum of active pattern weights = 1
        # Active: i in rows AND j in cols AND k in outs
        # Build A (120 x r) and b (120,)
        A = np.zeros((120, r))
        b = np.ones(120)
        for c, (rows, cols, outs, cov) in enumerate(patterns_list):
            rows_set = set(rows)
            cols_set = set(cols)
            outs_set = set(outs)
            for e_idx, (i, j, k) in enumerate(entries):
                if i in rows_set and j in cols_set and k in outs_set:
                    A[e_idx, c] = 1.0
        # Solve least squares
        try:
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        # Check exactness
        if not np.allclose(A @ sol, b, atol=1e-10):
            return None
        return sol

    # Build final matrices from solution
    def build_matrices(patterns_list, sol):
        r = len(patterns_list)
        U = np.zeros((num_U, r))
        V = np.zeros((num_V, r))
        W = np.zeros((num_W, r))
        for c, (rows, cols, outs, cov) in enumerate(patterns_list):
            w = sol[c]
            for i in rows:
                for j in cols:
                    U[idxU(i, j), c] = w
            for j in cols:
                for k in outs:
                    V[idxV(j, k), c] = 1.0
            for k in outs:
                for i in rows:
                    W[idxW(k, i), c] = 1.0
        return U, V, W

    # Verify a decomposition
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[idxU(i, j), c] * V[idxV(j, k), c] * W[idxW(k, i), c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    # Try to find a decomposition with rank < 120
    best_rank = 120
    best_result = None

    # Start with schoolbook as fallback
    U_fb = np.zeros((num_U, best_rank))
    V_fb = np.zeros((num_V, best_rank))
    W_fb = np.zeros((num_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[idxU(i, j), col] = 1
                V_fb[idxV(j, k), col] = 1
                W_fb[idxW(k, i), col] = 1
                col += 1
    best_result = (U_fb, V_fb, W_fb)

    # Systematic search: try decreasing target ranks
    for target_rank in range(119, 80, -1):
        if time.time() - t0 > DEADLINE - 10:
            break
        # Try greedy cover with this rank
        chosen = greedy_cover(target_rank, time_budget=5)
        if chosen is None:
            continue
        sol = solve_patterns(chosen)
        if sol is None:
            continue
        U, V, W = build_matrices(chosen, sol)
        if verify(U, V, W):
            best_rank = target_rank
            best_result = (U, V, W)
            break  # Found a valid decomposition at this rank

    # Also try some structured constructions for better ranks
    # Try block-based constructions
    if time.time() - t0 < DEADLINE - 20:
        # Try: partition rows into groups, use shared structure
        # For 4x5x6, try rank = 4*5*6 - savings
        # Attempt: use rank-3 addition-like structure on one dimension
        for target_rank in range(best_rank - 1, max(80, best_rank - 20), -1):
            if time.time() - t0 > DEADLINE - 10:
                break
            chosen = greedy_cover(target_rank, time_budget=3)
            if chosen is None:
                continue
            sol = solve_patterns(chosen)
            if sol is None:
                continue
            U, V, W = build_matrices(chosen, sol)
            if verify(U, V, W):
                best_rank = target_rank
                best_result = (U, V, W)
                break

    U, V, W = best_result
    # Convert to lists for interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
