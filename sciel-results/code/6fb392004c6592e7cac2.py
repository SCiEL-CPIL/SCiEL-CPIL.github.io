"""Rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication.

Uses a maximum-coverage greedy over all 2x2x2 Strassen blocks with a
smarter scoring function that accounts for future overlap potential,
then fills the remainder with schoolbook rank-1 terms. Also tries a
local search refinement to swap in better blocks.
"""
import time
import random

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0  # safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True


def _make_strassen_2x2x2_pattern(i0, i1, j0, j1, k0, k1, n, m, p):
    """Create a standard Strassen 2x2x2 pattern as a list of dicts."""
    pattern = []
    # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    u = {i0*m+j0: 1, i1*m+j1: 1}
    v = {j0*p+k0: 1, j1*p+k1: 1}
    w = {k0*n+i0: 1, k1*n+i1: 1}
    pattern.append((u, v, w))
    # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
    u = {i1*m+j0: 1, i1*m+j1: 1}
    v = {j0*p+k0: 1}
    w = {k0*n+i1: 1, k1*n+i1: -1}
    pattern.append((u, v, w))
    # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
    u = {i0*m+j0: 1}
    v = {j0*p+k1: 1, j1*p+k1: -1}
    w = {k1*n+i0: 1, k1*n+i1: 1}
    pattern.append((u, v, w))
    # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
    u = {i1*m+j1: 1}
    v = {j1*p+k0: 1, j0*p+k0: -1}
    w = {k0*n+i0: 1, k0*n+i1: 1}
    pattern.append((u, v, w))
    # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
    u = {i0*m+j0: 1, i0*m+j1: 1}
    v = {j1*p+k1: 1}
    w = {k0*n+i0: -1, k1*n+i0: 1}
    pattern.append((u, v, w))
    # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    u = {i1*m+j0: 1, i0*m+j0: -1}
    v = {j0*p+k0: 1, j0*p+k1: 1}
    w = {k1*n+i1: 1}
    pattern.append((u, v, w))
    # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    u = {i0*m+j1: 1, i1*m+j1: -1}
    v = {j1*p+k0: 1, j1*p+k1: 1}
    w = {k0*n+i0: 1}
    pattern.append((u, v, w))
    return pattern


def _block_covers(pattern, n, m, p):
    """Return the set of (i,j,k) tuples covered by a pattern."""
    covered = set()
    for u_dict, v_dict, w_dict in pattern:
        for u_idx in u_dict:
            i, j = divmod(u_idx, m)
            for v_idx in v_dict:
                jj, k = divmod(v_idx, p)
                if j == jj:
                    for w_idx in w_dict:
                        kk, ii = divmod(w_idx, n)
                        if ii == i and kk == k:
                            covered.add((i, j, k))
    return covered


def _apply_block_pattern(U, V, W, col_start, pattern):
    """Apply a block pattern to U, V, W starting at col_start."""
    for r, (u_dict, v_dict, w_dict) in enumerate(pattern):
        col = col_start + r
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx][col] += coeff
    return col_start + len(pattern)


def _build_decomposition(used_blocks, n, m, p):
    """Build U, V, W from a list of (pattern, covers) and return (U, V, W, rank)."""
    total_rank = sum(len(pattern) for pattern, _ in used_blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    for pattern, _ in used_blocks:
        col = _apply_block_pattern(U, V, W, col, pattern)
    return U, V, W, total_rank


def _greedy_max_coverage(blocks, all_tuples, max_blocks, time_fn, overlap_penalty=0.0):
    """Greedy maximum coverage: pick blocks that cover the most new triples.
    
    overlap_penalty: penalize blocks that overlap with already-used blocks,
    to encourage disjointness and leave more room for future blocks.
    """
    uncovered = set(all_tuples)
    used = []
    used_covers = []  # list of cover sets for overlap calculation
    for _ in range(max_blocks):
        if time_fn():
            break
        best_block = None
        best_score = -1
        best_new_covers = None
        for pattern, covers in blocks:
            new_covers = covers & uncovered
            if len(new_covers) == 0:
                continue
            # Base score: new coverage
            score = len(new_covers)
            # Penalty for overlap with already-used blocks
            if overlap_penalty > 0:
                overlap_count = 0
                for uc in used_covers:
                    overlap_count += len(covers & uc)
                score -= overlap_penalty * overlap_count
            if score > best_score:
                best_score = score
                best_block = (pattern, covers)
                best_new_covers = new_covers
        if best_block is None:
            break
        used.append(best_block)
        used_covers.append(best_block[1])
        uncovered -= best_block[1]
    return used, uncovered


def _local_search_refine(used, blocks, all_tuples, time_fn, max_iters=200):
    """Try to improve the solution by swapping one block for another.
    
    A swap is beneficial if the new block covers more uncovered triples
    (after removing the old block's coverage) than the old block did.
    """
    if not used:
        return used
    
    # Build initial uncovered set
    covered = set()
    for pattern, covers in used:
        covered |= covers
    uncovered = set(all_tuples) - covered
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time_fn():
            break
        improved = False
        iters += 1
        best_swap = None  # (remove_idx, add_pattern, add_covers, gain)
        best_gain = 0
        
        for remove_idx, (rem_pattern, rem_covers) in enumerate(used):
            # If we remove this block, what becomes uncovered?
            # Rebuild covered without this block
            covered_without = set()
            for j, (_, cov) in enumerate(used):
                if j != remove_idx:
                    covered_without |= cov
            new_uncovered = set(all_tuples) - covered_without
            
            # Try adding each candidate block
            for pattern, covers in blocks:
                new_covers = covers & new_uncovered
                gain = len(new_covers) - len(rem_covers & new_uncovered)
                # Actually: removing rem_covers frees up |rem_covers & new_uncovered| slots
                # Adding covers adds |new_covers| new ones
                # Net change in covered = len(new_covers) - len(rem_covers & new_uncovered)
                # But we also need: the new block shouldn't fully overlap with existing
                # Simpler: total covered after swap = covered_without | covers
                # total covered before = covered
                # gain = |covered_without | covers| - |covered|
                total_after = len(covered_without | covers)
                total_before = len(covered)
                gain = total_after - total_before
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (remove_idx, pattern, covers)
        
        if best_swap is not None and best_gain > 0:
            remove_idx, new_pattern, new_covers = best_swap
            used[remove_idx] = (new_pattern, new_covers)
            improved = True
    
    return used


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    def time_check():
        return time.time() - _t0 > _DEADLINE - 5.0

    # Generate all possible 2x2x2 Strassen blocks
    blocks = []
    for i0 in range(n - 1):
        for i1 in range(i0 + 1, n):
            for j0 in range(m - 1):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p - 1):
                        for k1 in range(k0 + 1, p):
                            pattern = _make_strassen_2x2x2_pattern(i0, i1, j0, j1, k0, k1, n, m, p)
                            covers = _block_covers(pattern, n, m, p)
                            blocks.append((pattern, covers))

    all_tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_tuples = len(all_tuples)  # 84

    # Try multiple strategies: different overlap penalties and max blocks
    best_result = None
    best_rank = n * m * p  # schoolbook fallback

    # Strategy 1: Pure greedy (overlap_penalty=0)
    for max_blocks in range(10, 0, -1):
        if time_check():
            break
        used, uncovered = _greedy_max_coverage(blocks, all_tuples, max_blocks, time_check, overlap_penalty=0.0)
        if not used:
            continue
        # Try local search refinement
        used = _local_search_refine(used, blocks, all_tuples, time_check, max_iters=100)
        
        # Add schoolbook for remaining
        covered = set()
        for pattern, covers in used:
            covered |= covers
        uncovered = set(all_tuples) - covered
        for (i, j, k) in sorted(uncovered):
            u = {i * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + i: 1}
            used.append(([(u, v, w)], {(i, j, k)}))
        rank = sum(len(pat) for pat, _ in used)
        if rank < best_rank:
            best_rank = rank
            best_result = used
        if rank <= 77:
            break

    # Strategy 2: Greedy with overlap penalty
    if not time_check():
        for penalty in [0.5, 1.0, 2.0]:
            if time_check():
                break
            for max_blocks in range(10, 0, -1):
                if time_check():
                    break
                used, uncovered = _greedy_max_coverage(blocks, all_tuples, max_blocks, time_check, overlap_penalty=penalty)
                if not used:
                    continue
                # Try local search refinement
                used = _local_search_refine(used, blocks, all_tuples, time_check, max_iters=50)
                
                covered = set()
                for pattern, covers in used:
                    covered |= covers
                uncovered = set(all_tuples) - covered
                for (i, j, k) in sorted(uncovered):
                    u = {i * m + j: 1}
                    v = {j * p + k: 1}
                    w = {k * n + i: 1}
                    used.append(([(u, v, w)], {(i, j, k)}))
                rank = sum(len(pat) for pat, _ in used)
                if rank < best_rank:
                    best_rank = rank
                    best_result = used
                if rank <= 77:
                    break

    # Strategy 3: Random restarts with local search
    if not time_check():
        rng = random.Random(42)
        for restart in range(5):
            if time_check():
                break
            # Random subset of blocks, greedy on that
            shuffled = blocks[:]
            rng.shuffle(shuffled)
            subset = shuffled[:50]  # use 50 random blocks
            for max_blocks in range(8, 0, -1):
                if time_check():
                    break
                used, uncovered = _greedy_max_coverage(subset, all_tuples, max_blocks, time_check, overlap_penalty=0.0)
                if not used:
                    continue
                used = _local_search_refine(used, blocks, all_tuples, time_check, max_iters=100)
                
                covered = set()
                for pattern, covers in used:
                    covered |= covers
                uncovered = set(all_tuples) - covered
                for (i, j, k) in sorted(uncovered):
                    u = {i * m + j: 1}
                    v = {j * p + k: 1}
                    w = {k * n + i: 1}
                    used.append(([(u, v, w)], {(i, j, k)}))
                rank = sum(len(pat) for pat, _ in used)
                if rank < best_rank:
                    best_rank = rank
                    best_result = used
                if rank <= 77:
                    break

    if best_result is None:
        # Pure schoolbook
        best_result = []
        for (i, j, k) in all_tuples:
            u = {i * m + j: 1}
            v = {j * p + k: 1}
            w = {k * n + i: 1}
            best_result.append(([(u, v, w)], {(i, j, k)}))
        best_rank = n * m * p

    # Build final U, V, W
    U, V, W, rank = _build_decomposition(best_result, n, m, p)

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback: deterministic known-good rank-78
        rank = 78
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
            U[i0 * m + j0][col_start] = 1
            U[i1 * m + j1][col_start] = 1
            V[j0 * p + k0][col_start] = 1
            V[j1 * p + k1][col_start] = 1
            W[k0 * n + i0][col_start] = 1
            W[k1 * n + i1][col_start] = 1
            U[i1 * m + j0][col_start + 1] = 1
            U[i1 * m + j1][col_start + 1] = 1
            V[j0 * p + k0][col_start + 1] = 1
            W[k0 * n + i1][col_start + 1] = 1
            W[k1 * n + i1][col_start + 1] = -1
            U[i0 * m + j0][col_start + 2] = 1
            V[j0 * p + k1][col_start + 2] = 1
            V[j1 * p + k1][col_start + 2] = -1
            W[k1 * n + i0][col_start + 2] = 1
            W[k1 * n + i1][col_start + 2] = 1
            U[i1 * m + j1][col_start + 3] = 1
            V[j1 * p + k0][col_start + 3] = 1
            V[j0 * p + k0][col_start + 3] = -1
            W[k0 * n + i0][col_start + 3] = 1
            W[k0 * n + i1][col_start + 3] = 1
            U[i0 * m + j0][col_start + 4] = 1
            U[i0 * m + j1][col_start + 4] = 1
            V[j1 * p + k1][col_start + 4] = 1
            W[k0 * n + i0][col_start + 4] = -1
            W[k1 * n + i0][col_start + 4] = 1
            U[i1 * m + j0][col_start + 5] = 1
            U[i0 * m + j0][col_start + 5] = -1
            V[j0 * p + k0][col_start + 5] = 1
            V[j0 * p + k1][col_start + 5] = 1
            W[k1 * n + i1][col_start + 5] = 1
            U[i0 * m + j1][col_start + 6] = 1
            U[i1 * m + j1][col_start + 6] = -1
            V[j1 * p + k0][col_start + 6] = 1
            V[j1 * p + k1][col_start + 6] = 1
            W[k0 * n + i0][col_start + 6] = 1
            return col_start + 7

        blocks6 = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 0, 1, 4, 5),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 2, 3, 2, 3),
            (0, 1, 2, 3, 4, 5),
        ]
        covered = set()
        for (i0, i1, j0, j1, k0, k1) in blocks6:
            col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        covered.add((ii, jj, kk))

        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    if (i0, j0, k0) in covered:
                        continue
                    U[i0 * m + j0][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + i0][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
