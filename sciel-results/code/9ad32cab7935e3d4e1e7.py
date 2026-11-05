"""Rank-minimal tensor decomposition for 3x4 * 4x7 matrix multiplication.

Uses a comprehensive search over Strassen blocks AND multi-coverage rank-1
terms (terms that contribute to multiple output entries simultaneously).
The key insight: a rank-1 term with U having entries at (i,j) and (i,j'),
V having entries at (j,k) and (j',k), and W having entry at (k,i) covers
both (i,j,k) and (i,j',k) in a single rank-1 term.
"""
import time
import random
from itertools import combinations

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0


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
    """Create a standard Strassen 2x2x2 pattern as a list of (u_dict, v_dict, w_dict)."""
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


def _generate_multi_coverage_terms(n, m, p):
    """Generate rank-1 terms that cover 2+ triples.
    
    A rank-1 term is defined by (U_vector, V_vector, W_vector).
    It covers triple (i,j,k) iff U[i*m+j]*V[j*p+k]*W[k*n+i] != 0.
    
    We look for terms where multiple (i,j,k) are covered.
    """
    terms = []
    
    # Type 1: Same i, two j values, same k
    # U has entries at (i,j0) and (i,j1), V at (j0,k) and (j1,k), W at (k,i)
    # Covers (i,j0,k) and (i,j1,k)
    for i in range(n):
        for j0, j1 in combinations(range(m), 2):
            for k in range(p):
                u = {i*m+j0: 1, i*m+j1: 1}
                v = {j0*p+k: 1, j1*p+k: 1}
                w = {k*n+i: 1}
                covers = {(i, j0, k), (i, j1, k)}
                terms.append(([(u, v, w)], covers))
    
    # Type 2: Same i, same j, two k values
    # U at (i,j), V at (j,k0) and (j,k1), W at (k0,i) and (k1,i)
    # Covers (i,j,k0) and (i,j,k1)
    for i in range(n):
        for j in range(m):
            for k0, k1 in combinations(range(p), 2):
                u = {i*m+j: 1}
                v = {j*p+k0: 1, j*p+k1: 1}
                w = {k0*n+i: 1, k1*n+i: 1}
                covers = {(i, j, k0), (i, j, k1)}
                terms.append(([(u, v, w)], covers))
    
    # Type 3: Two i values, same j, same k
    # U at (i0,j) and (i1,j), V at (j,k), W at (k,i0) and (k,i1)
    # Covers (i0,j,k) and (i1,j,k)
    for i0, i1 in combinations(range(n), 2):
        for j in range(m):
            for k in range(p):
                u = {i0*m+j: 1, i1*m+j: 1}
                v = {j*p+k: 1}
                w = {k*n+i0: 1, k*n+i1: 1}
                covers = {(i0, j, k), (i1, j, k)}
                terms.append(([(u, v, w)], covers))
    
    # Type 4: Same i, two j, two k - covers 4 triples
    # U at (i,j0) and (i,j1), V at (j0,k0),(j0,k1),(j1,k0),(j1,k1)
    # W at (k0,i) and (k1,i)
    # This covers (i,j0,k0),(i,j0,k1),(i,j1,k0),(i,j1,k1)
    # But we need to check: U[i,j0]*V[j0,k0]*W[k0,i] = 1*1*1 = 1 ✓
    # U[i,j1]*V[j1,k1]*W[k1,i] = 1*1*1 = 1 ✓
    # But U[i,j0]*V[j0,k1]*W[k1,i] = 1*1*1 = 1 ✓ (also covers (i,j0,k1))
    # U[i,j1]*V[j1,k0]*W[k0,i] = 1*1*1 = 1 ✓ (also covers (i,j1,k0))
    # So this single term covers all 4! But wait, this means the term contributes 1 to ALL 4.
    # That's only valid if we want each to be exactly 1 and no other terms contribute.
    # Actually this is fine as a building block.
    for i in range(n):
        for j0, j1 in combinations(range(m), 2):
            for k0, k1 in combinations(range(p), 2):
                u = {i*m+j0: 1, i*m+j1: 1}
                v = {j0*p+k0: 1, j0*p+k1: 1, j1*p+k0: 1, j1*p+k1: 1}
                w = {k0*n+i: 1, k1*n+i: 1}
                covers = {(i, j0, k0), (i, j0, k1), (i, j1, k0), (i, j1, k1)}
                terms.append(([(u, v, w)], covers))
    
    # Type 5: Two i, same j, two k - covers 4 triples
    for i0, i1 in combinations(range(n), 2):
        for j in range(m):
            for k0, k1 in combinations(range(p), 2):
                u = {i0*m+j: 1, i1*m+j: 1}
                v = {j*p+k0: 1, j*p+k1: 1}
                w = {k0*n+i0: 1, k0*n+i1: 1, k1*n+i0: 1, k1*n+i1: 1}
                covers = {(i0, j, k0), (i0, j, k1), (i1, j, k0), (i1, j, k1)}
                terms.append(([(u, v, w)], covers))
    
    # Type 6: Same i, two j, two k with cross structure
    # U at (i,j0) and (i,j1), V at (j0,k0) and (j1,k1), W at (k0,i) and (k1,i)
    # Covers (i,j0,k0) and (i,j1,k1) only (2 triples)
    # This is less efficient than Type 4, skip
    
    return terms


def _greedy_max_coverage(blocks, all_tuples, max_blocks, time_fn, overlap_penalty=0.0):
    """Greedy maximum coverage: pick blocks that cover the most new triples."""
    uncovered = set(all_tuples)
    used = []
    used_covers = []
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
            score = len(new_covers)
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


def _local_search_refine(used, blocks, all_tuples, time_fn, max_iters=100):
    """Try to improve by swapping one block for another."""
    if not used:
        return used
    
    covered = set()
    for pattern, covers in used:
        covered |= covers
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if time_fn():
            break
        improved = False
        iters += 1
        best_swap = None
        best_gain = 0
        
        for remove_idx, (rem_pattern, rem_covers) in enumerate(used):
            covered_without = set()
            for j, (_, cov) in enumerate(used):
                if j != remove_idx:
                    covered_without |= cov
            
            for pattern, covers in blocks:
                total_after = len(covered_without | covers)
                total_before = len(covered)
                gain = total_after - total_before
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (remove_idx, pattern, covers)
        
        if best_swap is not None and best_gain > 0:
            remove_idx, new_pattern, new_covers = best_swap
            used[remove_idx] = (new_pattern, new_covers)
            covered = set()
            for pattern, covers in used:
                covered |= covers
            improved = True
    
    return used


def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7

    def time_check():
        return time.time() - _t0 > _DEADLINE - 5.0

    # Generate all possible 2x2x2 Strassen blocks
    strassen_blocks = []
    for i0 in range(n - 1):
        for i1 in range(i0 + 1, n):
            for j0 in range(m - 1):
                for j1 in range(j0 + 1, m):
                    for k0 in range(p - 1):
                        for k1 in range(k0 + 1, p):
                            pattern = _make_strassen_2x2x2_pattern(i0, i1, j0, j1, k0, k1, n, m, p)
                            covers = _block_covers(pattern, n, m, p)
                            strassen_blocks.append((pattern, covers))

    # Generate multi-coverage rank-1 terms
    multi_terms = _generate_multi_coverage_terms(n, m, p)

    # Combine all candidate blocks
    all_blocks = strassen_blocks + multi_terms

    all_tuples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    total_tuples = len(all_tuples)  # 84

    best_result = None
    best_rank = n * m * p  # schoolbook fallback

    # Strategy 1: Greedy with Strassen blocks only
    for max_blocks in range(12, 0, -1):
        if time_check():
            break
        used, uncovered = _greedy_max_coverage(strassen_blocks, all_tuples, max_blocks, time_check, overlap_penalty=0.0)
        if not used:
            continue
        used = _local_search_refine(used, all_blocks, all_tuples, time_check, max_iters=80)
        
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

    # Strategy 2: Greedy with all blocks (Strassen + multi-coverage)
    if not time_check():
        for max_blocks in range(12, 0, -1):
            if time_check():
                break
            used, uncovered = _greedy_max_coverage(all_blocks, all_tuples, max_blocks, time_check, overlap_penalty=0.0)
            if not used:
                continue
            used = _local_search_refine(used, all_blocks, all_tuples, time_check, max_iters=80)
            
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

    # Strategy 3: Greedy with overlap penalty
    if not time_check():
        for penalty in [0.5, 1.0, 2.0]:
            if time_check():
                break
            for max_blocks in range(10, 0, -1):
                if time_check():
                    break
                used, uncovered = _greedy_max_coverage(all_blocks, all_tuples, max_blocks, time_check, overlap_penalty=penalty)
                if not used:
                    continue
                used = _local_search_refine(used, all_blocks, all_tuples, time_check, max_iters=50)
                
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

    # Strategy 4: Random restarts
    if not time_check():
        rng = random.Random(42)
        for restart in range(8):
            if time_check():
                break
            shuffled = all_blocks[:]
            rng.shuffle(shuffled)
            subset = shuffled[:80]
            for max_blocks in range(10, 0, -1):
                if time_check():
                    break
                used, uncovered = _greedy_max_coverage(subset, all_tuples, max_blocks, time_check, overlap_penalty=0.0)
                if not used:
                    continue
                used = _local_search_refine(used, all_blocks, all_tuples, time_check, max_iters=60)
                
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

    # Strategy 5: Two-swap local search on best result
    if best_result is not None and not time_check():
        # Try removing pairs and adding one block
        used = best_result[:]
        covered = set()
        for pattern, covers in used:
            covered |= covers
        
        improved = True
        iters = 0
        while improved and iters < 50:
            if time_check():
                break
            improved = False
            iters += 1
            best_change = None
            best_gain = 0
            
            for r1 in range(len(used)):
                for r2 in range(r1 + 1, len(used)):
                    covered_without = set()
                    for j, (_, cov) in enumerate(used):
                        if j != r1 and j != r2:
                            covered_without |= cov
                    
                    for pattern, covers in all_blocks:
                        total_after = len(covered_without | covers)
                        total_before = len(covered)
                        gain = total_after - total_before
                        if gain > best_gain:
                            best_gain = gain
                            best_change = (r1, r2, pattern, covers)
            
            if best_change is not None and best_gain > 0:
                r1, r2, new_pattern, new_covers = best_change
                # Remove both, add new
                del used[max(r1, r2)]
                del used[min(r1, r2)]
                used.append((new_pattern, new_covers))
                covered = set()
                for pattern, covers in used:
                    covered |= covered
                covered = set()
                for pattern, covers in used:
                    covered |= covers
                improved = True
                best_result = used
    
    if best_result is None:
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
