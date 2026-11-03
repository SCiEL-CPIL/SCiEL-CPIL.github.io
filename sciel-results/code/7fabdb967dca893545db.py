# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_terms(i_off, j_off, k_off):
    """
    Return the 7 Strassen terms for a 2x2x2 block.
    """
    terms = []
    U1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
    V1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
    W1 = {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
    terms.append((U1, V1, W1))

    U2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
    V2 = {(j_off, k_off): 1}
    W2 = {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
    terms.append((U2, V2, W2))

    U3 = {(i_off, j_off): 1}
    V3 = {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}
    W3 = {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
    terms.append((U3, V3, W3))

    U4 = {(i_off+1, j_off+1): 1}
    V4 = {(j_off+1, k_off): 1, (j_off, k_off): -1}
    W4 = {(k_off, i_off+1): 1, (k_off, i_off): 1}
    terms.append((U4, V4, W4))

    U5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
    V5 = {(j_off+1, k_off+1): 1}
    W5 = {(k_off+1, i_off): 1, (k_off, i_off): -1}
    terms.append((U5, V5, W5))

    U6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
    V6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
    W6 = {(k_off+1, i_off+1): 1}
    terms.append((U6, V6, W6))

    U7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
    V7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
    W7 = {(k_off, i_off): 1}
    terms.append((U7, V7, W7))

    return terms


def _build_and_verify(blocks):
    """Build decomposition from Strassen blocks + schoolbook for uncovered, verify."""
    n, m, p = N, M, P

    strassen_covered = set()
    for i_off, j_off, k_off in blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    strassen_covered.add((i_off + ii, j_off + jj, k_off + kk))

    all_terms = []
    for i_off, j_off, k_off in blocks:
        block_terms = _strassen_terms(i_off, j_off, k_off)
        for U_dict, V_dict, W_dict in block_terms:
            all_terms.append((U_dict, V_dict, W_dict))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    all_terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))

    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff

    if _verify(U, V, W):
        return U, V, W, rank
    return None


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Searches for optimal Strassen block packing to minimize rank.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = N, M, P

    # Known good: 4 disjoint blocks, rank 56
    known_good = [(0, 0, 0), (0, 2, 2), (1, 0, 3), (1, 2, 0)]
    best_result = _build_and_verify(known_good)
    best_rank = best_result[3] if best_result else RANK_SCHOOLBOOK

    # Generate all valid blocks
    all_blocks = []
    for i_off in range(n - 1):
        for j_off in range(m - 1):
            for k_off in range(p - 1):
                all_blocks.append((i_off, j_off, k_off))

    # Try sets of 5, 6, 7 blocks (search for better packings)
    for target_size in [7, 6, 5]:
        if time.time() > deadline:
            break
        min_possible_rank = 7 * target_size + (n * m * p - 8 * target_size)
        if min_possible_rank >= best_rank:
            continue
        for combo in combinations(range(len(all_blocks)), target_size):
            if time.time() > deadline:
                break
            blocks = [all_blocks[i] for i in combo]
            # Check disjointness
            covered = set()
            disjoint = True
            for b in blocks:
                i_off, j_off, k_off = b
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            cell = (i_off + ii, j_off + jj, k_off + kk)
                            if cell in covered:
                                disjoint = False
                                break
                            covered.add(cell)
                        if not disjoint:
                            break
                    if not disjoint:
                        break
                if not disjoint:
                    break
            if not disjoint:
                continue
            result = _build_and_verify(blocks)
            if result is not None and result[3] < best_rank:
                best_rank = result[3]
                best_result = result

    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}

    return _schoolbook()
# EVOLVE-BLOCK-END
