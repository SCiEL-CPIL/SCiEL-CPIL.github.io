import time
import random
import itertools
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 5.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _popcount(x):
    return bin(x).count("1")

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _get_golay_generator_matrix():
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    return G

def _generate_golay_code(G):
    codewords = []
    n_rows = len(G)
    n_cols = len(G[0])
    for mask in range(1 << n_rows):
        word = 0
        for i in range(n_rows):
            if (mask >> i) & 1:
                for j in range(n_cols):
                    if G[i][j]:
                        word ^= (1 << j)
        codewords.append(word)
    return codewords

def _find_shortening_set(G, codewords, target_size=512):
    n_cols = 24
    n_rows = 12
    best_set = None
    best_count = 0

    for cols in itertools.combinations(range(n_cols), 6):
        sub = [[G[r][c] for c in cols] for r in range(n_rows)]

        matrix = [row[:] for row in sub]
        rank = 0
        for c in range(6):
            pivot = -1
            for r in range(rank, 12):
                if matrix[r][c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            for r in range(12):
                if r != rank and matrix[r][c] == 1:
                    for cc in range(c, 6):
                        matrix[r][cc] ^= matrix[rank][cc]
            rank += 1
            if rank == 6:
                break

        if rank <= 3:
            count = 0
            for w in codewords:
                ok = True
                for c in cols:
                    if (w >> c) & 1:
                        ok = False
                        break
                if ok:
                    count += 1
            if count > best_count:
                best_count = count
                best_set = cols

    return best_set, best_count

def _shorten_code(codewords, cols_to_zero):
    remaining_cols = [c for c in range(24) if c not in cols_to_zero]
    shortened = []
    for w in codewords:
        if all(((w >> c) & 1) == 0 for c in cols_to_zero):
            new_word = 0
            for i, c in enumerate(remaining_cols):
                if (w >> c) & 1:
                    new_word |= (1 << i)
            shortened.append(new_word)
    return shortened

def _greedy_addition(base_words, N=18, min_dist=6, time_limit=10.0, seed=42):
    rng = random.Random(seed)
    best_words = list(base_words)
    start = time.time()
    while time.time() - start < time_limit:
        attempts = 0
        while attempts < 1000 and time.time() - start < time_limit:
            cand = rng.randint(0, (1 << N) - 1)
            ok = True
            for w in best_words:
                if _hamming(cand, w) < min_dist:
                    ok = False
                    break
            if ok:
                best_words.append(cand)
            attempts += 1
    return best_words

def _local_search(words, N=18, min_dist=6, time_limit=5.0, seed=123):
    rng = random.Random(seed)
    best = list(words)
    start = time.time()
    while time.time() - start < time_limit:
        if time.time() - start >= time_limit:
            break
        cand = rng.randint(0, (1 << N) - 1)
        ok = True
        for w in best:
            if _hamming(cand, w) < min_dist:
                ok = False
                break
        if ok:
            best.append(cand)
    return best

def _build_extended_quadratic_residue_code():
    """
    Build the extended quadratic residue code of length 18 (padded from 16)
    by using the structure of QR codes.
    """
    # Use the Reed-Muller(2,4) code as base (32 words, length 16)
    # Then extend by 2 bits in a way that maintains distance
    base = _reed_muller_extended()
    # For each base word, try all 4 extensions (2 extra bits)
    # and keep those that maintain min distance 6
    extended = []
    for w in base:
        for ext in range(4):
            new_w = (w << 2) | ext
            # Check distance to all existing extended words
            ok = True
            for ew in extended:
                if _hamming(new_w, ew) < 6:
                    ok = False
                    break
            if ok:
                extended.append(new_w)
    return extended

def _try_multiple_shortenings(G, golay_words, min_dist=6):
    """Try multiple shortening strategies and return the best code."""
    n_cols = 24
    n_rows = 12
    best_code = []
    best_size = 0

    # Try all combinations of 6 columns to zero out
    # Use rank condition to filter quickly
    for cols in itertools.combinations(range(n_cols), 6):
        sub = [[G[r][c] for c in cols] for r in range(n_rows)]

        matrix = [row[:] for row in sub]
        rank = 0
        for c in range(6):
            pivot = -1
            for r in range(rank, 12):
                if matrix[r][c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            for r in range(12):
                if r != rank and matrix[r][c] == 1:
                    for cc in range(c, 6):
                        matrix[r][cc] ^= matrix[rank][cc]
            rank += 1
            if rank == 6:
                break

        if rank <= 3:
            code = _shorten_code(golay_words, cols)
            if len(code) > best_size and _check_min_distance(code, min_dist):
                best_size = len(code)
                best_code = code

    return best_code

def _bron_kerbosch_mis(candidates, selected, min_dist, N, time_start, time_limit):
    """
    Bron-Kerbosch variant for maximum independent set in complement of Hamming graph.
    candidates: list of candidate codewords
    selected: list of selected codewords
    Returns the largest independent set found.
    """
    if time.time() - time_start > time_limit:
        return list(selected)

    if not candidates:
        return list(selected)

    # Pick a pivot to reduce branching
    pivot = candidates[0]

    # Split candidates into those connected to pivot and those not
    # In complement of Hamming graph, two nodes are connected if they violate min_dist
    # So "connected to pivot" means distance < min_dist
    connected = []
    not_connected = []
    for c in candidates:
        if _hamming(c, pivot) < min_dist:
            connected.append(c)
        else:
            not_connected.append(c)

    best_result = list(selected)

    # Try adding each candidate not connected to pivot
    for i, cand in enumerate(not_connected):
        if time.time() - time_start > time_limit:
            break

        new_selected = selected + [cand]
        # Remaining candidates must be at min_dist from cand AND from all in new_selected
        new_candidates = []
        for j in range(i + 1, len(candidates)):
            other = candidates[j]
            if other == cand:
                continue
            if _hamming(other, cand) >= min_dist:
                # Also check against all in new_selected
                ok = True
                for s in new_selected:
                    if _hamming(other, s) < min_dist:
                        ok = False
                        break
                if ok:
                    new_candidates.append(other)

        result = _bron_kerbosch_mis(new_candidates, new_selected, min_dist, N, time_start, time_limit)
        if len(result) > len(best_result):
            best_result = result

    return best_result

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # 1. Try Golay Shortening - this is the primary approach
    G = _get_golay_generator_matrix()
    golay_words = _generate_golay_code(G)

    # First try the known good shortening
    shortening_set, count = _find_shortening_set(G, golay_words, target_size=512)

    if shortening_set is not None and count >= 512:
        code = _shorten_code(golay_words, shortening_set)
        if len(code) >= 512 and _check_min_distance(code, MIN_DIST):
            return code

    # 2. Try multiple shortenings to find the best
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed

    if time_remaining > 30.0:
        best_code = _try_multiple_shortenings(G, golay_words, MIN_DIST)
        if len(best_code) > 512 and _check_min_distance(best_code, MIN_DIST):
            return best_code

    # 3. Fallback: Reed-Muller + Greedy
    rm_words = _reed_muller_extended()
    if not _check_min_distance(rm_words, MIN_DIST):
        raise ValueError("RM code failed distance check")

    best = list(rm_words)

    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    if time_remaining > 5.0:
        greedy_result = _greedy_addition(best, N, MIN_DIST, time_limit=time_remaining - 2.0, seed=42)
        if len(greedy_result) > len(best) and _check_min_distance(greedy_result, MIN_DIST):
            best = greedy_result

    # 4. Try QR code extension
    if time_remaining > 10.0:
        qr_code = _build_extended_quadratic_residue_code()
        if len(qr_code) > len(best) and _check_min_distance(qr_code, MIN_DIST):
            best = qr_code

    return best
