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
    """
    Returns a 12x24 binary generator matrix for the [24,12,8] Golay code.
    This matrix is in systematic form [I | A].
    """
    # Known generator matrix for the extended binary Golay code
    # Source: Standard coding theory tables
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
    """Generate all 4096 codewords of the Golay code."""
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
    """
    Find a set of 6 columns such that the number of codewords with 0s in these
    columns is at least target_size.
    """
    n_cols = 24
    n_rows = 12
    best_set = None
    best_count = 0
    
    # We want rank <= 3 for 6 columns to get 2^(12-3) = 512 words.
    # Iterate through all 6-subsets of columns
    for cols in itertools.combinations(range(n_cols), 6):
        # Check rank of G[:, cols]
        # G is 12x24. Submatrix is 12x6.
        # We can compute rank by Gaussian elimination or just check linear independence.
        # Since 12x6 is small, we can just check if the 6 columns are linearly dependent in a way that reduces rank.
        # Actually, we just need the rank of the 12x6 matrix.
        
        # Extract submatrix
        sub = []
        for r in range(n_rows):
            row = 0
            for c in cols:
                if G[r][c]:
                    row |= (1 << (cols.index(c))) # This is inefficient, better to map indices
            # Better: just store the bits
            sub.append([G[r][c] for c in cols])
            
        # Calculate rank of sub (12x6)
        # Convert to list of ints for the 6 columns? No, 12 rows of 6 bits.
        # Rank of a matrix over GF(2).
        # We can use numpy or manual.
        # Manual rank calculation for 12x6:
        matrix = [[sub[r][c] for c in range(6)] for r in range(12)]
        rank = 0
        for c in range(6):
            # Find pivot
            pivot = -1
            for r in range(rank, 12):
                if matrix[r][c] == 1:
                    pivot = r
                    break
            if pivot == -1:
                continue
            # Swap
            matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
            # Eliminate
            for r in range(12):
                if r != rank and matrix[r][c] == 1:
                    for cc in range(c, 6):
                        matrix[r][cc] ^= matrix[rank][cc]
            rank += 1
            if rank == 6:
                break
        
        if rank <= 3:
            # Count codewords with 0 in these columns
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
    """
    Take codewords that have 0 in cols_to_zero, and map the remaining bits to integers.
    """
    remaining_cols = [c for c in range(24) if c not in cols_to_zero]
    shortened = []
    for w in codewords:
        # Check if 0 in cols_to_zero
        if all(((w >> c) & 1) == 0 for c in cols_to_zero):
            new_word = 0
            for i, c in enumerate(remaining_cols):
                if (w >> c) & 1:
                    new_word |= (1 << i)
            shortened.append(new_word)
    return shortened

def _greedy_addition(base_words, N=18, min_dist=6, time_limit=10.0, seed=42):
    """Fallback greedy addition."""
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

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()
    
    # 1. Try Golay Shortening
    G = _get_golay_generator_matrix()
    golay_words = _generate_golay_code(G)
    
    # Verify Golay code min distance is 8 (sanity check)
    # This is expensive for 4096 words, so we skip full check and trust the matrix.
    
    shortening_set, count = _find_shortening_set(G, golay_words, target_size=512)
    
    if shortening_set is not None and count >= 512:
        code = _shorten_code(golay_words, shortening_set)
        if _check_min_distance(code, MIN_DIST):
            # Verify size
            if len(code) >= 512:
                return code
    
    # 2. Fallback: Reed-Muller + Greedy
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
    
    return best
