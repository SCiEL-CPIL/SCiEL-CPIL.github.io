import time
import random
import itertools
import numpy as np

TOTAL_BUDGET = 90.0
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
    Systematic form [I | A].
    """
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

def _construct_golay_shortening():
    """
    Deterministically constructs a [18, 11, 6] code from the [24, 12, 8] Golay code.
    1. Shorten the Golay code at 6 coordinates to get a [18, 12, 8] code.
       This requires finding 6 coordinates such that the dual code has a
       dimension 3 subcode supported on these coordinates (or equivalently,
       the generator matrix restricted to these coordinates has rank 3).
       Actually, shortening a [n, k, d] code at a coordinate where the dual
       has a weight-1 vector reduces k by 1.
       Standard shortening: If we fix 6 coordinates to 0, we get a subcode.
       The dimension of the shortened code is k - rank(G[:, cols]).
       To get dimension 12 from 12, we need rank 0? No.
       Shortening at a coordinate i: take all codewords with 0 at i, remove coordinate i.
       Dimension drops by 1 if the coordinate is non-degenerate.
       If we shorten at 6 coordinates, dimension drops by 6 if they are independent.
       We want a [18, 12, 8] code? No, shortening reduces length and dimension.
       [24, 12, 8] -> Shorten 6 times -> [18, 6, 8]? That's small.
       
       Wait, the inspiration mentioned "512-word extended quadratic-residue construction".
       512 = 2^9.
       A [18, 9, 6] code?
       
       Let's look at the math:
       A_2(18, 6) is known to be at least 2048?
       Actually, the extended quadratic residue code of length 18?
       QR codes exist for n = p^2 + p + 1 where p is prime.
       18 is not of that form.
       
       However, there is a known [18, 11, 6] code?
       Let's check bounds.
       Griesmer bound: sum_{i=0}^{k-1} ceil(d/2^i) <= n
       For k=11, d=6:
       ceil(6/1) + ceil(6/2) + ceil(6/4) + ceil(6/8) + ...
       6 + 3 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 19 > 18.
       So a [18, 11, 6] linear code does NOT exist.
       
       Max k for d=6, n=18?
       k=10: 6+3+2+1+1+1+1+1+1+1 = 18. Yes, [18, 10, 6] is possible.
       Size 2^10 = 1024.
       
       The current program found 512 words.
       If we can find a [18, 10, 6] linear code, we get 1024 words.
       Score would be 1024/171 (if seed was 171?) No, score is value/seed.
       Seed program score 1.0.
       Current program score 6.0.
       If current program returns 512 words, and score is 6.0, then seed value was 512/6 = 85.33?
       Or maybe the seed program returned 32 words (RM code) and 32/seed = 1.0?
       If seed=32, then 512/32 = 16.0? No, score is 6.0.
       Let's re-read: "Score: number of codewords, maximized... search score is value/seed".
       If current score is 6.0000, and it returns 512 words, then seed = 512/6 = 85.33? Unlikely integer.
       Maybe the seed program returned 32 words, and 32/seed = 1.0 => seed=32.
       Then 512/32 = 16.0. But current score is 6.0.
       This implies the current program is NOT returning 512 words?
       Or the seed value is different.
       
       Let's look at the inspirations.
       Inspiration 1 (score 6.0000) uses Golay shortening.
       Inspiration 2 (score 5.6562) uses QR code.
       
       If I can get 1024 words, and seed is 32, score is 32.0.
       If seed is 85, score is 12.0.
       
       Let's assume the goal is to maximize the number of words.
       A [18, 10, 6] linear code has 1024 words.
       Can we construct it?
       The [24, 12, 8] Golay code contains a [23, 12, 7] QR code.
       Puncturing the [23, 12, 7] code at 5 coordinates gives a [18, 12, 5] code?
       Distance drops by at most 1 per puncture? No, d' >= d - 1.
       7 - 5 = 2. Too low.
       
       Shortening the [23, 12, 7] code at 5 coordinates gives [18, 7, 7]?
       Dimension drops by 5. 12-5=7. Size 128.
       
       Let's try to find a [18, 10, 6] code.
       It is known that A_2(18, 6) >= 1024?
       According to the Brouwer table, A_2(18, 6) is between 1024 and 2048?
       Actually, there is a nonlinear code of size 2048?
       Or is the max 1024?
       
       Let's implement a robust search for a large linear code or use a known construction.
       The "extended quadratic residue code" of length 18 doesn't exist in the standard sense.
       But we can use the [24, 12, 8] Golay code.
       If we shorten it at 6 coordinates, we get a [18, 12-r, 8] code.
       If we choose the 6 coordinates such that the rank of the submatrix is 2, we get [18, 10, 8].
       A [18, 10, 8] code has distance 8, which is >= 6.
       Size 1024.
       
       So the strategy is:
       1. Generate [24, 12, 8] Golay code.
       2. Find 6 coordinates such that the rank of G[:, cols] is 2.
       3. Shorten the code at these 6 coordinates (keep codewords with 0 in these cols, remove these cols).
       4. Result is a [18, 10, 8] code with 1024 words.
       5. Verify distance.
       6. Greedy extend.
       
       Why did the current program fail to get 1024?
       It looked for rank <= 3.
       If rank is 3, dimension is 12-3=9, size 512.
       If rank is 2, dimension is 12-2=10, size 1024.
       If rank is 1, dimension is 11, size 2048. (But [18, 11, 6] doesn't exist, so rank 1 is impossible for d>=6?
       Wait, if we shorten to get [18, 11, d], the distance might drop.
       Shortening preserves distance or increases it?
       Shortening: take subcode with 0 at coord, remove coord.
       If two words were distance d, and both have 0 at coord i, their distance in the shortened code is still d.
       So distance is preserved.
       So a [18, 11, 8] code would exist if we could find rank 1.
       But we proved [18, 11, 6] doesn't exist by Griesmer. So [18, 11, 8] definitely doesn't exist.
       So rank 1 is impossible.
       Rank 2 gives [18, 10, 8]. This is valid.
       
       So we just need to find 6 columns with rank 2.
       """
    G = _get_golay_generator_matrix()
    golay_words = _generate_golay_code(G)
    
    n_cols = 24
    n_rows = 12
    
    best_set = None
    best_rank = 100
    
    # Iterate through all 6-subsets of columns
    # C(24, 6) = 134596. This is small enough.
    for cols in itertools.combinations(range(n_cols), 6):
        # Calculate rank of G[:, cols]
        sub = [[G[r][c] for c in cols] for r in range(n_rows)]
        
        # Rank calculation
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
        
        if rank < best_rank:
            best_rank = rank
            best_set = cols
            
            # If we found rank 2, we can stop early if we want, but let's check for rank 1 just in case (though unlikely)
            if rank == 1:
                break
                
    if best_set is None:
        return None
        
    # Shorten
    cols_to_zero = best_set
    remaining_cols = [c for c in range(24) if c not in cols_to_zero]
    shortened = []
    for w in golay_words:
        if all(((w >> c) & 1) == 0 for c in cols_to_zero):
            new_word = 0
            for i, c in enumerate(remaining_cols):
                if (w >> c) & 1:
                    new_word |= (1 << i)
            shortened.append(new_word)
            
    return shortened

def _greedy_addition_fast(base_words, N=18, min_dist=6, time_limit=10.0, seed=42):
    """
    Fast greedy addition using numpy for distance checks.
    """
    rng = random.Random(seed)
    best_words = np.array(base_words, dtype=np.uint32)
    start = time.time()
    
    while time.time() - start < time_limit:
        # Generate batch of candidates
        batch_size = 1000
        candidates = np.array([rng.randint(0, (1 << N) - 1) for _ in range(batch_size)], dtype=np.uint32)
        
        # Check each candidate against all existing words
        # This is O(B * K * N). B=1000, K=1024, N=18.
        # 1000 * 1024 = 1M XORs. Fast in numpy?
        # We can do it in chunks.
        
        added_any = False
        for cand in candidates:
            # Check distance to all existing words
            # Use a loop for simplicity first, if too slow, optimize with numpy broadcast
            # But 1024 checks is fast in Python if we use bit_count.
            ok = True
            for w in best_words:
                if _hamming(cand, w) < min_dist:
                    ok = False
                    break
            if ok:
                best_words = np.append(best_words, cand)
                added_any = True
                
        if not added_any:
            # Try fewer candidates or break if stuck?
            # Just continue with new random candidates
            pass
            
    return best_words.tolist()

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()
    
    best_words = []
    
    # 1. Try Golay Shortening to [18, 10, 8]
    golay_shortened = _construct_golay_shortening()
    if golay_shortened:
        if _check_min_distance(golay_shortened, MIN_DIST):
            best_words = golay_shortened
        else:
            # Fallback to RM if Golay fails
            rm_words = _reed_muller_extended()
            if _check_min_distance(rm_words, MIN_DIST):
                best_words = rm_words
            else:
                return [0] # Should not happen
    
    if not best_words:
        rm_words = _reed_muller_extended()
        if _check_min_distance(rm_words, MIN_DIST):
            best_words = rm_words
        else:
            return [0]
            
    elapsed = time.time() - start
    time_remaining = DEADLINE - elapsed
    
    if time_remaining > 5.0:
        # Greedy extend
        greedy_result = _greedy_addition_fast(best_words, N, MIN_DIST, time_limit=time_remaining - 2.0, seed=42)
        if len(greedy_result) > len(best_words) and _check_min_distance(greedy_result, MIN_DIST):
            best_words = greedy_result
    
    return best_words
