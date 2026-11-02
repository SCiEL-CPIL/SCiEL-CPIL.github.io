"""Bit-Parallel Iterative Compression for the ternary covering code K_3(7,1)."""
import time
import random
import itertools
import numpy as np

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to keep before the hard 150s kill


def _now():
    return time.time()


def _remaining():
    return 150.0 - _now() - _DEADLINE_MARGIN


def _to_idx(word, n):
    idx = 0
    for v in word:
        idx = idx * 3 + v
    return idx


def _from_idx(idx, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % 3
        idx //= 3
    return word


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_code(q, n, radius):
    # Direct sum of tetracode (length 4) with all tails
    tetracode = _tetracode()
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(h + list(tail))
    return code


def _ball_indices(word, q, n, radius):
    """Return a sorted list of indices in the ball of given radius around word."""
    if radius == 1:
        idx = _to_idx(word, n)
        res = [idx]
        for pos in range(n):
            for d in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + d) % q
                res.append(_to_idx(new_word, n))
        return res
    # General radius (not used for this task but safe)
    from itertools import product
    res = []
    for delta in product(range(q), repeat=n):
        dist = sum(1 for a, b in zip(word, delta) if a != b)
        if dist <= radius:
            new_word = [(word[i] + delta[i]) % q for i in range(n)]
            res.append(_to_idx(new_word, n))
    return res


def _build_coverage(code, q, n, radius):
    """Build a coverage bitset (numpy bool array) and per-codeword ball bitsets."""
    total = q ** n
    coverage = np.zeros(total, dtype=bool)
    balls = []
    for cw in code:
        ball = _ball_indices(cw, q, n, radius)
        bmask = np.zeros(total, dtype=bool)
        bmask[ball] = True
        balls.append(bmask)
        coverage |= bmask
    return coverage, balls


def _is_covered(coverage, q, n):
    return coverage.all()


def _try_merge_pair(code, coverage, balls, i, j, q, n, radius):
    """
    Try to replace codewords i and j with a single new codeword.
    The new codeword must cover all words that were ONLY covered by i or j (not by others).
    We search for a word w such that ball(w) covers (ball_i ∪ ball_j) \ (coverage_without_i_and_j).
    """
    # Words that would become uncovered if we remove both i and j
    # uncovered = (ball_i ∪ ball_j) & ~(coverage_without_i_and_j)
    # coverage_without_i_and_j = coverage & ~(ball_i | ball_j) ... no that's not right
    # Actually: coverage_without = coverage with i and j removed
    # Since we don't have that stored, compute: words in (ball_i ∪ ball_j) that are NOT covered by any other codeword
    
    bi = balls[i]
    bj = balls[j]
    union_ij = bi | bj
    
    # Find words in union_ij that are covered ONLY by i or j
    # A word x in union_ij is uniquely covered by {i,j} if coverage[x] is true but
    # removing i and j makes it false.
    # Equivalent: x is in union_ij AND x is not in (coverage & ~union_ij)
    # i.e., x is not covered by any codeword other than i and j.
    
    others_coverage = coverage & ~union_ij
    must_cover = union_ij & ~others_coverage
    
    # Now find a word w such that ball(w) ⊇ must_cover
    # For radius 1, ball(w) has 1 + n*(q-1) words.
    # must_cover must be a subset of some radius-1 ball.
    
    # Heuristic: if must_cover has more elements than ball size, impossible
    ball_size = 1 + n * (q - 1)
    must_cover_indices = np.where(must_cover)[0]
    if len(must_cover_indices) > ball_size:
        return None
    
    # Try to find a center w for a ball covering must_cover
    # For radius 1: w must be within distance 1 of every word in must_cover
    # So w is in the intersection of all balls of radius 1 around each word in must_cover
    # But that's expensive. Instead, try candidate centers:
    # - The codewords themselves (i and j)
    # - Words that are "between" i and j
    
    candidates = set()
    # Try i and j themselves
    candidates.add(tuple(code[i]))
    candidates.add(tuple(code[j]))
    
    # Try words that differ from i in one position and from j in one position
    # More systematically: for each position, try all q values
    # A valid center w must satisfy: for every x in must_cover, dist(w,x) <= 1
    # This means w and x differ in at most 1 position.
    # So w is determined by: for each position where any x in must_cover differs from w,
    # at most one such position can have variation.
    
    # Simpler: iterate over all words that are in the intersection of 1-balls around must_cover words
    # For small must_cover, we can compute this.
    
    if len(must_cover_indices) == 0:
        # Both balls are fully covered by others, just pick any word
        return list(code[i])
    
    # Get the actual words
    must_words = [_from_idx(int(idx), n) for idx in must_cover_indices]
    
    # The center w must be within distance 1 of all must_words
    # Start with w = must_words[0], then for each other word, constrain w
    # w can differ from must_words[0] in at most 1 position, and from each other in at most 1.
    
    # Enumerate candidates: start from must_words[0], change each position to each value
    base = must_words[0]
    candidates.add(tuple(base))
    for pos in range(n):
        for val in range(q):
            if val != base[pos]:
                cand = base[:]
                cand[pos] = val
                candidates.add(tuple(cand))
    
    # Also try from the second word if exists
    if len(must_words) >= 2:
        base2 = must_words[1]
        candidates.add(tuple(base2))
        for pos in range(n):
            for val in range(q):
                if val != base2[pos]:
                    cand = base2[:]
                    cand[pos] = val
                    candidates.add(tuple(cand))
    
    # Check each candidate
    for cand in candidates:
        # Verify cand covers all must_words
        ok = True
        for mw in must_words:
            dist = sum(1 for a, b in zip(cand, mw) if a != b)
            if dist > 1:
                ok = False
                break
        if ok:
            # Also verify that cand's ball doesn't cause issues (it won't, we're adding coverage)
            # But we need to make sure cand is not already in the code (or it's fine if it is,
            # but then we'd have duplicates - avoid that)
            cand_list = list(cand)
            if cand_list not in code:
                return cand_list
    
    return None


def _greedy_compression(code, q, n, radius, time_limit):
    """
    Iteratively try to compress the code by merging pairs.
    Returns the compressed code.
    """
    if not _is_covered(*_build_coverage(code, q, n, radius)[:1], q, n):
        pass
    
    coverage, balls = _build_coverage(code, q, n, radius)
    code = list(code)
    balls = list(balls)
    
    improved = True
    while improved and _remaining() > 0:
        improved = False
        m = len(code)
        
        # Shuffle pair order for diversity
        pairs = list(itertools.combinations(range(m), 2))
        random.shuffle(pairs)
        
        for i, j in pairs:
            if _remaining() < 1.0:
                break
            
            new_cw = _try_merge_pair(code, coverage, balls, i, j, q, n, radius)
            if new_cw is not None:
                # Remove i and j, add new_cw
                # Remove the larger index first to keep indices valid
                if i > j:
                    i, j = j, i
                # Remove j
                del code[j]
                del balls[j]
                # Remove i (now i <= j, and if i==j originally we already handled)
                if i != j:
                    del code[i]
                    del balls[i]
                else:
                    # i and j were the same? No, they're different
                    pass
                
                # Add new codeword
                code.append(new_cw)
                new_ball = _ball_indices(new_cw, q, n, radius)
                bmask = np.zeros(q ** n, dtype=bool)
                bmask[new_ball] = True
                balls.append(bmask)
                
                # Update coverage
                coverage = np.zeros(q ** n, dtype=bool)
                for b in balls:
                    coverage |= b
                
                improved = True
                break  # Restart pair iteration with new code
    
    return code


def _local_search(code, q, n, radius, time_limit):
    """
    Try to improve the code by replacing single codewords with better ones,
    or by moving codewords to reduce redundancy.
    """
    coverage, balls = _build_coverage(code, q, n, radius)
    code = list(code)
    balls = list(balls)
    total = q ** n
    
    while _remaining() > 2.0:
        m = len(code)
        random_idx = random.randint(0, m - 1)
        
        # Try to find a better position for codeword random_idx
        # A codeword is "redundant" if its ball is fully covered by others
        others = coverage & ~balls[random_idx]
        if not others.all():
            # This codeword covers some unique words, can't simply move it
            # Try to find a new position that covers the unique words
            unique = balls[random_idx] & ~others
            unique_indices = np.where(unique)[0]
            if len(unique_indices) > 1 + n * (q - 1):
                continue  # Can't cover with a single ball
            
            must_words = [_from_idx(int(idx), n) for idx in unique_indices]
            # Find a center covering all must_words (same logic as merge)
            base = must_words[0]
            found = False
            for pos in range(n):
                for val in range(q):
                    cand = base[:]
                    cand[pos] = val
                    ok = all(sum(1 for a, b in zip(cand, mw) if a != b) <= 1 for mw in must_words)
                    if ok and cand not in code:
                        # Move codeword
                        code[random_idx] = cand
                        new_ball = _ball_indices(cand, q, n, radius)
                        bmask = np.zeros(total, dtype=bool)
                        bmask[new_ball] = True
                        balls[random_idx] = bmask
                        coverage = np.zeros(total, dtype=bool)
                        for b in balls:
                            coverage |= b
                        found = True
                        break
                if found:
                    break
    
    return code


def construct(q, n, radius):
    start = _now()
    
    # Initial code: direct sum of tetracode
    code = _initial_code(q, n, radius)
    
    # Verify initial coverage
    coverage, balls = _build_coverage(code, q, n, radius)
    if not coverage.all():
        # Shouldn't happen for the tetracode construction, but be safe
        # Add more codewords until covered
        all_words = list(itertools.product(range(q), repeat=n))
        for w in all_words:
            if not coverage[_to_idx(w, n)]:
                code.append(list(w))
                ball = _ball_indices(w, q, n, radius)
                bmask = np.zeros(q ** n, dtype=bool)
                bmask[ball] = True
                coverage |= bmask
                balls.append(bmask)
    
    # Phase 1: Greedy compression
    if _remaining() > 5.0:
        code = _greedy_compression(code, q, n, radius, 60.0)
    
    # Phase 2: Local search
    if _remaining() > 3.0:
        code = _local_search(code, q, n, radius, 30.0)
    
    # Final verification
    coverage, _ = _build_coverage(code, q, n, radius)
    if not coverage.all():
        # Fallback to initial code if somehow broken
        code = _initial_code(q, n, radius)
    
    return code
