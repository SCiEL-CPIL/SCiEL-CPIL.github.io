"""Construct a covering code for the football pool problem K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

# Known small perfect / covering codes
# Ternary tetracode: 9 words length 4, radius 1 (perfect)
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

# Binary repetition code length 3: 2 words {000,111}, radius 1 (perfect for q=2,n=3,r=1)
_BIN_REP3 = [[0, 0, 0], [1, 1, 1]]

# Binary Hamming code length 7: 16 words, radius 1 (perfect for q=2,n=7,r=1)
def _bin_hamming7():
    cols = []
    for i in range(1, 8):
        cols.append([i & 1, (i >> 1) & 1, (i >> 2) & 1])
    G = [
        [1,0,0,0, 1,0,0],
        [0,1,0,0, 0,1,0],
        [0,0,1,0, 0,0,1],
        [0,0,0,1, 1,1,1],
    ]
    words = []
    for mask in range(16):
        v = [0]*7
        for i in range(4):
            if mask & (1 << i):
                for j in range(7):
                    v[j] ^= G[i][j]
        words.append(v)
    return words
_BIN_HAMMING7 = _bin_hamming7()

def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def _is_covered(word, code, q, n, radius):
    for c in code:
        if _hamming_dist(word, c) <= radius:
            return True
    return False

def _ball_indices(c, q, n, radius):
    """Return list of indices (base-q) of words within `radius` of codeword c."""
    if radius == 1:
        # c itself
        idx = 0
        for v in c:
            idx = idx * q + v
        result = [idx]
        for pos in range(n):
            orig = c[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                c2 = c[:]
                c2[pos] = new_val
                idx2 = 0
                for v in c2:
                    idx2 = idx2 * q + v
                result.append(idx2)
        return result
    else:
        # General radius: iterate over all words in ball (expensive, but only for small spaces)
        result = []
        for w in itertools.product(range(q), repeat=n):
            if _hamming_dist(list(w), c) <= radius:
                idx = 0
                for v in w:
                    idx = idx * q + v
                result.append(idx)
        return result

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Handle the specific known case: q=3, n=6, radius=1
    # Best known: 73 words. Direct product gives 9 * 3^2 = 81.
    
    # Initial code: direct product of tetracode and full space on remaining coords
    if q == 3 and n >= 4:
        rem = n - 4
        code = [h + list(tail) for h in _TETRACODE for tail in itertools.product(range(3), repeat=rem)]
    elif q == 2 and n >= 3:
        if n == 3:
            code = _BIN_REP3[:]
        elif n == 7:
            code = [list(w) for w in _BIN_HAMMING7]
        else:
            rem = n - 3
            code = [h + list(tail) for h in _BIN_REP3 for tail in itertools.product(range(2), repeat=rem)]
    else:
        # Fallback: start with a single codeword and greedy cover
        code = [[0] * n]
    
    total_words = q ** n
    if total_words > 200000:
        return code  # too large for coverage-based pruning
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w
    
    # Precompute ball indices for each codeword
    def build_balls(codewords):
        balls = []
        for c in codewords:
            balls.append(_ball_indices(c, q, n, radius))
        return balls
    
    # Build coverage count: for each word in the space, how many codewords cover it
    def build_coverage_count(balls):
        counts = [0] * total_words
        for ball in balls:
            for idx in ball:
                counts[idx] += 1
        return counts
    
    # Check if removing a codeword keeps coverage valid
    def is_redundant(ball, counts):
        for idx in ball:
            if counts[idx] <= 1:
                return False
        return True
    
    # Iterative greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        balls = build_balls(code)
        counts = build_coverage_count(balls)
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(balls[i], counts):
                code.pop(i)
                improved = True
                break  # Restart the scan from the end
    
    # Local search: try swapping codewords to reduce count further
    # A swap: remove one codeword, add another (possibly different) to maintain coverage
    # We try: for each codeword, try removing it and adding a random candidate
    # If the new code is valid, accept it.
    
    def is_valid_code(codewords):
        """Check if codewords cover the entire space."""
        if not codewords:
            return False
        counts = [0] * total_words
        for c in codewords:
            for idx in _ball_indices(c, q, n, radius):
                counts[idx] += 1
        for cnt in counts:
            if cnt == 0:
                return False
        return True
    
    # Try to improve by swapping: remove one, add one
    # We'll do a limited number of iterations
    max_swap_iters = 500
    swap_iter = 0
    best_code = code[:]
    best_len = len(code)
    
    while swap_iter < max_swap_iters and time.time() < deadline:
        swap_iter += 1
        # Pick a random codeword to remove
        if len(code) <= 1:
            break
        i = random.randint(0, len(code) - 1)
        removed = code[i]
        code.pop(i)
        
        # Now try to find a codeword to add that maintains coverage
        # We need to cover the words that were only covered by `removed`
        # Find uncovered words
        counts = [0] * total_words
        for c in code:
            for idx in _ball_indices(c, q, n, radius):
                counts[idx] += 1
        
        uncovered = [idx for idx in range(total_words) if counts[idx] == 0]
        
        if not uncovered:
            # We removed a redundant codeword! Accept.
            best_code = code[:]
            best_len = len(code)
            # Continue trying to remove more
            continue
        else:
            # Try to add a codeword that covers as many uncovered words as possible
            # Candidate: one of the uncovered words, or a random word
            # Try the first uncovered word
            candidate = idx_to_word(uncovered[0])
            
            # Check if adding candidate covers all uncovered
            cand_ball = set(_ball_indices(candidate, q, n, radius))
            covers_all = all(idx in cand_ball for idx in uncovered)
            
            if covers_all:
                code.append(candidate)
                best_code = code[:]
                best_len = len(code)
            else:
                # Try a few random candidates
                for _ in range(10):
                    rand_idx = random.randint(0, total_words - 1)
                    candidate = idx_to_word(rand_idx)
                    cand_ball = set(_ball_indices(candidate, q, n, radius))
                    covers_all = all(idx in cand_ball for idx in uncovered)
                    if covers_all:
                        code.append(candidate)
                        best_code = code[:]
                        best_len = len(code)
                        break
                else:
                    # Reject: put removed back
                    code.append(removed)
    
    # Return the best code found
    return best_code
# EVOLVE-BLOCK-END
