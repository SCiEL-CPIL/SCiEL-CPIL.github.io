"""Construct a covering code for the football pool problem K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

# Known small perfect / covering codes
# Ternary tetracode: 9 words length 4, radius 1 (perfect)
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

# Binary repetition code length 3: 2 words {000,111}, radius 1 (perfect for q=2,n=3,r=1)
_BIN_REP3 = [[0, 0, 0], [1, 1, 1]]

# Binary Hamming code length 7: 16 words, radius 1 (perfect for q=2,n=7,r=1)
def _bin_hamming7():
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

def _ball_words(center, q, n, radius=1):
    """Generate all words within Hamming distance <= radius from center."""
    if radius == 1:
        yield center
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w = center[:]
                w[pos] = new_val
                yield w
    else:
        # General case: enumerate all words in the ball
        def _gen(pos, dist_left, current):
            if pos == n:
                yield current[:]
                return
            if dist_left == 0:
                current[pos] = center[pos]
                yield from _gen(pos + 1, 0, current)
                return
            # Keep same value
            current[pos] = center[pos]
            yield from _gen(pos + 1, dist_left, current)
            # Change to other values
            for v in range(q):
                if v == center[pos]:
                    continue
                current[pos] = v
                yield from _gen(pos + 1, dist_left - 1, current)
        yield from _gen(0, radius, [0]*n)

def _compute_coverage(code, q, n, radius):
    """For each word in the space, compute how many codewords cover it."""
    total = q ** n
    coverage = np.zeros(total, dtype=np.int32)
    
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    for c in code:
        for w in _ball_words(c, q, n, radius):
            coverage[word_to_int(w)] += 1
    return coverage

def _try_remove(code, idx, q, n, radius, coverage):
    """Try removing code[idx]. Returns True if code remains valid."""
    c = code[idx]
    # Check all words in the ball of c: if any has coverage == 1, removing c breaks coverage
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    for w in _ball_words(c, q, n, radius):
        if coverage[word_to_int(w)] <= 1:
            return False
    return True

def _try_swap(code, idx1, idx2, q, n, radius, coverage):
    """Try swapping code[idx1] and code[idx2]. Returns True if valid and size doesn't increase."""
    # For swap to be valid, we need:
    # 1. All words in ball(code[idx1]) are covered by code without code[idx1] OR by code[idx2]
    # 2. All words in ball(code[idx2]) are covered by code without code[idx2] OR by code[idx1]
    # Actually, swap means we replace code[idx1] with code[idx2] and vice versa.
    # The code size stays the same.
    # We need to verify the new code is still a covering code.
    # This is expensive, so we do a quick check:
    # - Words in ball(code[idx1]) that were ONLY covered by code[idx1] must now be covered by code[idx2]
    # - Words in ball(code[idx2]) that were ONLY covered by code[idx2] must now be covered by code[idx1]
    
    c1 = code[idx1]
    c2 = code[idx2]
    
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    # Check words in ball(c1) with coverage == 1 (only covered by c1)
    for w in _ball_words(c1, q, n, radius):
        if coverage[word_to_int(w)] == 1:
            # This word is only covered by c1, so it must be covered by c2
            if _hamming_dist(w, c2) > radius:
                return False
    
    # Check words in ball(c2) with coverage == 1 (only covered by c2)
    for w in _ball_words(c2, q, n, radius):
        if coverage[word_to_int(w)] == 1:
            # This word is only covered by c2, so it must be covered by c1
            if _hamming_dist(w, c1) > radius:
                return False
    
    return True

def _local_improve(code, q, n, radius, deadline):
    """Local search: remove redundant codewords and try swaps."""
    # Compute initial coverage
    coverage = _compute_coverage(code, q, n, radius)
    
    # Phase 1: Remove redundant codewords (multiple passes)
    pass_count = 0
    while time.time() < deadline:
        pass_count += 1
        removed_any = False
        i = 0
        while i < len(code):
            if time.time() > deadline:
                return code
            if _try_remove(code, i, q, n, radius, coverage):
                c = code[i]
                # Update coverage
                def word_to_int(w):
                    val = 0
                    for j in range(n):
                        val = val * q + w[j]
                    return val
                for w in _ball_words(c, q, n, radius):
                    coverage[word_to_int(w)] -= 1
                code.pop(i)
                removed_any = True
                # Don't increment i
            else:
                i += 1
        if not removed_any:
            break
    
    # Phase 2: Try swaps to improve (keep same size, but might enable further removals)
    # We'll do a limited number of swap attempts
    swap_attempts = 0
    max_swap_attempts = 5000
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        found_swap = False
        # Try random pairs
        if len(code) < 2:
            break
        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            continue
        if _try_swap(code, i1, i2, q, n, radius, coverage):
            # Perform swap
            code[i1], code[i2] = code[i2], code[i1]
            # Update coverage: need to recompute for affected balls
            # This is complex, so we'll just recompute coverage periodically
            found_swap = True
            # Recompute coverage every 100 swaps
            if swap_attempts % 100 == 0:
                coverage = _compute_coverage(code, q, n, radius)
                # After swap, try removals again
                pass_count = 0
                while time.time() < deadline:
                    pass_count += 1
                    removed_any = False
                    i = 0
                    while i < len(code):
                        if time.time() > deadline:
                            return code
                        if _try_remove(code, i, q, n, radius, coverage):
                            c = code[i]
                            def word_to_int(w):
                                val = 0
                                for j in range(n):
                                    val = val * q + w[j]
                                return val
                            for w in _ball_words(c, q, n, radius):
                                coverage[word_to_int(w)] -= 1
                            code.pop(i)
                            removed_any = True
                        else:
                            i += 1
                    if not removed_any:
                        break
    
    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Handle the specific known case: q=3, n=6, radius=1
    # Best known: 73 words. Direct product gives 9 * 3^2 = 81.
    
    # Initial code: direct product of tetracode and full space on remaining coords
    if q == 3 and n >= 4:
        rem = n - 4
        code = [h + list(tail) for h in _TETRACODE for tail in itertools.product(range(3), repeat=rem)]
    elif q == 2 and n >= 3:
        if n == 3:
            code = _BIN_REP3[:]
        else:
            rem = n - 3
            code = [h + list(tail) for h in _BIN_REP3 for tail in itertools.product(range(2), repeat=rem)]
    else:
        code = [[0] * n]
    
    # Local improvement
    if time.time() < deadline:
        code = _local_improve(code, q, n, radius, deadline)
    
    return code
# EVOLVE-BLOCK-END
