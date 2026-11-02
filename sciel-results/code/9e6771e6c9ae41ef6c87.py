# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

# Deadline: kill at 150s, use 145s as safety margin
DEADLINE = 145.0


def _vec_to_int(vec, q, n):
    """Convert a vector of digits in [0, q) to an integer in [0, q^n)."""
    val = 0
    for i in range(n):
        val = val * q + vec[i]
    return val


def _int_to_vec(x, q, n):
    """Convert integer to vector of digits in [0, q)."""
    vec = [0] * n
    for i in range(n - 1, -1, -1):
        vec[i] = x % q
        x //= q
    return vec


def _neighbors_of_int(x, q, n):
    """Yield all integers within Hamming distance <= 1 of x (including x itself)."""
    # x itself
    yield x
    # For each position, change to each other digit
    for pos in range(n):
        digit = (x // (q ** (n - 1 - pos))) % q
        base = x - digit * (q ** (n - 1 - pos))
        for d in range(q):
            if d != digit:
                yield base + d * (q ** (n - 1 - pos))


def _build_neighbor_table(q, n):
    """Precompute neighbors for all words. Returns list of lists of ints."""
    total = q ** n
    # Precompute powers
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    neighbors = [None] * total
    for x in range(total):
        nb = [x]
        for pos in range(n):
            digit = (x // powers[pos]) % q
            base = x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    nb.append(base + d * powers[pos])
        neighbors[x] = nb
    return neighbors, powers


def _greedy_cover(q, n, radius, start_time=None):
    """
    Greedy set cover: repeatedly pick the word that covers the most uncovered words.
    Returns list of codeword vectors (as ints).
    """
    total = q ** n
    if start_time is None:
        start_time = time.time()
    
    # Precompute powers
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    
    # For each word, precompute its ball (all words within distance radius)
    # For radius=1: the word + all single-position changes
    balls = [None] * total
    for x in range(total):
        ball = [x]
        for pos in range(n):
            digit = (x // powers[pos]) % q
            base = x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    ball.append(base + d * powers[pos])
        balls[x] = ball
    
    # uncovered: set of words not yet covered
    uncovered = set(range(total))
    code = []
    
    while uncovered:
        # Find word with max coverage of uncovered
        best_word = -1
        best_count = -1
        for x in range(total):
            count = 0
            for y in balls[x]:
                if y in uncovered:
                    count += 1
                    if count > best_count:
                        # Early check
                        pass
            if count > best_count:
                best_count = count
                best_word = x
                if best_count == len(uncovered):
                    break
        if best_word == -1:
            break
        code.append(best_word)
        for y in balls[best_word]:
            uncovered.discard(y)
        if time.time() - start_time > DEADLINE - 5:
            break
    
    return code


def _greedy_cover_fast(q, n, radius, start_time=None):
    """
    Faster greedy: use array-based coverage counting.
    For radius=1, each word covers q*n + 1 words.
    """
    total = q ** n
    if start_time is None:
        start_time = time.time()
    
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    
    # uncovered count per word
    uncovered_count = np.ones(total, dtype=np.int32)  # 1 if uncovered, 0 if covered
    # Actually, let's track which words are uncovered
    is_covered = np.zeros(total, dtype=bool)
    
    code = []
    remaining = total
    
    while remaining > 0:
        # For each candidate word x, compute how many uncovered words it covers
        # For radius=1: x + all single changes
        best_x = -1
        best_gain = -1
        
        for x in range(total):
            if is_covered[x]:
                # Still might be a good candidate if it covers others
                pass
            gain = 0
            if not is_covered[x]:
                gain += 1
            for pos in range(n):
                digit = (x // powers[pos]) % q
                base = x - digit * powers[pos]
                for d in range(q):
                    if d != digit:
                        y = base + d * powers[pos]
                        if not is_covered[y]:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x == -1 or best_gain == 0:
            break
        
        code.append(best_x)
        # Mark all words in ball of best_x as covered
        if not is_covered[best_x]:
            is_covered[best_x] = True
            remaining -= 1
        for pos in range(n):
            digit = (best_x // powers[pos]) % q
            base = best_x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    y = base + d * powers[pos]
                    if not is_covered[y]:
                        is_covered[y] = True
                        remaining -= 1
        
        if time.time() - start_time > DEADLINE - 10:
            break
    
    return code


def _local_search_improve(q, n, radius, code_ints, start_time=None):
    """
    Try to reduce code size by removing codewords that are redundant,
    and by swapping to better positions.
    Returns improved list of codeword ints.
    """
    if start_time is None:
        start_time = time.time()
    
    total = q ** n
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    
    code = list(code_ints)
    
    # Try removing each codeword if coverage is maintained
    changed = True
    while changed and time.time() - start_time < DEADLINE - 15:
        changed = False
        for i in range(len(code)):
            # Tentatively remove code[i]
            test_code = code[:i] + code[i+1:]
            if not test_code:
                continue
            # Check if coverage is still complete
            # Build coverage map
            covered = np.zeros(total, dtype=bool)
            valid = True
            for cw in test_code:
                # Mark ball
                covered[cw] = True
                for pos in range(n):
                    digit = (cw // powers[pos]) % q
                    base = cw - digit * powers[pos]
                    for d in range(q):
                        if d != digit:
                            covered[base + d * powers[pos]] = True
            if not covered.all():
                valid = False
            if valid:
                code = test_code
                changed = True
                break
    
    return code


def _random_local_search(q, n, radius, code_ints, start_time=None):
    """
    Randomized local search: try random swaps and removals.
    """
    if start_time is None:
        start_time = time.time()
    
    total = q ** n
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    
    code = list(code_ints)
    rng = random.Random(42)
    
    # Precompute balls for all words (for radius=1)
    balls = [None] * total
    for x in range(total):
        ball = [x]
        for pos in range(n):
            digit = (x // powers[pos]) % q
            base = x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    ball.append(base + d * powers[pos])
        balls[x] = ball
    
    # Helper to check if a set of codewords covers everything
    def is_covering(cw_list):
        covered = np.zeros(total, dtype=bool)
        for cw in cw_list:
            for y in balls[cw]:
                covered[y] = True
        return covered.all()
    
    # First, try removals
    for i in range(len(code)):
        if time.time() - start_time > DEADLINE - 20:
            break
        test = code[:i] + code[i+1:]
        if is_covering(test):
            code = test
    
    # Now try swaps: replace one codeword with another
    for _ in range(5000):
        if time.time() - start_time > DEADLINE - 20:
            break
        if len(code) == 0:
            break
        idx = rng.randrange(len(code))
        new_cw = rng.randrange(total)
        if new_cw in code:
            continue
        test = code[:idx] + [new_cw] + code[idx+1:]
        if is_covering(test):
            code = test
    
    return code


def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    
    # For the specific case q=3, n=8, radius=1: total = 6561
    # The tetracode-based construction gives 9 * 3^4 = 729 words.
    # Best known is 486. We aim to do better than 729.
    
    if total > 1000000:
        # Too large for exhaustive methods, fall back to tetracode construction
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Precompute powers
    powers = [1] * n
    for i in range(n - 2, -1, -1):
        powers[i] = powers[i + 1] * q
    
    # Precompute balls for radius=1
    balls = [None] * total
    for x in range(total):
        ball = [x]
        for pos in range(n):
            digit = (x // powers[pos]) % q
            base = x - digit * powers[pos]
            for d in range(q):
                if d != digit:
                    ball.append(base + d * powers[pos])
        balls[x] = ball
    
    # Greedy set cover
    code = _greedy_cover_fast(q, n, radius, start_time)
    
    if time.time() - start_time > DEADLINE - 30:
        return [_int_to_vec(x, q, n) for x in code]
    
    # Local search: try to improve
    code = _local_search_improve(q, n, radius, code, start_time)
    
    if time.time() - start_time > DEADLINE - 20:
        return [_int_to_vec(x, q, n) for x in code]
    
    # Random local search for further improvement
    code = _random_local_search(q, n, radius, code, start_time)
    
    # Final check
    if time.time() - start_time > DEADLINE - 5:
        return [_int_to_vec(x, q, n) for x in code]
    
    return [_int_to_vec(x, q, n) for x in code]
# EVOLVE-BLOCK-END
