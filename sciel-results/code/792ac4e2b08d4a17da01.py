"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Ternary covering code K_3(8,1).
    # We use a greedy set-covering approach with a good initial seed.
    # The alphabet is {0,1,2}, length n=8, radius=1.
    # Total space size = 3**8 = 6561.
    # Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words.
    # Lower bound: ceil(6561/17) = 386. Best known: 486.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _greedy_cover(q, n, radius)
    
    # For the specific case q=3, n=8, radius=1:
    # We'll use a greedy algorithm with randomized restarts to find a small covering.
    # To stay within time limits, we'll do a single greedy pass with a heuristic.
    
    # Represent words as integers in [0, 3**8)
    total = 3 ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # Precompute for each word, the list of words within Hamming distance 1
    # This is expensive to precompute fully, so we compute on the fly or use a trick.
    
    # Alternative: use a greedy approach where we pick the word that covers the most uncovered words.
    # For efficiency, we can iterate through all words and pick the best one each time.
    # But 6561 words, each covering 17, so we need about 6561/17 ~ 386 iterations.
    # Each iteration scans 6561 words, checking 17 neighbors each -> ~6561*17*386 ~ 43M ops, feasible.
    
    # Let's precompute neighbors for each word to speed things up.
    # neighbors[i] = list of words within Hamming distance 1 of word i
    # This is 6561 * 17 = 111537 entries, very manageable.
    
    # Convert integer to base-3 digits
    def int_to_digits(x, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits
    
    def digits_to_int(digits):
        x = 0
        for i in range(n):
            x = x * 3 + digits[i]
        return x
    
    # Precompute neighbors
    neighbors = []
    for i in range(total):
        digits = int_to_digits(i, n)
        nb = [i]
        for pos in range(n):
            orig = digits[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                digits[pos] = new_val
                nb.append(digits_to_int(digits))
            digits[pos] = orig
        neighbors.append(nb)
    
    # Greedy set cover
    remaining = set(range(total))
    for _ in range(2000):  # Safety cap
        if not remaining:
            break
        best_word = -1
        best_count = -1
        for w in remaining:
            count = sum(1 for x in neighbors[w] if x in remaining)
            if count > best_count:
                best_count = count
                best_word = w
        if best_word == -1:
            break
        codewords.append(best_word)
        for x in neighbors[best_word]:
            remaining.discard(x)
    
    # Convert back to list of digits
    result = [int_to_digits(c, n) for c in codewords]
    return result


def _greedy_cover(q, n, radius):
    """Fallback greedy covering for general q, n, radius."""
    total = q ** n
    if total > 100000:
        # Too large, return a trivial solution (all words)
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    def int_to_digits(x, n):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    def digits_to_int(digits):
        x = 0
        for i in range(n):
            x = x * q + digits[i]
        return x
    
    # Precompute neighbors within Hamming distance 'radius'
    # For small radius, this is manageable
    neighbors = []
    for i in range(total):
        digits = int_to_digits(i, n)
        nb = set()
        # Generate all words within Hamming distance radius
        # For radius=1, this is easy. For larger radius, we use itertools.
        if radius == 1:
            nb.add(i)
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    nb.add(digits_to_int(digits))
                digits[pos] = orig
        else:
            # General case: generate all combinations of positions to change
            from itertools import combinations, product
            nb.add(i)
            for r in range(1, radius + 1):
                for positions in combinations(range(n), r):
                    for vals in product(range(q), repeat=r):
                        new_digits = digits[:]
                        for pos, val in zip(positions, vals):
                            new_digits[pos] = val
                        nb.add(digits_to_int(new_digits))
        neighbors.append(list(nb))
    
    remaining = set(range(total))
    codewords = []
    for _ in range(total):
        if not remaining:
            break
        best_word = -1
        best_count = -1
        for w in remaining:
            count = sum(1 for x in neighbors[w] if x in remaining)
            if count > best_count:
                best_count = count
                best_word = w
        if best_word == -1:
            break
        codewords.append(best_word)
        for x in neighbors[best_word]:
            remaining.discard(x)
    
    return [int_to_digits(c, n) for c in codewords]
# EVOLVE-BLOCK-END
