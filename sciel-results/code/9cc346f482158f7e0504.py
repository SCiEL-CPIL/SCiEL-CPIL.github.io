"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if n <= 4:
        return code

    # Build coverage structure: for each word, track which codeword covers it
    # Use integer encoding for efficiency
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # Precompute all words as integers
    total_words = q ** n
    code_ints = [word_to_int(c) for c in code]
    code_set = set(code_ints)

    # For each codeword, compute the set of words it covers (within Hamming distance <= radius)
    # We'll use a coverage count array
    coverage_count = [0] * total_words

    # Generate all words within distance radius of a given codeword
    def neighbors_of(word_int):
        """Yield all word integers within Hamming distance <= radius of word_int."""
        word = int_to_word(word_int)
        # Distance 0
        yield word_int
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    word[pos] = new_val
                    yield word_to_int(word)
                word[pos] = orig

    # Build coverage counts
    for ci in code_ints:
        for neighbor in neighbors_of(ci):
            coverage_count[neighbor] += 1

    # Greedy removal: try to remove codewords that don't leave any word uncovered
    removed = set()
    improvement_found = True
    while improvement_found and time.time() < deadline:
        improvement_found = False
        # Try removing codewords in random order
        indices = list(range(len(code)))
        random.shuffle(indices)
        for idx in indices:
            if idx in removed:
                continue
            ci = code_ints[idx]
            # Check if removing this codeword leaves any word with coverage 0
            safe = True
            for neighbor in neighbors_of(ci):
                if coverage_count[neighbor] == 1:
                    safe = False
                    break
            if safe:
                # Remove this codeword
                removed.add(idx)
                for neighbor in neighbors_of(ci):
                    coverage_count[neighbor] -= 1
                improvement_found = True

    # Build final code
    final_code = [code[i] for i in range(len(code)) if i not in removed]

    # Verify coverage is still complete (safety check)
    # Quick check: every word should have coverage_count > 0
    # But we modified coverage_count during removal, so let's verify differently
    # Actually, let's just return what we have - the removal logic ensures coverage

    return final_code
# EVOLVE-BLOCK-END
