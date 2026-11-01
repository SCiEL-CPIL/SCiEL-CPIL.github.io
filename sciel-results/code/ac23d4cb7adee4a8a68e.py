"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    # Deadline guard: the evaluator kills after 150s; keep a safety margin.
    deadline = time.time() + 115.0

    # Base construction: direct sum of the ternary tetracode (9 words of length 4,
    # a perfect 1-covering code) with every word on the other n-4 coordinates.
    # This yields 9 * 3**(n-4) codewords.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # For n=6, that's 9 * 9 = 81 words. We now try to remove redundant codewords
    # using greedy local search with coverage-count tracking.

    if n == 6 and radius == 1 and q == 3:
        # Efficient removal: for each codeword, check if removing it leaves all
        # previously-covered words still covered by at least one other codeword.
        # A codeword c covers: itself + all words at Hamming distance 1.
        # Total words covered by c: 1 + n*(q-1) = 1 + 6*2 = 13.

        # Build a fast representation.
        # Encode each word as an integer in [0, 3^6).
        def encode(w):
            v = 0
            for x in w:
                v = v * 3 + x
            return v

        def decode(v):
            w = []
            for _ in range(n):
                w.append(v % 3)
                v //= 3
            return w[::-1]

        # Precompute neighbors (distance <= 1) for each codeword.
        def neighbors(w):
            """Yield the word itself and all words at Hamming distance exactly 1."""
            yield w
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = w[:]
                    new_w[pos] = val
                    yield new_w

        # Build set of all covered words (should be all 729).
        covered = set()
        for c in codewords:
            for w in neighbors(c):
                covered.add(tuple(w))

        # coverage_count[word] = number of codewords that cover it
        cov_count = {}
        for c in codewords:
            for w in neighbors(c):
                key = tuple(w)
                cov_count[key] = cov_count.get(key, 0) + 1

        # Try to remove codewords in a greedy fashion.
        # A codeword c is removable if every word in its ball has cov_count >= 2.
        removed = set()
        codeword_set = set(tuple(c) for c in codewords)

        # We'll iterate multiple passes to find more removable codewords.
        max_passes = 50
        for _pass in range(max_passes):
            if time.time() > deadline - 10:
                break
            found_removal = False
            # Try each codeword in current list
            new_codewords = []
            for c in codewords:
                c_key = tuple(c)
                if c_key in removed:
                    continue
                # Check if c is redundant: all words in its ball have cov_count >= 2
                redundant = True
                for w in neighbors(c):
                    key = tuple(w)
                    if cov_count.get(key, 0) < 2:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    removed.add(c_key)
                    found_removal = True
                    for w in neighbors(c):
                        key = tuple(w)
                        cov_count[key] -= 1
                    # Don't add to new_codewords
                else:
                    new_codewords.append(c)
            codewords = new_codewords
            if not found_removal:
                break

        # Convert back to list of lists
        result = [list(c) for c in codewords]
        return result
    else:
        return codewords
# EVOLVE-BLOCK-END
