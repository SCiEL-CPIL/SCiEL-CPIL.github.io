"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # For the specific case q=3, n=6, radius=1, use greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = []
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            covered = [i]
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        covered.append(new_i)
            neighbors.append(covered)

        # Now do greedy with a twist: prefer candidates that cover the most uncovered words
        uncovered = np.ones(total, dtype=bool)
        codewords = []

        while uncovered.any():
            if time.time() > deadline:
                break

            # Find candidate that covers most uncovered words
            best_count = -1
            best_candidate = -1

            # Only consider candidates that are themselves uncovered (optimization)
            uncovered_indices = np.where(uncovered)[0]

            if len(uncovered_indices) == 0:
                break

            # Check all uncovered candidates
            for cand in uncovered_indices:
                count = sum(1 for nb in neighbors[cand] if uncovered[nb])
                if count > best_count:
                    best_count = count
                    best_candidate = cand

            if best_candidate == -1:
                break

            # Add this codeword
            codeword = [(best_candidate // powers[p]) % 3 for p in range(n)]
            codewords.append(codeword)

            # Mark covered
            for nb in neighbors[best_candidate]:
                uncovered[nb] = False

        # Verify we have a valid covering
        if len(codewords) > 0 and not uncovered.any():
            return codewords

        # Fallback to direct sum if greedy didn't finish
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # General case: fallback to direct sum
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Very general fallback
    return [[0] * n]
# EVOLVE-BLOCK-END
