"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def _fallback_construct(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140

    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        powers = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)

        all_indices = np.arange(total_words, dtype=np.int64)
        digits = np.zeros((total_words, n), dtype=np.int8)
        for i in range(n):
            digits[:, i] = (all_indices // powers[i]) % 3

        num_covered = 1 + n * (q - 1)
        covered_indices = np.zeros((total_words, num_covered), dtype=np.int64)
        for idx in range(total_words):
            covered_indices[idx, 0] = idx
            pos = 1
            for i in range(n):
                d = digits[idx, i]
                for v in range(q):
                    if v != d:
                        covered_indices[idx, pos] = idx + (v - d) * powers[i]
                        pos += 1

        uncovered = np.ones(total_words, dtype=bool)
        codewords = []

        while uncovered.any():
            if time.time() > deadline:
                return _fallback_construct(q, n, radius)

            counts = uncovered[covered_indices].sum(axis=1)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0:
                break

            uncovered[covered_indices[best_idx]] = False
            codewords.append(digits[best_idx].tolist())

        if not uncovered.any():
            return codewords
        return _fallback_construct(q, n, radius)

    return _fallback_construct(q, n, radius)
# EVOLVE-BLOCK-END
