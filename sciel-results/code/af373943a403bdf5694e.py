import time
import numpy as np


def construct(q, n, radius):
    """Heuristic covering code: greedy set-cover with local improvement."""
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 12.0

    total = q ** n

    # Precompute digit vectors
    digits = np.empty((total, n), dtype=np.int8)
    for i in range(total):
        x = i
        for pos in range(n):
            digits[i, pos] = x % q
            x //= q

    # Precompute cover sets: for each word, indices of words within Hamming distance <= radius
    # For radius=1, q=3, n=7: each covers 1 + 7*2 = 15 words
    cover_sets = []
    for i in range(total):
        w = digits[i]
        covered = [i]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for d in range(q):
                    if d == orig:
                        continue
                    delta = (d - orig) * (q ** (n - 1 - pos))
                    covered.append(i + delta)
        cover_sets.append(np.array(covered, dtype=np.int32))

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    chosen = []

    while uncovered.any():
        if time.time() - t0 > DEADLINE - MARGIN:
            break

        best_i = -1
        best_count = -1
        for i in range(total):
            cnt = np.count_nonzero(uncovered[cover_sets[i]])
            if cnt > best_count:
                best_count = cnt
                best_i = i

        if best_count <= 0:
            break

        chosen.append(best_i)
        uncovered[cover_sets[best_i]] = False

    # Local improvement: try removing codewords if coverage is maintained
    # Use a coverage count array: cov_count[w] = number of chosen codewords covering w
    if time.time() - t0 < DEADLINE - MARGIN:
        cov_count = np.zeros(total, dtype=np.int32)
        for c in chosen:
            cov_count[cover_sets[c]] += 1

        improved = True
        while improved and time.time() - t0 < DEADLINE - MARGIN:
            improved = False
            for idx in range(len(chosen)):
                if time.time() - t0 > DEADLINE - MARGIN:
                    break
                c = chosen[idx]
                # Check if any word has cov_count == 1 for this codeword (i.e., only covered by c)
                # If all words covered by c have cov_count >= 2, we can remove c
                can_remove = np.all(cov_count[cover_sets[c]] >= 2)
                if can_remove:
                    # Remove c
                    cov_count[cover_sets[c]] -= 1
                    chosen.pop(idx)
                    improved = True
                    break  # Restart loop

    result = [digits[i].tolist() for i in chosen]
    return result
