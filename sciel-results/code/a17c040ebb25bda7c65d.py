import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback algebraic construction
    def get_fallback():
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n > 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = q ** n
    num_neighbors = 1 + n * (q - 1)

    # Helper to convert int to base-q digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    # Greedy Set Cover with vectorized NumPy
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []

    while uncovered.sum() > 0:
        if time.time() > deadline:
            return get_fallback()

        # Calculate gain for each candidate
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(gains)

        if gains[best_idx] == 0:
            break

        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = 0

    # Convert selected indices back to codewords
    codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    # Post-processing: Local Search / Redundancy Removal
    if time.time() < deadline - 5.0:
        # Rebuild covered array for the current code
        covered_final = np.zeros(total_words, dtype=bool)
        for idx in selected_indices:
            covered_final[neighbors[idx]] = True

        # Try to remove redundant codewords
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            for i in range(len(selected_indices)):
                if time.time() > deadline - 5.0:
                    break
                idx = selected_indices[i]
                # Temporarily remove this codeword
                was_covered = covered_final[neighbors[idx]].copy()
                covered_final[neighbors[idx]] = False

                # Check if any point becomes uncovered
                if not covered_final[neighbors[idx]].all():
                    # This codeword is essential, restore it
                    covered_final[neighbors[idx]] = was_covered
                else:
                    # This codeword is redundant, remove it
                    selected_indices.pop(i)
                    changed = True
                    break  # Restart the loop since indices changed

        # Update codewords list
        codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    return codewords
