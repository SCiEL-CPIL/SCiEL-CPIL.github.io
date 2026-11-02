import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    limit = 145.0  # Safety margin for 150s limit

    total_words = q ** n
    powers = [q ** i for i in range(n)]

    def get_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        return digits

    # Precompute neighbor masks as bitmasks
    # For 6561 bits, Python integers handle this efficiently.
    neighbor_masks = [0] * total_words

    for idx in range(total_words):
        mask = 1 << idx
        # Radius 1 neighbors
        for i in range(n):
            original = (idx // powers[i]) % q
            base = idx - original * powers[i]
            for new_val in range(q):
                if new_val != original:
                    neighbor_idx = base + new_val * powers[i]
                    mask |= (1 << neighbor_idx)
        neighbor_masks[idx] = mask

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    solution_indices = []

    while uncovered:
        if time.time() - start_time > limit:
            break

        best_word = -1
        max_covered = 0

        # Optimization: Iterate only through words that might still cover something?
        # Actually, we can iterate all, but bit_count is fast.
        # We can also skip words already in solution if we track them, but bit_count of 0 is fast enough.
        # However, as the solution grows, checking 6561 words is the bottleneck.
        # Let's just do the standard loop.
        for w in range(total_words):
            count = (neighbor_masks[w] & uncovered).bit_count()
            if count > max_covered:
                max_covered = count
                best_word = w

        if best_word == -1:
            break

        solution_indices.append(best_word)
        uncovered &= ~neighbor_masks[best_word]

    # Local Search: Remove redundant codewords
    # A codeword is redundant if its neighborhood is fully covered by the union of the others.
    new_solution = list(solution_indices)
    changed = True
    while changed and (time.time() - start_time < limit):
        changed = False
        for i in range(len(new_solution) - 1, -1, -1):
            if time.time() - start_time > limit:
                break
            w = new_solution[i]
            # Check if w is redundant
            # Build mask of all other codewords
            others_mask = 0
            for j in range(len(new_solution)):
                if j != i:
                    others_mask |= neighbor_masks[new_solution[j]]

            if (neighbor_masks[w] & ~others_mask) == 0:
                # w is redundant
                new_solution.pop(i)
                changed = True
                break  # Restart loop as indices changed

    # Additional Local Search: Try to replace pairs of codewords with one?
    # This is much harder. Let's stick to removal for now as it's fast and effective.
    # Another idea: "Shifting" or moving a codeword to a better position?
    # Given the time, removal is the most reliable improvement over greedy.

    return [get_digits(idx) for idx in new_solution]
