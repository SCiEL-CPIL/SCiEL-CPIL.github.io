"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a greedy algorithm to minimize the number of codewords, seeded with a known good
    construction (ternary tetracode direct sum) to improve initial coverage and reduce
    the number of greedy steps needed.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Total number of words in the space
    total_words = q ** n

    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        # Center itself
        neighbors[i].append(i)
        # Variations at distance 1 (if radius >= 1)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        # Variations at distance 2 (if radius >= 2)
        if radius >= 2:
            # This becomes more expensive, but for n=6, q=3, radius=1, we don't need this.
            # If radius was 2, we'd need to iterate pairs of positions.
            pass

    # Coverage mask: True if covered
    covered = [False] * total_words

    # Greedy selection
    codewords = []
    codeword_indices = set()

    # To speed up, we can maintain a list of uncovered words.
    # However, for small n, iterating all total_words is fine.
    # Let's optimize by keeping a set of uncovered indices.
    uncovered_set = set(range(total_words))

    # Seed with a known good construction if applicable:
    # For q=3, n=6, radius=1, use direct sum of ternary tetracode (length 4) with all tails (length 2)
    # Tetracode: 9 words of length 4, covering radius 1
    # Direct sum: 9 * 3^2 = 81 words
    # This gives a valid covering code of size 81, which is better than starting from scratch (which might yield ~100+)

    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    codewords.append(idx)
                    codeword_indices.add(idx)
                    for nb in neighbors[idx]:
                        if not covered[nb]:
                            covered[nb] = True
                            if nb in uncovered_set:
                                uncovered_set.remove(nb)

    while uncovered_set:
        if time.time() > deadline:
            break

        best_center = -1
        best_count = -1

        # We only need to consider centers that are themselves uncovered?
        # Actually, any center can be chosen, but choosing an uncovered one is often good.
        # However, a center that is covered might cover many *other* uncovered words.
        # So we should check all possible centers.

        # Optimization: If the number of uncovered words is small, we might find a center quickly.
        # But iterating all total_words is O(total_words * radius_volume).
        # For n=6, q=3, total_words=729, radius_volume=13. 729*13 is very small.

        for candidate in range(total_words):
            count = 0
            # Count how many neighbors of candidate are in uncovered_set
            # Using a set for O(1) lookup
            for nb in neighbors[candidate]:
                if nb in uncovered_set:
                    count += 1

            if count > best_count:
                best_count = count
                best_center = candidate

        if best_center == -1:
            break

        # Add best_center to codewords
        codewords.append(best_center)
        codeword_indices.add(best_center)

        # Update uncovered_set
        for nb in neighbors[best_center]:
            if nb in uncovered_set:
                uncovered_set.remove(nb)

    # Post-processing: try to remove redundant codewords
    # We iterate until no more can be removed
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Tentatively remove codewords[i]
            c_idx = codewords[i]

            # Check if removing c_idx leaves any word uncovered
            # A word w is uncovered if its only covering codeword was c_idx
            # For efficiency, we check each word in ball(c_idx) to see if it's covered by another codeword

            redundant = True
            for w_idx in neighbors[c_idx]:
                # Check if w_idx is covered by any other codeword
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    # Check if w_idx is in neighbors of codewords[j]
                    # Instead of checking membership in a list (O(n)), we can precompute
                    # a reverse mapping or use a set. But for small n, it's okay.
                    # Actually, let's just check distance directly for clarity and correctness.
                    # But distance check is O(n). Let's use the neighbors list.
                    # We need to check if w_idx is in neighbors[codewords[j]].
                    # This is O(len(neighbors)) = O(13) per check.
                    # Total: O(13 * 70 * 70) ~ 63k ops, which is fine.
                    if w_idx in neighbors[codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break

            if redundant:
                codewords.pop(i)
                codeword_indices.remove(c_idx)
                changed = True
                break  # Restart the loop as indices changed

    # Convert integer indices back to lists of integers
    result = [int_to_word(c, n, q) for c in codewords]

    return result
# EVOLVE-BLOCK-END
