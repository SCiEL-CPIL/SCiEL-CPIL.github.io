"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    total_words = q ** n

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))

    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute ball masks for each word
    # ball_masks[i] is a bitmask of all words within 'radius' of all_words[i]
    ball_masks = [0] * total_words

    # Helper to compute Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)

    # Precomputation step
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                mask |= (1 << j)
        ball_masks[i] = mask

    # Initialize uncovered set as all 1s
    # Using an integer as a bitset
    uncovered = (1 << total_words) - 1

    code = []

    # Symmetry breaking: Fix the first codeword to all zeros
    # This is valid because the automorphism group is transitive on the space.
    first_word = (0,) * n
    first_idx = word_to_idx[first_word]
    code.append(list(first_word))
    uncovered &= ~ball_masks[first_idx]

    # Greedy Set Cover
    while uncovered != 0:
        if time.time() > deadline:
            break

        best_idx = -1
        best_count = 0

        # Find the word that covers the most uncovered words
        for i in range(total_words):
            covered = uncovered & ball_masks[i]
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        # Add the best word
        code.append(list(all_words[best_idx]))
        uncovered &= ~ball_masks[best_idx]

    return code
# EVOLVE-BLOCK-END
