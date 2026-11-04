"""Football pool problem: ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    # Precompute all words and their neighbors (self + distance-1 variants)
    # Total space: q^n = 3^7 = 2187
    # Each codeword covers 1 + n*(q-1) = 1 + 7*2 = 15 words

    # Generate all words as integers 0..q^n-1
    total_words = q ** n
    # Convert integer to word (list of digits)
    def int_to_word(x):
        return [x // (q ** (n - 1 - i)) % q for i in range(n)]

    def word_to_int(w):
        return sum(w[i] * q ** (n - 1 - i) for i in range(n))

    # Precompute coverage sets: for each word, which words does it cover?
    # cover[word_int] = set of word_ints covered (word + all distance-1 variants)
    cover_sets = []
    for x in range(total_words):
        w = int_to_word(x)
        covered = set([x])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w[pos] = val
                    covered.add(word_to_int(w))
            w[pos] = orig
        cover_sets.append(covered)

    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    uncovered = set(range(total_words))
    selected = []
    # For efficiency, track coverage count for each candidate
    # cov_count[c] = number of currently uncovered words in cover_sets[c]
    cov_count = np.array([len(cover_sets[c] & uncovered) for c in range(total_words)], dtype=np.int32)

    while uncovered:
        # Find candidate with max coverage
        best_idx = int(np.argmax(cov_count))
        if cov_count[best_idx] == 0:
            break  # Should not happen if uncovered is non-empty
        selected.append(best_idx)
        # Remove covered words from uncovered
        to_remove = cover_sets[best_idx] & uncovered
        uncovered -= to_remove
        # Update cov_count for all candidates (only those that lost coverage)
        for word in to_remove:
            # For each candidate c, if word was in cover_sets[c], decrement
            # This is expensive; instead, rebuild periodically or use a different approach
            pass

        # Periodic rebuild of cov_count (every 20 selections or when it gets stale)
        if len(selected) % 20 == 0 and uncovered:
            cov_count = np.array([len(cover_sets[c] & uncovered) for c in range(total_words)], dtype=np.int32)

        # Check deadline
        if time.time() > deadline - 5:
            break

    # If we broke early due to deadline, we may have uncovered words - add them directly
    if uncovered and time.time() < deadline - 2:
        # Greedily add remaining uncovered words
        for w in list(uncovered):
            if time.time() > deadline - 1:
                break
            if w in uncovered:
                selected.append(w)
                uncovered -= cover_sets[w]

    # Now try to reduce: remove codewords that don't leave any word uncovered
    # A codeword c can be removed if every word in cover_sets[c] is also covered by some other selected codeword
    if time.time() < deadline - 3:
        # Build coverage map: for each word, how many selected codewords cover it?
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            for w in cover_sets[c]:
                cover_count[w] += 1

        # Try removing codewords in order of lowest "unique coverage"
        # Unique coverage of c = number of words where cover_count[w] == 1 and c covers w
        removable = True
        while removable and time.time() < deadline - 2:
            removable = False
            for i in range(len(selected)):
                if time.time() > deadline - 2:
                    break
                c = selected[i]
                # Check if removing c leaves any word uncovered
                # A word w in cover_sets[c] is left uncovered if cover_count[w] == 1
                can_remove = True
                for w in cover_sets[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    for w in cover_sets[c]:
                        cover_count[w] -= 1
                    selected.pop(i)
                    removable = True
                    break

    # Convert back to list of words
    result = [int_to_word(c) for c in selected]

    # Ensure distinctness (should be, but just in case)
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)

    return unique_result
# EVOLVE-BLOCK-END
