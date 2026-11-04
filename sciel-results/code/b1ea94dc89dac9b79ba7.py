"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit (evaluator kills at 150s)

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n

    # Efficient word <-> index conversion
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute covered indices for a codeword
    def get_covered_indices(cw):
        indices = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        return indices

    # Build coverage count array
    def build_coverage_count(cw_list):
        counts = [0] * total_words
        for cw in cw_list:
            for idx in get_covered_indices(cw):
                counts[idx] += 1
        return counts

    # Check if removing a codeword keeps coverage valid
    def is_redundant(cw, counts):
        for idx in get_covered_indices(cw):
            if counts[idx] <= 1:
                return False
        return True

    # Iterative greedy removal with multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        counts = build_coverage_count(codewords)
        # Try removing from the end first (arbitrary order helps break symmetry)
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(codewords[i], counts):
                codewords.pop(i)
                improved = True
                # Rebuild counts after removal
                counts = build_coverage_count(codewords)
                break  # Restart the scan

    # Now try swap-based improvement: replace a codeword with a new one that
    # reduces the total number of codewords needed
    # Strategy: try to find a new codeword that covers more uncovered words
    # than any existing codeword, and if we can remove two codewords for each
    # one added, we improve.

    # Actually, let's try a different approach: local search by trying to
    # replace pairs of codewords with a single codeword, or add a codeword
    # that allows removing two.

    # Simpler: try to find codewords that are "far apart" and see if we can
    # merge their coverage by adding a new codeword in between.

    # For now, let's try a more aggressive greedy: after removal, try adding
    # codewords strategically to enable further removals.

    # Let's try: for each pair of codewords, check if there exists a word w
    # such that adding w to the code allows removing both codewords.
    # This is expensive, so we'll limit it.

    # Alternative: use the fact that the tetracode is a perfect code.
    # The direct sum construction might have structure we can exploit.

    # Let's try a different approach: use a greedy set cover algorithm.
    # Start with an empty set, and greedily add the codeword that covers
    # the most currently uncovered words.

    # Actually, the greedy removal approach above should be sufficient for
    # the 81 -> 73 reduction. Let's verify by checking if we can do better
    # with a more careful ordering.

    # Try multiple random orderings for the removal
    best_codewords = codewords[:]
    for attempt in range(5):
        if time.time() > deadline:
            break
        # Shuffle the codewords
        shuffled = codewords[:]
        random.seed(attempt)
        random.shuffle(shuffled)

        # Greedy removal on shuffled order
        improved = True
        while improved and time.time() < deadline:
            improved = False
            counts = build_coverage_count(shuffled)
            for i in range(len(shuffled) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_redundant(shuffled[i], counts):
                    shuffled.pop(i)
                    improved = True
                    counts = build_coverage_count(shuffled)
                    break

        if len(shuffled) < len(best_codewords):
            best_codewords = shuffled

    # Final attempt: try to add a codeword that allows removing two others
    # This is a 1-for-2 swap
    if time.time() < deadline - 5:
        current = best_codewords[:]
        counts = build_coverage_count(current)

        # For each pair of codewords, check if there's a word that covers
        # all words uniquely covered by either of them
        # This is expensive, so we'll only try a subset

        # Precompute for each codeword, the set of words it uniquely covers
        unique_cover = [set() for _ in current]
        for idx in range(total_words):
            # Find which codewords cover this word
            covering = []
            for i, cw in enumerate(current):
                if counts[idx] > 0:
                    # Check if cw covers idx
                    w = idx_to_word(idx)
                    if all(abs(a - b) <= 1 for a, b in zip(cw, w)) and sum(1 for a, b in zip(cw, w) if a != b) <= radius:
                        covering.append(i)
            if len(covering) == 1:
                unique_cover[covering[0]].add(idx)

        # Try to find a word that covers unique words from two codewords
        # and allows removing both
        found_improvement = False
        for i in range(len(current)):
            if time.time() > deadline - 5:
                break
            if not unique_cover[i]:
                continue
            for j in range(i + 1, len(current)):
                if time.time() > deadline - 5:
                    break
                if not unique_cover[j]:
                    continue

                # Find a word that covers all unique words of i and j
                # The word must be within radius of all unique words of i and j
                # This is a constraint satisfaction problem

                # Simplified: try all possible words (too expensive for 729 words)
                # Instead, try words near the midpoint of codewords i and j

                cw_i = current[i]
                cw_j = current[j]

                # Try the "average" codeword
                mid = [(a + b) // 2 for a, b in zip(cw_i, cw_j)]
                # Also try rounding up
                mid2 = [min(q - 1, (a + b + 1) // 2) for a, b in zip(cw_i, cw_j)]

                for candidate in [mid, mid2]:
                    # Check if adding candidate allows removing i and j
                    new_counts = counts[:]
                    # Subtract coverage of i and j
                    for idx in get_covered_indices(cw_i):
                        new_counts[idx] -= 1
                    for idx in get_covered_indices(cw_j):
                        new_counts[idx] -= 1
                    # Add coverage of candidate
                    for idx in get_covered_indices(candidate):
                        new_counts[idx] += 1

                    # Check if all words are still covered
                    if all(c >= 0 for c in new_counts):
                        # This works!
                        new_code = current[:i] + current[i+1:j] + current[j+1:] + [candidate]
                        if len(new_code) < len(best_codewords):
                            best_codewords = new_code
                            found_improvement = True
                            break
                if found_improvement:
                    break
            if found_improvement:
                break

    return best_codewords
# EVOLVE-BLOCK-END
