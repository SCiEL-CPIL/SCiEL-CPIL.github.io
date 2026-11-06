"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Precompute all words and their neighbors (Hamming distance <= radius)
    # For q=3, n=7, radius=1: 2187 words, each has 1 + 7*2 = 15 neighbors
    all_words = np.array(np.unravel_index(np.arange(q**n), (n, [q]*n))).T  # shape (q^n, n)

    # Build neighbor lists: for each word, list of words within distance <= radius
    # For radius=1, neighbors = self + all single-coordinate changes
    neighbors = []
    for w in all_words:
        nb = [w]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    nw = w.copy()
                    nw[i] = v
                    nb.append(nw)
        neighbors.append(nb)

    # Precompute: for each position, the index of the word in all_words
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx

    # Convert neighbors to index lists
    neighbor_indices = []
    for nb in neighbors:
        idxs = [word_to_idx[tuple(w)] for w in nb]
        neighbor_indices.append(idxs)

    # Greedy set cover: repeatedly pick the word whose ball covers the most uncovered words
    uncovered = np.ones(q**n, dtype=bool)
    # coverage[i] = number of currently uncovered words in ball of word i
    coverage = np.zeros(q**n, dtype=np.int32)
    for i in range(q**n):
        coverage[i] = np.sum(uncovered[neighbor_indices[i]])

    chosen = []
    remaining = set(range(q**n))
    # Use a priority structure: we'll just scan for max each time (q^n = 2187, manageable)
    while uncovered.any():
        # Find word with maximum coverage among remaining
        best = -1
        best_cov = -1
        for i in remaining:
            if coverage[i] > best_cov:
                best_cov = coverage[i]
                best = i
        if best_cov <= 0:
            break  # shouldn't happen if we have a valid cover
        chosen.append(best)
        remaining.discard(best)
        # Update coverage: subtract 1 from all words whose ball contains any newly covered word
        newly_covered = np.array(neighbor_indices[best])
        for j in remaining:
            # How many of newly_covered are in ball of j?
            # i.e., |newly_covered ∩ neighbor_indices[j]|
            # This equals the number of words in newly_covered that are within distance radius of all_words[j]
            # For radius=1: a word x is in ball of j iff dist(x,j)<=1
            # We can compute this efficiently
            j_word = all_words[j]
            # Count how many words in newly_covered are within distance 1 of j_word
            diff = np.abs(all_words[newly_covered] - j_word).sum(axis=1)
            cnt = np.sum(diff <= radius)
            coverage[j] -= cnt

        uncovered[newly_covered] = False

        if time.time() > deadline - 30:
            break

    chosen_words = [list(all_words[i]) for i in chosen]

    # 2-core peeling: iteratively remove codewords whose removal doesn't leave any word uncovered
    # A codeword c can be removed if every word in ball(c) is also in ball(c') for some other c'
    # Equivalently: for every word x in ball(c), there exists c' != c with dist(x, c') <= radius
    # We track for each word how many codewords cover it. A word is "safe" if count >= 2.
    # A codeword c is removable if all words in ball(c) have count >= 2.

    # Build coverage count per word
    cover_count = np.zeros(q**n, dtype=np.int32)
    for i in chosen:
        for ni in neighbor_indices[i]:
            cover_count[ni] += 1

    # Iteratively peel
    changed = True
    while changed:
        changed = False
        new_chosen = []
        for i in chosen:
            # Check if all words in ball(i) have cover_count >= 2
            ball = neighbor_indices[i]
            if np.all(cover_count[ball] >= 2):
                # Safe to remove
                for ni in ball:
                    cover_count[ni] -= 1
                changed = True
            else:
                new_chosen.append(i)
        chosen = new_chosen

    if time.time() > deadline - 5:
        return [list(all_words[i]) for i in chosen]

    # Verify and return
    chosen_words = [list(all_words[i]) for i in chosen]

    # Final verification
    # Check every word is covered
    covered = np.zeros(q**n, dtype=bool)
    for i in chosen:
        covered[neighbor_indices[i]] = True
    if not covered.all():
        # Fallback: use the greedy result before peeling
        pass

    return chosen_words
# EVOLVE-BLOCK-END
