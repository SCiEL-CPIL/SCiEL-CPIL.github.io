"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _neighbors_of(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    yield tuple(word)
    if radius >= 1:
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                yield tuple(new_word)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin within 150s limit

    total_words = q ** n

    # Precompute all words and neighbor indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor index lists for each word
    neighbor_indices = []
    for w in all_words:
        nbs = list(_neighbors_of(w, q, n, radius))
        neighbor_indices.append([word_to_idx[nei] for nei in nbs])

    # Coverage count array
    coverage = np.zeros(total_words, dtype=np.int32)

    # --- Step 1: Start with tetracode product construction ---
    # Tetracode: 9 words of length 4, perfect code radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Direct sum: for n=6, tetracode on first 4 coords, all 3^2=9 tails on last 2
    # This gives 9 * 3^(n-4) codewords
    if n >= 4:
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                initial_codewords.append(tuple(h + list(tail)))
    else:
        # Fallback: greedy for small n
        initial_codewords = []

    # Convert to indices
    initial_indices = [word_to_idx[w] for w in initial_codewords if w in word_to_idx]

    # Build coverage from initial code
    for idx in initial_indices:
        for ni in neighbor_indices[idx]:
            coverage[ni] += 1

    codeword_indices = list(initial_indices)
    in_code = set(codeword_indices)

    # --- Step 2: Local search to reduce code size ---
    def try_remove(i):
        """Try to remove codeword at position i. Returns True if removed."""
        if time.time() > deadline - 5.0:
            return False
        idx = codeword_indices[i]
        # Check if all neighbors of this codeword are covered by others
        for ni in neighbor_indices[idx]:
            if coverage[ni] <= 1:
                return False
        # Remove it
        for ni in neighbor_indices[idx]:
            coverage[ni] -= 1
        codeword_indices.pop(i)
        in_code.discard(idx)
        return True

    def try_swap(i, new_idx):
        """Try to swap codeword at position i with new_idx. Returns True if swapped."""
        if time.time() > deadline - 5.0:
            return False
        old_idx = codeword_indices[i]
        if new_idx == old_idx or new_idx in in_code:
            return False

        # Check: after removing old and adding new, is everything covered?
        # First, check which words would become uncovered if we remove old
        would_uncover = []
        for ni in neighbor_indices[old_idx]:
            if coverage[ni] <= 1:
                would_uncover.append(ni)

        # Check if new covers all those words
        new_neighbors = set(neighbor_indices[new_idx])
        for w in would_uncover:
            if w not in new_neighbors:
                return False

        # Perform swap
        for ni in neighbor_indices[old_idx]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx]:
            coverage[ni] += 1
        codeword_indices[i] = new_idx
        in_code.discard(old_idx)
        in_code.add(new_idx)
        return True

    def find_candidates_covering(words_to_cover, max_candidates=200):
        """Find codewords that cover all words in words_to_cover.
        Uses intersection of candidate sets for efficiency."""
        if not words_to_cover:
            return []
        
        # Start with candidates that cover the first word
        first_word = words_to_cover[0]
        # A word x covers w if dist(x, w) <= radius
        # For radius=1, x covers w if x == w or x differs in exactly 1 position
        
        # Build candidate set incrementally
        candidate_set = set()
        
        # For each word to cover, find all words within radius of it
        # Then intersect all these sets
        for w in words_to_cover:
            w_tuple = all_words[w]
            words_covering_w = set()
            
            # The word itself
            words_covering_w.add(w)
            
            # Words at distance 1
            if radius >= 1:
                for pos in range(n):
                    orig = w_tuple[pos]
                    for val in range(q):
                        if val == orig:
                            continue
                        new_word = list(w_tuple)
                        new_word[pos] = val
                        tw = tuple(new_word)
                        if tw in word_to_idx:
                            words_covering_w.add(word_to_idx[tw])
            
            if candidate_set is not None and len(candidate_set) > 0:
                candidate_set = candidate_set & words_covering_w
            else:
                candidate_set = words_covering_w
            
            if len(candidate_set) == 0:
                return []
        
        # Filter out existing codewords
        candidates = [c for c in candidate_set if c not in in_code]
        return candidates[:max_candidates]

    # Iterative improvement: repeatedly try removals and swaps
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False

        # Phase 1: Try removals
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            if try_remove(i):
                improved = True

        if time.time() > deadline - 10.0:
            break

        # Phase 2: Try swaps - for each codeword, try replacing with targeted candidates
        for i in range(len(codeword_indices)):
            if time.time() > deadline - 10.0:
                break

            old_idx = codeword_indices[i]

            # Find words that are ONLY covered by this codeword
            exclusive_words = []
            for ni in neighbor_indices[old_idx]:
                if coverage[ni] == 1:
                    exclusive_words.append(ni)

            if not exclusive_words:
                # This codeword is redundant (should have been caught in removal)
                continue

            # Find candidates that cover all exclusive words
            candidates = find_candidates_covering(exclusive_words, max_candidates=100)
            
            # Try each candidate
            for cand in candidates:
                if time.time() > deadline - 10.0:
                    break
                if try_swap(i, cand):
                    improved = True
                    break

            # If no targeted swap worked, try some random swaps as fallback
            if not improved and time.time() < deadline - 10.0:
                for _ in range(20):
                    if time.time() > deadline - 10.0:
                        break
                    cand = random.randint(0, total_words - 1)
                    if cand in in_code:
                        continue
                    if try_swap(i, cand):
                        improved = True
                        break

    # --- Step 3: Final cleanup pass ---
    if time.time() < deadline - 5.0:
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            try_remove(i)

    # Convert back to lists
    result = [list(all_words[idx]) for idx in codeword_indices]

    # Verify coverage (debug/safety)
    # final_coverage = np.zeros(total_words, dtype=np.int32)
    # for idx in codeword_indices:
    #     for ni in neighbor_indices[idx]:
    #         final_coverage[ni] += 1
    # if np.any(final_coverage == 0):
    #     print(f"WARNING: {np.sum(final_coverage == 0)} uncovered words")

    return result
# EVOLVE-BLOCK-END
