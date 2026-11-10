"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _get_neighbors(word, q, n, radius, word_to_idx):
    """Get all words within Hamming distance <= radius of word as indices."""
    result = [word_to_idx[tuple(word)]]
    if radius >= 1:
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                tw = tuple(new_word)
                if tw in word_to_idx:
                    result.append(word_to_idx[tw])
    return result


def _apply_permutation(word, perm):
    """Apply coordinate permutation to a word."""
    return [word[p] for p in perm]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin within 150s limit

    total_words = q ** n

    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor index lists for each word
    neighbor_indices = []
    for w in all_words:
        nbs = _get_neighbors(w, q, n, radius, word_to_idx)
        neighbor_indices.append(nbs)

    # Tetracode: 9 words of length 4, perfect code radius 1
    tetracode_base = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # Generate different coordinate permutations for each subcube to break symmetry
    # For n=6, we have 4+2 structure: tetracode on 4 coords, free on 2
    # We can permute which 4 coordinates the tetracode uses
    if n >= 4:
        # All ways to choose 4 coordinates out of n for the tetracode
        coord_choices = list(itertools.combinations(range(n), 4))
        # For each subcube index, assign a different coordinate choice
        num_subcubes = q ** (n - 4)
        
        # Build initial code with diverse tetracode placements
        initial_codewords = []
        used_perms = []
        
        for sub_idx in range(num_subcubes):
            # Determine the "tail" values (the n-4 free coordinates)
            tail_digits = []
            temp = sub_idx
            for _ in range(n - 4):
                tail_digits.append(temp % q)
                temp //= q
            tail_digits = tuple(tail_digits)
            
            # Choose coordinate positions for tetracode based on sub_idx
            # Use a deterministic but varied assignment
            perm_idx = sub_idx % len(coord_choices)
            tetracode_coords = coord_choices[perm_idx]
            
            # The remaining coordinates get the tail values
            remaining_coords = [c for c in range(n) if c not in tetracode_coords]
            
            # Assign tail values to remaining coords
            coord_val_map = {}
            for i, c in enumerate(remaining_coords):
                coord_val_map[c] = tail_digits[i]
            
            # For each tetracode word, create a full codeword
            for tc in tetracode_base:
                word = [0] * n
                for i, c in enumerate(tetracode_coords):
                    word[c] = tc[i]
                for c, v in coord_val_map.items():
                    word[c] = v
                initial_codewords.append(tuple(word))
        
        # Remove duplicates
        initial_codewords = list(set(initial_codewords))
    else:
        # Fallback for small n: greedy
        initial_codewords = []

    # Convert to indices
    initial_indices = [word_to_idx[w] for w in initial_codewords if w in word_to_idx]

    # Build coverage from initial code
    coverage = np.zeros(total_words, dtype=np.int32)
    for idx in initial_indices:
        for ni in neighbor_indices[idx]:
            coverage[ni] += 1

    codeword_indices = list(initial_indices)
    in_code = set(codeword_indices)

    def try_remove(i):
        """Try to remove codeword at position i. Returns True if removed."""
        if time.time() > deadline - 5.0:
            return False
        idx = codeword_indices[i]
        for ni in neighbor_indices[idx]:
            if coverage[ni] <= 1:
                return False
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

        would_uncover = []
        for ni in neighbor_indices[old_idx]:
            if coverage[ni] <= 1:
                would_uncover.append(ni)

        new_neighbors = set(neighbor_indices[new_idx])
        for w in would_uncover:
            if w not in new_neighbors:
                return False

        for ni in neighbor_indices[old_idx]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx]:
            coverage[ni] += 1
        codeword_indices[i] = new_idx
        in_code.discard(old_idx)
        in_code.add(new_idx)
        return True

    def find_candidates_covering(words_to_cover, max_candidates=100):
        """Find non-codeword indices that cover all words in words_to_cover."""
        if not words_to_cover:
            return []
        
        # For each word to cover, compute the set of all words within radius
        # Then intersect these sets
        candidate_set = None
        for w in words_to_cover:
            w_tuple = all_words[w]
            covering_words = set()
            
            # The word itself
            covering_words.add(w)
            
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
                            covering_words.add(word_to_idx[tw])
            
            if candidate_set is None:
                candidate_set = covering_words
            else:
                candidate_set = candidate_set & covering_words
            
            if len(candidate_set) == 0:
                return []
        
        candidates = [c for c in candidate_set if c not in in_code]
        return candidates[:max_candidates]

    # Iterative improvement loop
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline - 15.0 and iteration < max_iterations:
        improved = False
        iteration += 1

        # Phase 1: Try removals (reverse order to handle index shifts)
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 15.0:
                break
            if try_remove(i):
                improved = True

        if time.time() > deadline - 15.0:
            break

        # Phase 2: Targeted swaps
        for i in range(len(codeword_indices)):
            if time.time() > deadline - 15.0:
                break

            old_idx = codeword_indices[i]

            # Find words that are ONLY covered by this codeword
            exclusive_words = []
            for ni in neighbor_indices[old_idx]:
                if coverage[ni] == 1:
                    exclusive_words.append(ni)

            if not exclusive_words:
                continue

            # Find candidates that cover all exclusive words
            candidates = find_candidates_covering(exclusive_words, max_candidates=50)
            
            for cand in candidates:
                if time.time() > deadline - 15.0:
                    break
                if try_swap(i, cand):
                    improved = True
                    break

        # Phase 3: Random swaps for diversity
        if time.time() < deadline - 15.0:
            num_random_tries = min(100, total_words // 10)
            for _ in range(num_random_tries):
                if time.time() > deadline - 15.0:
                    break
                i = random.randint(0, len(codeword_indices) - 1)
                old_idx = codeword_indices[i]
                
                exclusive_words = []
                for ni in neighbor_indices[old_idx]:
                    if coverage[ni] == 1:
                        exclusive_words.append(ni)
                
                if not exclusive_words:
                    continue
                
                # Try a few random candidates
                for _ in range(5):
                    if time.time() > deadline - 15.0:
                        break
                    cand = random.randint(0, total_words - 1)
                    if cand in in_code:
                        continue
                    if try_swap(i, cand):
                        improved = True
                        break

    # Final cleanup pass
    if time.time() < deadline - 5.0:
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            try_remove(i)

    # Convert back to lists
    result = [list(all_words[idx]) for idx in codeword_indices]

    return result
# EVOLVE-BLOCK-END
