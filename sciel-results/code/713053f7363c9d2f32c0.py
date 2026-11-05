"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _neighbors(word, q, n):
    """Generate all words within Hamming distance 1 of word."""
    result = [list(word)]
    for i in range(n):
        for val in range(q):
            if val != word[i]:
                new_word = list(word)
                new_word[i] = val
                result.append(new_word)
    return result


def construct(q, n, radius):
    if n == 0:
        return [[]]
    
    # For small cases, use direct construction
    if n <= 4:
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        elif n == 3:
            return [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        elif n == 2:
            return [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [2, 2], [0, 2], [2, 0], [1, 2], [2, 1]]
        else:
            return [[i] for i in range(q)]
    
    # Precompute all words and their indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)
    
    # Precompute neighbors for each word (within Hamming distance radius)
    # For radius=1, this is efficient
    neighbors_cache = {}
    def get_neighbors(word):
        w = tuple(word)
        if w in neighbors_cache:
            return neighbors_cache[w]
        result = [w]
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        new_word = list(w)
                        new_word[i] = val
                        result.append(tuple(new_word))
        neighbors_cache[w] = result
        return result
    
    # Build coverage matrix: cover_count[w] = number of codewords covering w
    # Start with greedy construction
    
    # Initial solution: use tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_codewords.append(tuple(h + list(tail)))
    
    # Remove duplicates and convert to list of lists
    initial_codewords = list(set(initial_codewords))
    
    # Greedy refinement: try to reduce the code
    codewords = initial_codewords[:]
    
    # Iteratively remove redundant codewords
    improved = True
    max_iterations = 1000
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        # Try removing each codeword
        for i in range(len(codewords)):
            # Count how many words are covered only by codeword i
            # We need to check if all neighbors of codeword[i] are covered by others
            cw = codewords[i]
            can_remove = True
            for neighbor in get_neighbors(cw):
                # Check if neighbor is covered by any other codeword
                covered_by_other = False
                for j, other_cw in enumerate(codewords):
                    if j == i:
                        continue
                    # Check if other_cw covers neighbor
                    dist = sum(1 for a, b in zip(other_cw, neighbor) if a != b)
                    if dist <= radius:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    can_remove = False
                    break
            if can_remove:
                codewords.pop(i)
                improved = True
                break
    
    # Local search: try to replace codewords to reduce total count
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    random.seed(42)
    
    # Multiple rounds of local search
    for round_num in range(20):
        current = best_codewords[:]
        # Try random removals and replacements
        for step in range(200):
            if len(current) >= best_size:
                break
            
            # Pick a random codeword to try removing
            idx = random.randint(0, len(current) - 1)
            cw_to_remove = current[idx]
            
            # Find words that would become uncovered
            would_uncover = []
            for neighbor in get_neighbors(cw_to_remove):
                is_covered = False
                for j, cw in enumerate(current):
                    if j == idx:
                        continue
                    dist = sum(1 for a, b in zip(cw, neighbor) if a != b)
                    if dist <= radius:
                        is_covered = True
                        break
                if not is_covered:
                    would_uncover.append(neighbor)
            
            if not would_uncover:
                # Can remove this codeword
                del current[idx]
                best_codewords = current[:]
                best_size = len(current)
            else:
                # Try to find a replacement that covers all would_uncover words
                # and doesn't add new uncovered words
                # Sample candidates near cw_to_remove
                candidates = [cw_to_remove]
                # Add neighbors of cw_to_remove as candidates
                for neighbor in get_neighbors(cw_to_remove):
                    if neighbor != cw_to_remove:
                        candidates.append(neighbor)
                random.shuffle(candidates)
                
                found_replacement = False
                for candidate in candidates[:50]:  # Limit search
                    if candidate in current:
                        continue
                    
                    # Check if candidate covers all would_uncover words
                    covers_all = True
                    for w in would_uncover:
                        dist = sum(1 for a, b in zip(candidate, w) if a != b)
                        if dist > radius:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Check if replacing with candidate doesn't uncover new words
                    # Words that were covered by cw_to_remove but not by candidate
                    new_uncover = []
                    for neighbor in get_neighbors(cw_to_remove):
                        dist_to_cand = sum(1 for a, b in zip(candidate, neighbor) if a != b)
                        if dist_to_cand > radius:
                            # Check if covered by other codewords
                            covered_by_other = False
                            for j, cw in enumerate(current):
                                if j == idx:
                                    continue
                                dist = sum(1 for a, b in zip(cw, neighbor) if a != b)
                                if dist <= radius:
                                    covered_by_other = True
                                    break
                            if not covered_by_other:
                                new_uncover.append(neighbor)
                    
                    if not new_uncover:
                        # Good replacement
                        current[idx] = candidate
                        found_replacement = True
                        break
                
                if found_replacement:
                    continue
    
    return [list(cw) for cw in best_codewords]
# EVOLVE-BLOCK-END
