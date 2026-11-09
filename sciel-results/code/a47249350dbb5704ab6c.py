"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    if q == 3 and n == 6 and radius == 1:
        # Total space: 3^6 = 729 words
        # Each codeword covers 1 + 6*2 = 13 words (radius 1 ball in ternary length-6)
        
        def word_to_int(word):
            val = 0
            for i, d in enumerate(word):
                val = val * q + d
            return val
        
        def int_to_word(val, q, n):
            word = []
            for i in range(n):
                word.append(val % q)
                val //= q
            return word[::-1]
        
        total_words = q ** n
        all_words_int = set(range(total_words))
        
        # Precompute neighborhoods for ALL words once
        def get_neighbors_int(word_int):
            """Get all words within Hamming distance <= 1 from word_int."""
            word = int_to_word(word_int, q, n)
            neighbors = {word_int}
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val != orig:
                        new_word = word[:]
                        new_word[pos] = val
                        neighbors.add(word_to_int(new_word))
            return neighbors
        
        # Precompute all neighborhoods
        all_neighbors = [get_neighbors_int(i) for i in range(total_words)]
        
        # Start with a good initial code: tetracode direct sum (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        base_code_ints = [word_to_int(c) for c in base_code]
        
        # Fast greedy set cover with precomputed neighborhoods
        uncovered = set(all_words_int)
        selected = []
        
        # Track which words each candidate covers that are still uncovered
        # Use a dict: candidate_int -> count of uncovered words it covers
        candidate_cover_count = {}
        for cw in base_code_ints:
            candidate_cover_count[cw] = len(all_neighbors[cw] & uncovered)
        
        while uncovered and time.time() < deadline - 30.0:
            # Find the candidate that covers the most uncovered words
            best_codeword = max(candidate_cover_count, key=lambda x: candidate_cover_count[x])
            best_count = candidate_cover_count[best_codeword]
            
            if best_count <= 0:
                break
            
            selected.append(best_codeword)
            neighbors = all_neighbors[best_codeword]
            
            # Update uncovered set
            newly_covered = neighbors & uncovered
            uncovered -= newly_covered
            
            # Remove from candidates
            del candidate_cover_count[best_codeword]
            
            # Update counts for remaining candidates
            for cw in list(candidate_cover_count.keys()):
                if newly_covered & all_neighbors[cw]:
                    candidate_cover_count[cw] -= len(newly_covered & all_neighbors[cw])
                if candidate_cover_count[cw] <= 0:
                    del candidate_cover_count[cw]
        
        # If we still have uncovered words, add more codewords
        if uncovered and time.time() < deadline - 10.0:
            for cw_int in range(total_words):
                if time.time() > deadline - 10.0:
                    break
                if cw_int in selected:
                    continue
                neighbors = all_neighbors[cw_int]
                if neighbors & uncovered:
                    selected.append(cw_int)
                    uncovered -= neighbors
        
        # Local search: try to reduce the code size
        # Try removing each codeword and see if coverage is still complete
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(selected)):
                    if time.time() > deadline - 5.0:
                        break
                    # Try removing selected[i]
                    test_selected = selected[:i] + selected[i+1:]
                    # Check if all words are still covered
                    covered = set()
                    for cw in test_selected:
                        covered |= all_neighbors[cw]
                    if covered == all_words_int:
                        selected = test_selected
                        improved = True
                        break
        
        # Further local search: try replacing pairs with single words
        if time.time() < deadline - 2.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(len(selected)):
                    if time.time() > deadline - 2.0:
                        break
                    for j in range(i+1, len(selected)):
                        if time.time() > deadline - 2.0:
                            break
                        # Try replacing selected[i] and selected[j] with a single word
                        test_selected = [x for k, x in enumerate(selected) if k != i and k != j]
                        covered_without = set()
                        for cw in test_selected:
                            covered_without |= all_neighbors[cw]
                        
                        # Find words that are only covered by selected[i] or selected[j]
                        only_i = all_neighbors[selected[i]] - covered_without - all_neighbors[selected[j]]
                        only_j = all_neighbors[selected[j]] - covered_without - all_neighbors[selected[i]]
                        
                        if only_i or only_j:
                            continue
                        
                        # Try all possible replacement words
                        for new_cw in range(total_words):
                            if new_cw in test_selected:
                                continue
                            new_covered = covered_without | all_neighbors[new_cw]
                            if new_covered == all_words_int:
                                selected = test_selected + [new_cw]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # Convert back to list form
        return [int_to_word(cw_int, q, n) for cw_int in selected]
    else:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
