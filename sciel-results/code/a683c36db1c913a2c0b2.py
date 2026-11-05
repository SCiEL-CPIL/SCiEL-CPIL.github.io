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
    
    if n == 6 and q == 3 and radius == 1:
        # Start with the 81-word direct sum of Tetracode and full length-2 space
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
        
        # Fast local search to reduce size
        # Represent codewords as tuples for hashing
        codewords = [tuple(c) for c in initial_codewords]
        
        # Precompute all words and their neighbors for fast lookup
        all_words = list(itertools.product(range(3), repeat=6))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Coverage array: count of codewords covering each word
        covered_count = np.zeros(len(all_words), dtype=np.int8)
        
        def add_cw(cw):
            for nb in _neighbors(cw, 3, 6):
                idx = word_to_idx[tuple(nb)]
                covered_count[idx] += 1
        
        def remove_cw(cw):
            for nb in _neighbors(cw, 3, 6):
                idx = word_to_idx[tuple(nb)]
                covered_count[idx] -= 1
        
        for cw in codewords:
            add_cw(cw)
            
        # Local Search
        random.seed(42)
        improved = True
        while improved:
            improved = False
            # Try to remove a random codeword if possible
            for _ in range(1000):
                if not codewords:
                    break
                idx = random.randint(0, len(codewords) - 1)
                cw = codewords[idx]
                # Check if removal leaves any word uncovered
                can_remove = True
                for nb in _neighbors(cw, 3, 6):
                    idx_nb = word_to_idx[tuple(nb)]
                    if covered_count[idx_nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    remove_cw(cw)
                    del codewords[idx]
                    improved = True
                    break
        
        # If stuck, try swaps
        # Try to replace a codeword with another to enable a removal
        # This is a 2-opt style move
        for _ in range(5000):
            if len(codewords) < 73:
                break
            idx1 = random.randint(0, len(codewords) - 1)
            cw1 = codewords[idx1]
            
            # Find a word in cw1's neighborhood that is covered only by cw1
            # If none, cw1 is redundant (handled above)
            # We want to move cw1 to a new position cw_new such that
            # 1. cw_new covers the words that cw1 uniquely covered.
            # 2. This allows removal of another codeword cw2.
            
            # Simple heuristic: Try all possible cw_new? Too many.
            # Try random cw_new.
            cw_new = tuple(random.randint(0, 2) for _ in range(6))
            if cw_new == cw1:
                continue
            if cw_new in codewords:
                continue
            
            # Simulate move
            # Remove cw1
            remove_cw(cw1)
            # Add cw_new
            add_cw(cw_new)
            
            # Check if all covered
            valid = True
            for i in range(len(all_words)):
                if covered_count[i] == 0:
                    valid = False
                    break
            
            if valid:
                # Try to remove another codeword
                removed_other = False
                for idx2 in range(len(codewords)):
                    if idx2 == idx1:
                        continue
                    cw2 = codewords[idx2]
                    can_remove = True
                    for nb in _neighbors(cw2, 3, 6):
                        idx_nb = word_to_idx[tuple(nb)]
                        if covered_count[idx_nb] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        remove_cw(cw2)
                        del codewords[idx2]
                        removed_other = True
                        break
                
                if removed_other:
                    codewords[idx1] = cw_new
                    improved = True
                    break
                else:
                    # Revert
                    remove_cw(cw_new)
                    add_cw(cw1)
            else:
                # Revert
                remove_cw(cw_new)
                add_cw(cw1)
                
        return [list(c) for c in codewords]

    # Fallback for other cases
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    covered = [False] * len(all_words)
    codewords = []
    
    for cw in initial_codewords:
        codewords.append(cw)
        for neighbor in _neighbors(cw, q, n):
            idx = word_to_idx[tuple(neighbor)]
            covered[idx] = True
            
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            test_covered = covered[:]
            for neighbor in _neighbors(codewords[i], q, n):
                idx = word_to_idx[tuple(neighbor)]
                still_covered = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if _hamming_dist(codewords[j], all_words[idx]) <= radius:
                        still_covered = True
                        break
                if not still_covered:
                    break
            else:
                del codewords[i]
                covered = [False] * len(all_words)
                for cw in codewords:
                    for neighbor in _neighbors(cw, q, n):
                        idx = word_to_idx[tuple(neighbor)]
                        covered[idx] = True
                improved = True
                break
                
    return codewords
# EVOLVE-BLOCK-END
