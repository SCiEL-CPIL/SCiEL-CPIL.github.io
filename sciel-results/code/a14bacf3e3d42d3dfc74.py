"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s limit
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction for other parameters
        if q == 3 and n >= 4:
            tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
            tails = np.array(list(itertools.product(range(3), repeat=n - 4)))
            code = np.tile(tetracode[:, None, :], (1, len(tails), 1))
            code = code + tails[None, :, :]
            return code.reshape(-1, n).tolist()
        else:
            # Very simple fallback
            return [[0] * n]
    
    # Precompute: for each position and value, the mask of words that have that value at that position
    # Total words in space: 3^6 = 729
    total_words = 3 ** n
    
    # Represent each word as an integer 0..728
    def word_to_int(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx
    
    def int_to_word(idx):
        word = []
        for i in range(n):
            word.append(idx % 3)
            idx //= 3
        return word[::-1]
    
    # Generate all words as integers
    all_words = list(range(total_words))
    
    # Precompute neighbors: for each word, the set of words within distance 1 (including itself)
    # A word covers itself + all words differing in exactly one position
    # For position i, changing value v to v+1 or v+2
    neighbors = []
    for w in all_words:
        word = int_to_word(w)
        covered = {w}
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    word[pos] = new_val
                    covered.add(word_to_int(word))
                    word[pos] = orig
        neighbors.append(covered)
    
    # Initialize with tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    code_ints = set(word_to_int(w) for w in initial_code)
    
    def is_removable(cand_idx, current_code_set, neighbors):
        """Check if removing cand_idx still covers all words."""
        # For each word covered by cand_idx, check if it's covered by another codeword
        for covered_word in neighbors[cand_idx]:
            # Check if any other codeword covers this word
            found = False
            for cw in current_code_set:
                if cw == cand_idx:
                    continue
                if covered_word in neighbors[cw]:
                    found = True
                    break
            if not found:
                return False
        return True
    
    def try_remove_redundant(code_set, neighbors, deadline):
        """Try to remove redundant codewords. Returns new code set."""
        code_list = list(code_set)
        random.shuffle(code_list)
        changed = True
        while changed:
            changed = False
            for cand in code_list[:]:
                if time.time() > deadline:
                    return code_set
                if cand not in code_set:
                    continue
                # Check if removable
                removable = True
                for covered_word in neighbors[cand]:
                    # Check if another codeword covers this word
                    covered_by_other = False
                    for cw in code_set:
                        if cw == cand:
                            continue
                        if covered_word in neighbors[cw]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        removable = False
                        break
                if removable:
                    code_set.discard(cand)
                    changed = True
        return code_set
    
    best_code = set(code_ints)
    best_size = len(best_code)
    
    # Multiple reduction passes with different orderings
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        current_code = set(code_ints)
        
        # Try removal with random ordering
        current_code = try_remove_redundant(current_code, neighbors, deadline)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)
        
        # If we reach 73, we're at the best known - stop
        if best_size <= 73:
            break
        
        # ILS-style perturbation: add a random word, then reduce
        if time.time() > deadline - 5:
            break
        
        # Add 1-3 random words not in current code
        all_word_list = list(all_words)
        random.shuffle(all_word_list)
        for aw in all_word_list:
            if aw not in current_code:
                current_code.add(aw)
                break
        for aw in all_word_list:
            if aw not in current_code and random.random() < 0.5:
                current_code.add(aw)
                break
        
        # Reduce again
        current_code = try_remove_redundant(current_code, neighbors, deadline)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = set(current_code)
        
        if best_size <= 73:
            break
    
    # Convert back to list of words
    result = [int_to_word(idx) for idx in best_code]
    random.shuffle(result)
    return result

# EVOLVE-BLOCK-END
