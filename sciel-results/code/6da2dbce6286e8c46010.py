"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        all_words = list(itertools.product(range(q), repeat=n))
        uncovered = set(all_words)
        code = []
        while uncovered:
            best_word = None
            best_count = -1
            for w in all_words:
                count = sum(1 for u in uncovered if sum(a != b for a, b in zip(w, u)) <= radius)
                if count > best_count:
                    best_count = count
                    best_word = w
            code.append(list(best_word))
            uncovered = {u for u in uncovered if sum(a != b for a, b in zip(best_word, u)) > radius}
        return code

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods as bitsets (Python integers)
    masks = [0] * total_words
    for i in indices:
        w = idx_to_word(i)
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                nb_idx = word_to_idx(new_w)
                mask |= (1 << nb_idx)
        masks[i] = mask

    target_mask = (1 << total_words) - 1

    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    # Phase 1: Greedy Initialization
    uncovered_mask = target_mask
    code_indices = []
    
    while uncovered_mask:
        best_idx = -1
        best_count = -1
        for i in indices:
            count = popcount(masks[i] & uncovered_mask)
            if count > best_count:
                best_count = count
                best_idx = i
        
        code_indices.append(best_idx)
        uncovered_mask &= ~masks[best_idx]
        
    current_code = set(code_indices)
    current_mask = 0
    for c in current_code:
        current_mask |= masks[c]
        
    # Phase 2: Simulated Annealing
    start_time = time.time()
    time_limit = 110
    random.seed(42)
    
    best_code = set(current_code)
    best_size = len(current_code)
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    def get_score(code_set, mask):
        size = len(code_set)
        uncovered = popcount(target_mask & ~mask)
        return size + 100.0 * uncovered

    current_score = get_score(current_code, current_mask)
    best_score = current_score
    
    while time.time() - start_time < time_limit and T > min_T:
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to remove a random codeword
            if not current_code:
                T *= cooling_rate
                continue
            c_out = random.choice(list(current_code))
            new_mask = current_mask & ~masks[c_out]
            new_code = current_code.copy()
            new_code.remove(c_out)
            new_score = get_score(new_code, new_mask)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_code = new_code
                current_mask = new_mask
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_code = current_code.copy()
                    
        elif move_type < 0.7:
            # Try to swap
            if not current_code:
                T *= cooling_rate
                continue
            c_out = random.choice(list(current_code))
            c_in = random.choice(indices)
            if c_in in current_code:
                T *= cooling_rate
                continue
                
            new_mask = (current_mask & ~masks[c_out]) | masks[c_in]
            new_code = current_code.copy()
            new_code.remove(c_out)
            new_code.add(c_in)
            new_score = get_score(new_code, new_mask)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_code = new_code
                current_mask = new_mask
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_code = current_code.copy()
                    
        else:
            # Try to add a random codeword
            c_in = random.choice(indices)
            if c_in in current_code:
                T *= cooling_rate
                continue
                
            new_mask = current_mask | masks[c_in]
            new_code = current_code.copy()
            new_code.add(c_in)
            new_score = get_score(new_code, new_mask)
            
            delta = new_score - current_score
            if delta < 0 or (random.random() < math.exp(-delta / T) and T > 1.0):
                current_code = new_code
                current_mask = new_mask
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_code = current_code.copy()
        
        T *= cooling_rate
        
        # Periodic local search (greedy reduction) to clean up
        if random.random() < 0.05:
            changed = True
            while changed:
                changed = False
                for c in list(current_code):
                    new_mask = current_mask & ~masks[c]
                    new_uncovered = popcount(target_mask & ~new_mask)
                    if new_uncovered == 0:
                        current_code.remove(c)
                        current_mask = new_mask
                        current_score = get_score(current_code, current_mask)
                        if current_score < best_score:
                            best_score = current_score
                            best_code = current_code.copy()
                        changed = True
                        break

    # Final cleanup
    final_mask = 0
    for c in best_code:
        final_mask |= masks[c]
        
    if popcount(target_mask & ~final_mask) != 0:
        if popcount(target_mask & ~current_mask) == 0:
            best_code = current_code
        else:
            best_code = set(code_indices)

    return [list(idx_to_word(i)) for i in best_code]

# EVOLVE-BLOCK-END
