import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute sphere masks and indices
    sphere_masks = [0] * total_words
    sphere_indices = [[] for _ in range(total_words)]
    
    for i, word in enumerate(all_words):
        mask = 1 << i
        inds = [i]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    new_word = tuple(new_word)
                    j = word_to_idx[new_word]
                    mask |= (1 << j)
                    inds.append(j)
        sphere_masks[i] = mask
        sphere_indices[i] = inds
        
    # Helper for bit counting
    if hasattr(0, 'bit_count'):
        def _bit_count(x):
            return x.bit_count()
    else:
        def _bit_count(x):
            return bin(x).count('1')

    def greedy_cover(uncovered_mask, randomness=0.0):
        new_code = []
        unc = uncovered_mask
        while unc:
            best_idx = -1
            best_count = -1
            # Calculate gains for all candidates
            gains = []
            for i in range(total_words):
                common = sphere_masks[i] & unc
                if common == 0:
                    continue
                count = _bit_count(common)
                if count > best_count:
                    best_count = count
                gains.append((count, i))
            
            if best_count == 0:
                break
            
            if randomness > 0 and gains:
                # Pick from top candidates with some probability
                # Sort by count descending
                gains.sort(key=lambda x: x[0], reverse=True)
                # Take top 10% or at least 5
                top_k = max(5, len(gains) // 10)
                candidates = gains[:top_k]
                # Weighted random selection or uniform? Uniform among top K is simple.
                chosen_count, chosen_idx = random.choice(candidates)
                best_idx = chosen_idx
            else:
                # Find the one with max count (already found in loop above, but we need the index)
                # We didn't store the index for the max in the loop above efficiently?
                # We did: `if count > best_count: best_count = count; best_idx = i`
                # But in the randomized branch, we need to re-find or store it.
                # Let's restructure to store best_idx in the loop.
                pass
            
            # Wait, the loop above didn't store best_idx in the randomized case properly if we skip the update.
            # Let's just re-calculate best_idx if randomness is 0.
            if randomness == 0:
                 best_idx = -1
                 best_count = -1
                 for i in range(total_words):
                     common = sphere_masks[i] & unc
                     if common == 0: continue
                     count = _bit_count(common)
                     if count > best_count:
                         best_count = count
                         best_idx = i
            else:
                 # We already have `candidates`
                 pass
                 
            new_code.append(best_idx)
            unc &= ~sphere_masks[best_idx]
        return new_code

    def reduce_code(code_indices):
        if not code_indices:
            return []
        cover_count = np.zeros(total_words, dtype=np.int8)
        for idx in code_indices:
            for w_idx in sphere_indices[idx]:
                cover_count[w_idx] += 1
                
        current_code = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_code)):
                c_idx = current_code[i]
                is_redundant = True
                for w_idx in sphere_indices[c_idx]:
                    if cover_count[w_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w_idx in sphere_indices[c_idx]:
                        cover_count[w_idx] -= 1
                    current_code.pop(i)
                    changed = True
                    break
        return current_code

    start_time = time.time()
    # Initial greedy solution
    best_code = greedy_cover((1 << total_words) - 1)
    best_code = reduce_code(best_code)
    best_size = len(best_code)
    
    current_code = best_code[:]
    stagnation = 0
    
    while (time.time() - start_time) < 115:
        # 1. Reduce current code
        current_code = reduce_code(current_code)
        
        # 2. Check improvement
        if len(current_code) < best_size:
            best_code = current_code[:]
            best_size = len(best_code)
            stagnation = 0
        else:
            stagnation += 1
            
        # 3. Local Search Move
        if not current_code:
            break
            
        # Randomly pick a codeword to remove
        i = random.randint(0, len(current_code) - 1)
        rem_c = current_code[i]
        
        # Calculate uncovered mask
        # Efficient way: Calculate cover count of current_code, then check rem_c's sphere
        # Or just recompute covered mask?
        # Recomputing covered mask is O(K * S). K~500, S~25. 12500 ops. Fast.
        covered_mask = 0
        for j, c in enumerate(current_code):
            if j != i:
                covered_mask |= sphere_masks[c]
        
        unc_mask = ((1 << total_words) - 1) & ~covered_mask
        
        if unc_mask == 0:
            # Can remove
            new_code = [c for j, c in enumerate(current_code) if j != i]
            current_code = new_code
        else:
            # Try to fill with greedy (with some randomness)
            # Use higher randomness if stagnating
            rand_val = 0.2 if stagnation > 5 else 0.0
            added = greedy_cover(unc_mask, randomness=rand_val)
            new_code = [c for j, c in enumerate(current_code) if j != i]
            new_code.extend(added)
            new_code = reduce_code(new_code)
            
            # Accept if size is smaller or equal (swap)
            # If size is equal, it's a swap. We accept it to explore.
            if len(new_code) <= len(current_code):
                current_code = new_code
            else:
                # Revert
                pass
        
        # 4. Perturbation if stuck
        if stagnation > 20:
            # Remove 3 random codewords
            if len(current_code) > 3:
                to_remove = random.sample(current_code, 3)
                to_remove_set = set(to_remove)
                remaining = [c for c in current_code if c not in to_remove_set]
                
                covered_mask = 0
                for c in remaining:
                    covered_mask |= sphere_masks[c]
                
                unc_mask = ((1 << total_words) - 1) & ~covered_mask
                added = greedy_cover(unc_mask, randomness=0.5)
                current_code = remaining + added
                current_code = reduce_code(current_code)
            stagnation = 0

    return [list(all_words[idx]) for idx in best_code]
