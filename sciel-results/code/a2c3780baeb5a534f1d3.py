"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=6, radius=1 using local search with coverage counts.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _generic_greedy(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    
    # Map words to indices 0..728
    # idx = w[0]*3^5 + w[1]*3^4 + ... + w[5]
    # Or just use itertools product index
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute neighbors as lists of indices
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        nb = [i] # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                nb.append(word_to_idx[new_w])
        neighbors[i] = nb

    # Start with tetracode-based construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = tuple(h + list(tail))
            initial_code_indices.add(word_to_idx[word])
    
    code_set = set(initial_code_indices)
    
    # cover_count[i] = number of codewords within distance 1 of word i
    cover_count = np.zeros(total_words, dtype=int)
    
    def add_codeword(c_idx):
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1
        code_set.add(c_idx)
        
    def remove_codeword(c_idx):
        for nb in neighbors[c_idx]:
            cover_count[nb] -= 1
        code_set.remove(c_idx)
        
    def is_valid():
        return np.min(cover_count) >= 1

    # Initialize cover_count
    for c in code_set:
        for nb in neighbors[c]:
            cover_count[nb] += 1

    # Local Search
    # We want to minimize len(code_set) while maintaining is_valid()
    
    # Phase 1: Greedy Removal
    # Iterate until no more removals are possible
    while True:
        removed = False
        # Shuffle to break symmetry? Not strictly necessary for removal, but good for swaps.
        for c in list(code_set):
            # Try removing c
            for nb in neighbors[c]:
                cover_count[nb] -= 1
            
            if is_valid():
                # Keep it removed
                code_set.remove(c)
                removed = True
            else:
                # Revert
                for nb in neighbors[c]:
                    cover_count[nb] += 1
            
            if removed:
                # Restart the loop to check for further removals
                break
        
        if not removed:
            break

    # Phase 2: Swap to escape local minima
    # If we can't remove, try swapping one codeword with another to create a "hole" that allows removal.
    # Strategy: Try all possible swaps (c_old, c_new) where c_old in code, c_new not in code.
    # If swap is valid, accept it. Then try removal again.
    # To avoid infinite loops or getting stuck, we limit the number of swap attempts or use a random restart strategy if stuck.
    # Given the small size, we can try a few rounds of "Best First" or "Random" swaps.
    
    max_swap_rounds = 100
    stuck_count = 0
    
    for _ in range(max_swap_rounds):
        # Try to remove again (in case a previous swap enabled a removal that we missed? 
        # Actually, we break out of removal loop only when no removal is possible.
        # So we are in a state where no single removal is valid.
        
        # Try to find a valid swap
        swap_made = False
        c_old_list = list(code_set)
        random.shuffle(c_old_list) # Randomize order to escape deterministic traps
        
        for c_old in c_old_list:
            # Try all c_new not in code
            # Optimization: Only try c_new that are "close" to c_old? Or just all?
            # 729 * 81 is ~60k checks. Each check is O(13) updates + O(729) min check.
            # 60k * 729 is ~43M ops. Might be too slow in Python if done naively.
            # We can optimize the validity check.
            
            # Instead of checking min(cover_count) >= 1 for every candidate,
            # we can note that removing c_old decreases counts for neighbors of c_old.
            # Adding c_new increases counts for neighbors of c_new.
            
            # Let's just do the full swap simulation for a subset of candidates to keep it fast,
            # or optimize the check.
            
            # Actually, let's just try a random subset of swaps if it's too slow.
            # But 81 * 729 is small enough if we are careful.
            
            # Let's try to find ANY valid swap.
            candidates = [i for i in range(total_words) if i not in code_set]
            random.shuffle(candidates)
            
            for c_new in candidates:
                # Simulate swap
                # 1. Remove c_old
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                # 2. Add c_new
                for nb in neighbors[c_new]:
                    cover_count[nb] += 1
                
                if is_valid():
                    # Accept swap
                    code_set.remove(c_old)
                    code_set.add(c_new)
                    swap_made = True
                    break
                else:
                    # Revert
                    for nb in neighbors[c_new]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c_old]:
                        cover_count[nb] += 1
                
            if swap_made:
                break
        
        if swap_made:
            # After a swap, try to remove again
            # We can integrate this by just continuing the outer loop,
            # but we need to check for removals first.
            # Let's re-run the removal phase.
            while True:
                removed = False
                for c in list(code_set):
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    if is_valid():
                        code_set.remove(c)
                        removed = True
                    else:
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
                    if removed:
                        break
                if not removed:
                    break
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > 5:
                # Stuck in a local minimum that requires multi-swap or is a dead end for this heuristic.
                # For this problem, 73 is the target. If we are at 74 or 75, we might be stuck.
                # We can try a random restart from a slightly different initial state?
                # Or just break and return what we have.
                break

    result = [list(idx_to_word[i]) for i in code_set]
    return result

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = q ** n
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        nb = [i]
        # Generate neighbors within radius
        # For small n, brute force distance check is fine
        for j, v in enumerate(all_words):
            if i == j: continue
            dist = sum(1 for k in range(n) if w[k] != v[k])
            if dist <= radius:
                nb.append(j)
        neighbors[i] = nb
        
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w_idx in range(total_words):
            if w_idx in code:
                continue
            # Count how many uncovered words this covers
            covered_count = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered:
                    covered_count += 1
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w_idx
                
        if best_word is None:
            break
            
        code.append(best_word)
        for nb in neighbors[best_word]:
            uncovered.discard(nb)
            
    return [list(idx_to_word[i]) for i in code]

# EVOLVE-BLOCK-END
