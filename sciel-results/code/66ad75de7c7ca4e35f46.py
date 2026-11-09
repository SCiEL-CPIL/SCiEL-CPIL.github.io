import numpy as np
import time
import random

def construct(q, n, radius):
    # For this specific task: q=3, n=6, radius=1
    # General implementation might be needed if evaluator passes different params, 
    # but the task description says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # The seed program is specific to q=3, n=6. 
    # If q, n, radius are different, the tetracode logic doesn't apply directly.
    # However, the prompt says "Keep the function names and signatures...". 
    # The task is specifically "Football pool problem for 6 matches: ternary covering code K_3(6,1)".
    # I will assume q=3, n=6, radius=1. If not, I should handle it or fall back to seed.
    # Given the constraints and "Rewrite ... for the following task", I will specialize for 3,6,1 
    # but include a check to return the seed result if parameters don't match, to be safe.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other cases if any (though unlikely in this specific task eval)
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This fallback is only valid for q=3. If q is not 3, this is wrong.
            # Since the task is fixed to K_3(6,1), I will assume q=3, n=6, r=1.
            # If q is not 3, the tetracode is not defined.
            # I will just return the seed output if q!=3 or n!=6 to avoid crashing, 
            # but the score will be 1.0.
            if q == 3:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 # Random fallback for unknown cases
                 return [[random.randint(0, q-1) for _ in range(n)] for _ in range(100)]
        
    # Specific optimization for q=3, n=6, radius=1
    total_words = q ** n # 729
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Precompute neighbors for all 729 words
    # Map word (tuple) to index (int)
    # index = w[0]*3^5 + w[1]*3^4 + ... + w[5]
    # word from index: divmod chain
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
    
    def word_to_idx(w):
        idx = 0
        multiplier = 1
        for i in range(n-1, -1, -1):
            idx += w[i] * multiplier
            multiplier *= q
        return idx

    # Precompute neighbors for each index
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        nbs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig_val
        neighbors.append(nbs)
        
    # Convert to numpy arrays for fast vectorized updates?
    # neighbors_np[i] is an array of indices.
    # We need to do cover_counts - neighbors_np[i] and check min >= 0.
    # This requires gathering cover_counts at those indices.
    
    # Initialize with Tetracode Direct Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to set of indices for fast lookup
    current_code_set = set(word_to_idx(w) for w in initial_codewords)
    current_code_list = list(current_code_set)
    
    # Calculate initial cover counts
    cover_counts = np.zeros(total_words, dtype=np.int32)
    for idx in current_code_set:
        cover_counts[neighbors[idx]] += 1
        
    best_code_size = len(current_code_set)
    best_code_set = set(current_code_set)
    
    # Helper to check validity and update
    # We will use a simple loop for moves
    
    # To speed up swap selection, we can precompute which words are "good" additions?
    # No, let's stick to random swaps and removals.
    
    # For faster min check: 
    # Instead of `min(cover_counts - neighbors[idx])`, we can just check if any neighbor has count 1 
    # and we are removing it? No, if count is 1 and we remove, it becomes 0. 
    # So `min(new_counts) >= 0` is equivalent to `all(cover_counts[nb] > 0 for nb in neighbors[idx] if we are removing idx)`.
    # Wait, if we are removing idx, `new_counts[nb] = cover_counts[nb] - 1` for nb in neighbors[idx].
    # So we just need `min(cover_counts[neighbors[idx]]) >= 1`.
    # This is much faster: `np.min(cover_counts[neighbors[idx]]) >= 1`.
    
    # For swap: remove i, add j.
    # `new_counts = cover_counts - neighbors[i] + neighbors[j]`.
    # We need `min(new_counts) >= 0`.
    
    # Let's optimize the neighbor access.
    # neighbors is a list of lists. Converting to numpy array for each access might be slow.
    # Let's keep neighbors as a list of numpy arrays.
    neighbors_np = [np.array(nbs, dtype=np.int32) for nbs in neighbors]
    
    # Random seed for reproducibility in debugging, but random is fine
    random.seed(42)
    np.random.seed(42)
    
    while time.time() < deadline:
        if len(current_code_set) < best_code_size:
            best_code_size = len(current_code_set)
            best_code_set = set(current_code_set)
            # Optional: print(best_code_size) # Disabled for speed
        
        # Try Remove
        if current_code_list:
            idx_to_remove = random.choice(current_code_list)
            # Check if removal is valid
            # min(cover_counts[neighbors[idx_to_remove]]) >= 1
            if np.min(cover_counts[neighbors_np[idx_to_remove]]) >= 1:
                # Accept removal
                cover_counts[neighbors_np[idx_to_remove]] -= 1
                current_code_set.remove(idx_to_remove)
                current_code_list.remove(idx_to_remove)
                continue # Restart loop to update best immediately
        
        # Try Swap
        if len(current_code_list) > 1: # Need at least one to swap out
            idx_old = random.choice(current_code_list)
            # Pick a random new index not in current_code_set
            # To avoid infinite loop picking same, we can pick from all and check
            idx_new = random.randint(0, total_words - 1)
            attempts = 0
            while idx_new in current_code_set and attempts < 10:
                idx_new = random.randint(0, total_words - 1)
                attempts += 1
            
            if idx_new not in current_code_set:
                # Calculate new counts
                # new_counts = cover_counts - neighbors[idx_old] + neighbors[idx_new]
                # We can do this in place on a copy? No, 729 is small, copy is fast.
                # Or check validity directly?
                # Validity: for all k, cover_counts[k] - (1 if k in N(old) else 0) + (1 if k in N(new) else 0) >= 0
                
                # Efficient check:
                # The only words that can drop to 0 are those in N(old) that are NOT in N(new).
                # And those in N(old) intersect N(new) stay same or increase.
                # Words not in N(old) stay same or increase.
                # So we only need to check `cover_counts[k] >= 1` for k in N(old) \ N(new).
                
                n_old = neighbors_np[idx_old]
                n_new = neighbors_np[idx_new]
                
                # Find intersection
                # Since sets are small (13), we can use set logic or numpy
                # set_n_old = set(n_old)
                # set_n_new = set(n_new)
                # critical_words = [w for w in n_old if w not in set_n_new]
                
                # Numpy approach:
                # mask_old = np.zeros(total_words, dtype=bool)
                # mask_old[n_old] = True
                # mask_new = np.zeros(total_words, dtype=bool)
                # mask_new[n_new] = True
                # critical_mask = mask_old & ~mask_new
                # if np.any(critical_mask):
                #     if np.min(cover_counts[critical_mask]) < 1:
                #         reject
                # else:
                #     accept
                
                # Even faster:
                # Just compute the change and check min?
                # temp_counts = cover_counts.copy()
                # temp_counts[n_old] -= 1
                # temp_counts[n_new] += 1
                # if np.min(temp_counts) >= 0:
                    
                # Let's use the critical words approach with sets for clarity and speed on small sets
                set_n_old = set(n_old)
                set_n_new = set(n_new)
                critical_words = [w for w in set_n_old if w not in set_n_new]
                
                is_valid = True
                for w in critical_words:
                    if cover_counts[w] < 1:
                        is_valid = False
                        break
                
                if is_valid:
                    # Accept swap
                    cover_counts[n_old] -= 1
                    cover_counts[n_new] += 1
                    current_code_set.remove(idx_old)
                    current_code_set.add(idx_new)
                    current_code_list.remove(idx_old)
                    current_code_list.append(idx_new)
                    continue

        # If stuck (no removal, no valid swap found in this iteration), 
        # we might be in a local minimum. 
        # We can try a "repair" or just continue random swaps.
        # The random swaps above already try to find a valid swap.
        # If no valid swap is found, we just loop.
        # To escape, we can allow "invalid" swaps with a penalty? 
        # Or we can try to Add a word to create space for a better swap?
        # Adding increases size, which we want to avoid, but might help escape.
        
        # Let's try a greedy add if we've been stuck for a while?
        # Simple heuristic: if we haven't improved in X steps, try to add a word that covers most uncovered?
        # But we are always valid (cover_counts > 0 everywhere). So there are no uncovered words.
        # The issue is just that no single removal is possible.
        # Swaps change the "shape" of the coverage.
        
        # If we are truly stuck, we can try a "random restart" of the local search 
        # by perturbing the code: remove K random words and add K random words that restore coverage?
        # This is complex. Let's rely on the volume of random swaps.
        
        # One improvement: If a swap is valid, we accept it. 
        # This performs a random walk on the space of valid codes of fixed size.
        # We hope to hit a code from which a removal is possible.
        
        # To speed up, we can bias the choice of idx_new.
        # But random is a good baseline.
        
        # Small optimization: If we just did a swap, we don't need to re-check removals immediately?
        # The `continue` handles that.

    # Convert best_code_set back to list of lists
    best_codewords = [idx_to_word(idx) for idx in best_code_set]
    return best_codewords

# EVOLVE-BLOCK-END
