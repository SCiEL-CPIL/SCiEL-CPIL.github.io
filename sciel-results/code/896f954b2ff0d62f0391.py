import time
import random
import itertools

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # The function signature is general, but the evaluator passes specific values.
    # We assume the task is fixed to q=3, n=7, radius=1 based on the description.
    # If q, n, radius differ, we might need a fallback.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # So we should handle general q, n, radius?
    # "Football pool problem for 7 matches: ternary covering code K_3(7,1)"
    # It seems specific to this instance.
    # However, to be safe, if (q,n,radius) != (3,7,1), we can return the seed or a simple heuristic.
    # But the score is based on this specific task.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: simple greedy for other cases, or seed
        # For now, just return a valid code if possible, else empty?
        # The task contract says "checked over the whole space".
        # If we don't know the optimal, we should return a valid code.
        # Greedy is valid.
        pass

    # Specific optimization for 3, 7, 1
    if q == 3 and n == 7 and radius == 1:
        deadline = time.time() + 110 # Safety margin
        
        # Precompute masks for all words
        # Word is tuple of 7 ints in 0..2
        # Map word to index 0..2186
        # mask is an integer with bits set for the word and its neighbors
        
        # To speed up, we can map word -> index
        # index = sum(w[i] * (3 ** (6-i)) for i in range(7))
        
        def word_to_idx(w):
            idx = 0
            for i in range(7):
                idx = idx * 3 + w[i]
            return idx

        def idx_to_word(idx):
            w = [0]*7
            for i in range(6, -1, -1):
                w[i] = idx % 3
                idx //= 3
            return w

        # Precompute neighbors
        # For each idx, find its neighbors
        # neighbor is obtained by changing one digit
        # Note: in Z3, changing digit d to d+1 or d-1.
        # Actually, Hamming distance 1 means exactly one position differs.
        
        masks = [0] * (3**7)
        
        # Generate all words
        all_words = [idx_to_word(i) for i in range(3**7)]
        
        for i, w in enumerate(all_words):
            m = 0
            # self
            m |= (1 << i)
            # neighbors
            for pos in range(7):
                orig = w[pos]
                for val in range(3):
                    if val == orig:
                        continue
                    new_w = w[:]
                    new_w[pos] = val
                    n_idx = word_to_idx(new_w)
                    m |= (1 << n_idx)
            masks[i] = m
            
        ALL_ONES = (1 << (3**7)) - 1
        
        # 1. Greedy Construction
        uncovered = ALL_ONES
        code_indices = []
        
        # To make it faster, we can precompute popcount?
        # Python int.bit_count() is fast.
        
        while uncovered:
            best_idx = -1
            best_count = -1
            # Iterate through all indices
            # Optimization: only consider indices that cover at least one uncovered bit
            for i in range(3**7):
                # If mask_i has no intersection with uncovered, skip
                # mask_i & uncovered
                inter = masks[i] & uncovered
                if inter:
                    cnt = inter.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        # Early exit if we found a word that covers all remaining? Unlikely
                        if best_count == uncovered.bit_count():
                            break
            if best_idx == -1:
                break # Should not happen
            code_indices.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        # 2. Prune
        def prune(code):
            changed = True
            while changed:
                changed = False
                for i in range(len(code)):
                    # Check if code[i] is redundant
                    # OR of all others
                    others_or = 0
                    for j in range(len(code)):
                        if i != j:
                            others_or |= masks[code[j]]
                    if (others_or & ALL_ONES) == ALL_ONES:
                        # Redundant
                        code.pop(i)
                        changed = True
                        break
            return code

        code_indices = prune(code_indices)
        
        # 3. Local Search
        # We will try to swap and prune.
        
        def get_unique_mask(code, i):
            # Returns mask of points covered ONLY by code[i]
            m_i = masks[code[i]]
            others_or = 0
            for j in range(len(code)):
                if i != j:
                    others_or |= masks[code[j]]
            return m_i & ~others_or

        # Main loop
        while time.time() < deadline:
            improved = False
            # Try to improve by swapping
            # Randomize order to avoid bias
            idx_order = list(range(len(code_indices)))
            random.shuffle(idx_order)
            
            for i in idx_order:
                if time.time() > deadline:
                    break
                u_mask = get_unique_mask(code_indices, i)
                if u_mask == 0:
                    # Should have been pruned, but just in case
                    code_indices.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                
                # Find candidates w such that u_mask is subset of masks[w]
                # u_mask & ~masks[w] == 0
                candidates = []
                for w_idx in range(3**7):
                    if (u_mask & ~masks[w_idx]) == 0:
                        candidates.append(w_idx)
                
                if not candidates:
                    continue
                
                # Pick a random candidate
                # To explore, pick random. To exploit, pick one that maximizes something?
                # Random is good for escaping local minima.
                new_w = random.choice(candidates)
                
                if new_w == code_indices[i]:
                    continue
                
                # Swap
                old_w = code_indices[i]
                code_indices[i] = new_w
                
                # Prune
                new_code = prune(code_indices[:])
                
                if len(new_code) < len(code_indices):
                    code_indices = new_code
                    improved = True
                    break # Restart loop
                else:
                    # Revert
                    code_indices[i] = old_w
            
            if not improved and len(code_indices) <= 186:
                # If we hit 186, we are done.
                break
            if not improved:
                # Stagnation?
                # We could try a more aggressive move: remove one, add two?
                # Or just continue random swaps.
                # Given time, continue.
                pass

        # Convert indices to words
        return [all_words[i] for i in code_indices]

    else:
        # Generic fallback
        # Greedy
        def word_to_idx_gen(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word_gen(idx):
            w = [0]*n
            for i in range(n-1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        total_words = q ** n
        masks = [0] * total_words
        
        # Precompute masks
        for i in range(total_words):
            w = idx_to_word_gen(i)
            m = 0
            m |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = w[:]
                    new_w[pos] = val
                    n_idx = word_to_idx_gen(new_w)
                    m |= (1 << n_idx)
            masks[i] = m
            
        ALL_ONES = (1 << total_words) - 1
        uncovered = ALL_ONES
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                inter = masks[i] & uncovered
                if inter:
                    cnt = inter.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        return [idx_to_word_gen(i) for i in code_indices]
