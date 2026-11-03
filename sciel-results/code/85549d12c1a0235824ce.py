import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_space = q ** n
    
    # Precompute word to index and index to word
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks as Python integers (bitsets)
    # cover_masks[i] is a bitset where bit j is 1 if word j is within radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Precompute reverse cover: for each word, which codewords cover it
    # This is useful for quick lookup but might be memory heavy if stored as lists of lists.
    # For 2187 words, it's fine.
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Helper: Get list of set bits
    def get_bits(mask):
        bits = []
        while mask:
            low = mask & (-mask)
            bits.append(low.bit_length() - 1)
            mask ^= low
        return bits

    # Helper: Greedy cover for a specific uncovered mask
    def greedy_cover(uncovered_mask):
        code = []
        unc = uncovered_mask
        while unc:
            best_gain = 0
            best_cand = -1
            # Iterate through all possible codewords
            # Optimization: We can skip codewords that don't cover any uncovered word
            for i in range(total_space):
                gain = (cover_masks[i] & unc).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cand = i
                    if gain == unc.bit_count():
                        break # Can't do better
            
            if best_cand == -1:
                break
                
            code.append(best_cand)
            unc &= ~cover_masks[best_cand]
            
        return code

    # Helper: Local Search (Reduction and Swaps)
    def local_search(code):
        if not code:
            return code
        
        current_code = code[:]
        current_code_set = set(current_code)
        
        # Build cover_count array
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Phase 1: Removal of redundant codewords
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            # Phase 2: 1-for-1 Swaps
            if not improved:
                # Try to replace a codeword with another to allow further removals or just for better positioning?
                # Actually, standard local search for minimum set cover often just does removals.
                # Swaps are useful if we get stuck in a local minimum where no single removal works,
                # but a swap enables a removal.
                
                # Heuristic: Try swapping a codeword with low "efficiency" with one with high efficiency?
                # Or simply: For each codeword, try to find a replacement that covers the same unique words
                # but potentially allows other codewords to be removed?
                # This is complex. A simpler and effective move is:
                # If we can't remove anything, try to "shake" the solution.
                # But Iterated Local Search (perturbation) is better than complex swaps.
                
                # Let's try a simple swap: Replace C_i with C_j if C_j covers all unique words of C_i
                # AND C_j is not in the set. This doesn't reduce size, so it's only useful if it leads to
                # a state where a removal is possible in the next iteration.
                
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    
                    if unique_mask == 0:
                        # This is redundant, should have been caught in Phase 1
                        # Remove it
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    # Find candidates that cover all unique words
                    # Intersection of reverse_cover for all words in unique_mask
                    words_in_unique = get_bits(unique_mask)
                    if not words_in_unique:
                        continue
                        
                    candidates = set(reverse_cover[words_in_unique[0]])
                    for w in words_in_unique[1:]:
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        # Pick the one that covers the most total words (greedy choice)
                        best_cand = -1
                        best_gain = -1
                        for cand in candidates:
                            gain = cover_masks[cand].bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_cand = cand
                        
                        if best_cand != -1:
                            # Perform swap
                            m_old = cover_masks[idx]
                            m_new = cover_masks[best_cand]
                            
                            # Update counts
                            m = m_old
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                                
                            m = m_new
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m ^= low
                                
                            current_code[i] = best_cand
                            current_code_set.remove(idx)
                            current_code_set.add(best_cand)
                            improved = True
                            break
                            
        return current_code

    # Iterated Local Search
    def ils():
        best_code = []
        best_size = float('inf')
        
        # Initial greedy
        init_code = greedy_cover(full_mask)
        init_code = local_search(init_code)
        best_code = init_code
        best_size = len(init_code)
        
        iterations = 0
        while time.time() < deadline - 5.0:
            iterations += 1
            
            # Perturb: Remove k random codewords
            k = random.randint(3, 10)
            if len(best_code) <= k:
                continue
                
            perturbed_code = best_code[:]
            removed = random.sample(perturbed_code, k)
            for r in removed:
                perturbed_code.remove(r)
            
            # Compute uncovered
            covered = 0
            for idx in perturbed_code:
                covered |= cover_masks[idx]
            uncovered = full_mask & ~covered
            
            # Repair: Greedy cover the uncovered
            repair_code = greedy_cover(uncovered)
            
            # Combine
            new_code = perturbed_code + repair_code
            
            # Local Search
            new_code = local_search(new_code)
            
            if len(new_code) < best_size:
                best_size = len(new_code)
                best_code = new_code
                
        return best_code

    # Run ILS
    # We can run multiple independent ILS chains if time permits, or just one long one.
    # Given the problem size, one long ILS is usually sufficient to find good solutions.
    # But random restarts of the whole process might help if we get stuck.
    
    best_code = ils()
    
    # If time remains, try a few more random greedy starts + local search
    if time.time() < deadline - 10.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            # Randomized Greedy
            unc = full_mask
            code = []
            while unc:
                # Sample a few candidates and pick the best? Or just greedy with random tie-breaking?
                # Let's do standard greedy but with random tie-breaking for diversity
                best_gain = 0
                candidates = []
                for i in range(total_space):
                    gain = (cover_masks[i] & unc).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [i]
                    elif gain == best_gain and gain > 0:
                        candidates.append(i)
                
                if not candidates:
                    break
                best_idx = random.choice(candidates)
                code.append(best_idx)
                unc &= ~cover_masks[best_idx]
                
            code = local_search(code)
            if len(code) < len(best_code):
                best_code = code

    return [all_words[i] for i in best_code]
