import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

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
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Precompute reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Greedy Set Cover with cached gains
    def greedy_cover():
        uncovered = full_mask
        code = []
        gains = [cover_masks[i].bit_count() for i in range(total_space)]
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_space):
                if gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            new_uncovered = uncovered & ~cover_masks[best_idx]
            if new_uncovered == uncovered:
                uncovered = new_uncovered
                break
            delta = uncovered ^ new_uncovered
            m = delta
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                m ^= low
                for i in reverse_cover[w_idx]:
                    gains[i] -= 1
            uncovered = new_uncovered
        return code

    # Local Search: Remove redundant codewords, perform swaps, and swap+add
    def local_search_reduce(code):
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
            
            # Phase 1: Removal
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
            
            # Phase 2: Swaps
            if not improved:
                max_unique_to_cover = 4
                
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
                    
                    if unique_mask.bit_count() > max_unique_to_cover:
                        continue
                    
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    
                    candidates = set(reverse_cover[words_in_unique[0]])
                    
                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        # Prefer candidates that also cover many other words
                        cand_list = list(candidates)
                        if len(cand_list) > 1:
                            # Score candidates by how much they help overall
                            best_cand = None
                            best_score = -1
                            for c in cand_list[:20]:  # Limit to top 20 for speed
                                score = (cover_masks[c] & ~mask).bit_count()
                                if score > best_score:
                                    best_score = score
                                    best_cand = c
                            new_idx = best_cand
                        else:
                            new_idx = cand_list[0]
                        
                        m = cover_masks[idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        
                        current_code[i] = new_idx
                        current_code_set.remove(idx)
                        current_code_set.add(new_idx)
                        
                        improved = True
                        break
            
            # Phase 3: Try to remove a codeword whose unique words are covered by existing codewords
            if not improved and len(current_code) > 1:
                n_cw = len(current_code)
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
                    
                    if unique_mask.bit_count() > 5:
                        continue
                    
                    if unique_mask == 0:
                        # Should have been removed in Phase 1
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
                    
                    # Check if all unique words are covered by other codewords
                    all_covered = True
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        # Find if any other codeword covers this word
                        covered_by_other = False
                        for j in range(n_cw):
                            if j == i:
                                continue
                            if (cover_masks[current_code[j]] >> w_idx) & 1:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            all_covered = False
                            break
                        m ^= low
                    
                    if all_covered:
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
                        
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    best_size = len(best_code)
    
    # Apply local search
    best_code = local_search_reduce(best_code)
    best_size = len(best_code)
    
    def greedy_cover_randomized(temperature=0.1):
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Temperature-based selection
            if len(candidates) > 1 and random.random() < temperature:
                # Pick a random candidate from top gains
                top_gains = []
                for i in candidates:
                    g = (cover_masks[i] & uncovered).bit_count()
                    top_gains.append((g, i))
                top_gains.sort(reverse=True)
                # Pick from top 3
                k = min(3, len(top_gains))
                best_idx = top_gains[random.randint(0, k-1)][1]
            else:
                best_idx = random.choice(candidates)
            
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run randomized restarts if time permits
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        temp = 0.1 + 0.05 * (restart_count % 3)
        cand_code = greedy_cover_randomized(temp)
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < best_size:
            best_code = cand_code
            best_size = len(cand_code)
        
        # Also try local search on the best code found so far with different random seeds
        if restart_count % 5 == 0 and time.time() < deadline - 5.0:
            trial = best_code[:]
            trial = local_search_reduce(trial)
            if len(trial) < best_size:
                best_code = trial
                best_size = len(trial)
    
    return [all_words[i] for i in best_code]
