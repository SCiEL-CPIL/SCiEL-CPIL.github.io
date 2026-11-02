import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Lookup table for word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (including itself)
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        w_list = w[:]
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    idx = word_to_idx.get(tuple(w_list), -1)
                    if idx != -1 and idx not in nb:
                        nb.append(idx)
            w_list[pos] = orig_val
        neighbors.append(nb)
        
    # Precompute bitset for each ball
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
    
    # Precompute: for each word, which other words can cover its unique points
    # This helps speed up swap operations
    # We'll use a different approach: precompute compatibility
    # For each pair (a, b), check if ball_b covers all unique points of a
    # This is expensive to precompute fully, so we'll do it on-the-fly but optimize
    
    best_size = N
    best_codewords = []
    
    # Multiple restarts of Greedy + Local Search
    random.seed(42)
    
    while time.time() < deadline - 10:
        # 1. Initial Construction: Greedy Set Cover with Random Tie-Breaking
        covered_mask = 0
        all_mask = (1 << N) - 1
        codewords = []
        is_chosen = [False] * N
        
        # Greedy
        while covered_mask != all_mask:
            if time.time() > deadline - 5:
                break
                
            best_i = -1
            best_gain = 0
            best_candidates = []
            
            for i in range(N):
                if is_chosen[i]:
                    continue
                gain = (ball_bitsets[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_i == -1:
                break
                
            # Random tie-breaking
            chosen_i = random.choice(best_candidates)
            codewords.append(chosen_i)
            is_chosen[chosen_i] = True
            covered_mask |= ball_bitsets[chosen_i]
        
        # If we broke out early due to time, check if valid
        if covered_mask != all_mask:
             # Fallback: complete coverage
             while covered_mask != all_mask:
                 i = random.randint(0, N - 1)
                 if not is_chosen[i]:
                     codewords.append(i)
                     is_chosen[i] = True
                 covered_mask |= ball_bitsets[i]
        
        # 2. Local Search with optimized swap
        cover_count = [0] * N
        for cw in codewords:
            for j in neighbors[cw]:
                cover_count[j] += 1
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to escape local minima
            random.shuffle(codewords)
            
            for idx in range(len(codewords)):
                if time.time() > deadline - 5:
                    break
                    
                cw_a = codewords[idx]
                
                # Find unique points covered by cw_a
                unique_mask = 0
                for j in neighbors[cw_a]:
                    if cover_count[j] == 1:
                        unique_mask |= (1 << j)
                        
                if unique_mask == 0:
                    # Remove cw_a
                    for j in neighbors[cw_a]:
                        cover_count[j] -= 1
                    codewords.pop(idx)
                    is_chosen[cw_a] = False
                    improved = True
                    break # Restart loop
                    
                else:
                    # Try swap: find a word that covers all unique points
                    found_swap = False
                    # Optimization: Only check words that could potentially cover unique_mask
                    # We need ball_bitsets[cw_b] & ~unique_mask == 0, i.e., unique_mask is subset of ball_bitsets[cw_b]
                    for cw_b in range(N):
                        if is_chosen[cw_b]:
                            continue
                        if (unique_mask & ~ball_bitsets[cw_b]) == 0:
                            # Perform swap
                            # Decrement cover_count for cw_a
                            for j in neighbors[cw_a]:
                                cover_count[j] -= 1
                            is_chosen[cw_a] = False
                            
                            # Increment cover_count for cw_b
                            for j in neighbors[cw_b]:
                                cover_count[j] += 1
                            is_chosen[cw_b] = True
                            
                            codewords[idx] = cw_b
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break # Restart loop
                        
        # 3. Final aggressive cleanup: try to remove or swap each word
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 5:
                break
            cw_a = codewords[i]
            
            # First, check if we can simply remove
            can_remove = True
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    can_remove = False
                    break
            if can_remove:
                for j in neighbors[cw_a]:
                    cover_count[j] -= 1
                codewords.pop(i)
                is_chosen[cw_a] = False
                continue
            
            # Try to swap with a better word
            unique_mask = 0
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    unique_mask |= (1 << j)
            
            if unique_mask > 0:
                # Try to find a replacement that covers unique_mask and has less overlap
                best_b = -1
                best_overlap = float('inf')
                for cw_b in range(N):
                    if is_chosen[cw_b]:
                        continue
                    if (unique_mask & ~ball_bitsets[cw_b]) == 0:
                        # Check overlap: how many points are covered by both
                        overlap = (ball_bitsets[cw_a] & ball_bitsets[cw_b]).bit_count()
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_b = cw_b
                
                if best_b != -1:
                    # Perform swap
                    for j in neighbors[cw_a]:
                        cover_count[j] -= 1
                    is_chosen[cw_a] = False
                    
                    for j in neighbors[best_b]:
                        cover_count[j] += 1
                    is_chosen[best_b] = True
                    
                    codewords[i] = best_b
                    # Don't increment i, recheck this position
                    continue
            
            i += 1
                
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
        
    result = [all_words[cw] for cw in best_codewords]
    return result

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Lookup table for word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (including itself)
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        w_list = w[:]
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    idx = word_to_idx.get(tuple(w_list), -1)
                    if idx != -1 and idx not in nb:
                        nb.append(idx)
            w_list[pos] = orig_val
        neighbors.append(nb)
        
    # Precompute bitset for each ball
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask
    
    best_size = N
    best_codewords = []
    
    # Multiple restarts of Greedy + Local Search
    random.seed(42)
    
    while time.time() < deadline - 10:
        # 1. Initial Construction: Greedy Set Cover with Random Tie-Breaking
        covered_mask = 0
        all_mask = (1 << N) - 1
        codewords = []
        is_chosen = [False] * N
        
        # Greedy
        while covered_mask != all_mask:
            if time.time() > deadline - 5:
                break
                
            best_i = -1
            best_gain = 0
            best_candidates = []
            
            for i in range(N):
                if is_chosen[i]:
                    continue
                gain = (ball_bitsets[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_i == -1:
                break
                
            # Random tie-breaking
            chosen_i = random.choice(best_candidates)
            codewords.append(chosen_i)
            is_chosen[chosen_i] = True
            covered_mask |= ball_bitsets[chosen_i]
        
        # If we broke out early due to time, check if valid
        if covered_mask != all_mask:
             # Fallback: complete coverage
             while covered_mask != all_mask:
                 i = random.randint(0, N - 1)
                 if not is_chosen[i]:
                     codewords.append(i)
                     is_chosen[i] = True
                 covered_mask |= ball_bitsets[i]
        
        # 2. Local Search with optimized swap
        cover_count = [0] * N
        for cw in codewords:
            for j in neighbors[cw]:
                cover_count[j] += 1
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to escape local minima
            random.shuffle(codewords)
            
            for idx in range(len(codewords)):
                if time.time() > deadline - 5:
                    break
                    
                cw_a = codewords[idx]
                
                # Find unique points covered by cw_a
                unique_mask = 0
                for j in neighbors[cw_a]:
                    if cover_count[j] == 1:
                        unique_mask |= (1 << j)
                        
                if unique_mask == 0:
                    # Remove cw_a
                    for j in neighbors[cw_a]:
                        cover_count[j] -= 1
                    codewords.pop(idx)
                    is_chosen[cw_a] = False
                    improved = True
                    break # Restart loop
                    
                else:
                    # Try swap: find a word that covers all unique points
                    found_swap = False
                    for cw_b in range(N):
                        if is_chosen[cw_b]:
                            continue
                        if (unique_mask & ~ball_bitsets[cw_b]) == 0:
                            # Perform swap
                            for j in neighbors[cw_a]:
                                cover_count[j] -= 1
                            is_chosen[cw_a] = False
                            
                            for j in neighbors[cw_b]:
                                cover_count[j] += 1
                            is_chosen[cw_b] = True
                            
                            codewords[idx] = cw_b
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break # Restart loop
                        
        # 3. Final aggressive cleanup: try to remove or swap each word
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 5:
                break
            cw_a = codewords[i]
            
            # First, check if we can simply remove
            can_remove = True
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    can_remove = False
                    break
            if can_remove:
                for j in neighbors[cw_a]:
                    cover_count[j] -= 1
                codewords.pop(i)
                is_chosen[cw_a] = False
                continue
            
            # Try to swap with a better word
            unique_mask = 0
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    unique_mask |= (1 << j)
            
            if unique_mask > 0:
                # Try to find a replacement that covers unique_mask and has less overlap
                best_b = -1
                best_overlap = float('inf')
                for cw_b in range(N):
                    if is_chosen[cw_b]:
                        continue
                    if (unique_mask & ~ball_bitsets[cw_b]) == 0:
                        overlap = (ball_bitsets[cw_a] & ball_bitsets[cw_b]).bit_count()
                        if overlap < best_overlap:
                            best_overlap = overlap
                            best_b = cw_b
                
                if best_b != -1:
                    # Perform swap
                    for j in neighbors[cw_a]:
                        cover_count[j] -= 1
                    is_chosen[cw_a] = False
                    
                    for j in neighbors[best_b]:
                        cover_count[j] += 1
                    is_chosen[best_b] = True
                    
                    codewords[i] = best_b
                    continue
            
            i += 1
                
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
        
    result = [all_words[cw] for cw in best_codewords]
    return result

def _direct_sum_construct(q, n, radius):
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = q ** n  # 2187
    
    # Precompute all words
    all_words = [list(w) for w in product(range(q), repeat=n)]
    
    # Lookup table for word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (including itself)
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        w_list = w[:]
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w_list[pos] = new_val
                    idx = word_to_idx.get(tuple(w_list), -1)
                    if idx != -1 and idx not in nb:
                        nb.append(idx)
            w_list[pos] = orig_val
        neighbors.append(nb)
        
    # Precompute bitset for each ball
    # Each ball is a Python int representing a set of N bits
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    best_size = N
    best_codewords = []
    
    # Multiple restarts of Greedy + Local Search
    random.seed(42)
    
    while time.time() < deadline - 10:
        # 1. Initial Construction: Greedy Set Cover with Random Tie-Breaking
        covered_mask = 0
        all_mask = (1 << N) - 1
        codewords = []
        is_chosen = [False] * N
        
        # Greedy
        while covered_mask != all_mask:
            if time.time() > deadline - 5:
                break
                
            best_i = -1
            best_gain = 0
            best_candidates = []
            
            for i in range(N):
                if is_chosen[i]:
                    continue
                gain = (ball_bitsets[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_i == -1:
                break
                
            # Random tie-breaking
            chosen_i = random.choice(best_candidates)
            codewords.append(chosen_i)
            is_chosen[chosen_i] = True
            covered_mask |= ball_bitsets[chosen_i]
            
        # If we broke out early due to time, check if valid
        if covered_mask != all_mask:
             # Fallback: complete coverage
             while covered_mask != all_mask:
                 i = random.randint(0, N - 1)
                 if not is_chosen[i]:
                     codewords.append(i)
                     is_chosen[i] = True
                 covered_mask |= ball_bitsets[i]
        
        # 2. Local Search
        cover_count = [0] * N
        for cw in codewords:
            for j in neighbors[cw]:
                cover_count[j] += 1
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle to escape local minima
            random.shuffle(codewords)
            
            for idx in range(len(codewords)):
                if time.time() > deadline - 5:
                    break
                    
                cw_a = codewords[idx]
                
                # Find unique points covered by cw_a
                unique_mask = 0
                for j in neighbors[cw_a]:
                    if cover_count[j] == 1:
                        unique_mask |= (1 << j)
                        
                if unique_mask == 0:
                    # Remove cw_a
                    for j in neighbors[cw_a]:
                        cover_count[j] -= 1
                    codewords.pop(idx)
                    is_chosen[cw_a] = False
                    improved = True
                    break # Restart loop
                    
                else:
                    # Try swap
                    found_swap = False
                    # Iterate through all potential replacements
                    # Optimization: Only check if it covers unique_mask
                    for cw_b in range(N):
                        if is_chosen[cw_b]:
                            continue
                        if (unique_mask & ~ball_bitsets[cw_b]) == 0:
                            # Perform swap
                            # Decrement cover_count for cw_a
                            for j in neighbors[cw_a]:
                                cover_count[j] -= 1
                            is_chosen[cw_a] = False
                            
                            # Increment cover_count for cw_b
                            for j in neighbors[cw_b]:
                                cover_count[j] += 1
                            is_chosen[cw_b] = True
                            
                            codewords[idx] = cw_b
                            improved = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break # Restart loop
                        
        # Final cleanup pass (just removals)
        i = 0
        while i < len(codewords):
            if time.time() > deadline - 5:
                break
            cw_a = codewords[i]
            can_remove = True
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    can_remove = False
                    break
            if can_remove:
                for j in neighbors[cw_a]:
                    cover_count[j] -= 1
                codewords.pop(i)
                is_chosen[cw_a] = False
            else:
                i += 1
                
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
        # If we reached 186, we can stop early? No, lower bound is 156.
        # But 186 is the best known. If we hit 186, we can still try to go lower.
        
    result = [all_words[cw] for cw in best_codewords]
    return result
