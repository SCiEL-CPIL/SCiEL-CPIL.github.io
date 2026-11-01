"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1) using Greedy + Local Search (Removal & Swap)."""
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    covers = [0] * N
    cover_indices = [[] for _ in range(N)]
    
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        indices = [i]
        
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * powers[pos]
                    mask |= (1 << new_i)
                    indices.append(new_i)
        
        covers[i] = mask
        cover_indices[i] = indices
    
    full_mask = (1 << N) - 1
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_count = -1
            best_indices = []
            
            for i in all_indices:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_indices = [i]
                elif cnt == best_count and cnt > 0:
                    best_indices.append(i)
            
            if best_count == 0:
                break
                
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search(code, deadline):
        if not code:
            return code
        
        code_set = set(code)
        
        counts = [0] * N
        for c in code_set:
            for j in cover_indices[c]:
                counts[j] += 1
        
        best_code = list(code_set)
        best_size = len(best_code)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Removals
            current_list = list(code_set)
            random.shuffle(current_list)
            
            for c in current_list:
                if c not in code_set:
                    continue
                
                safe = True
                for j in cover_indices[c]:
                    if counts[j] <= 1:
                        safe = False
                        break
                
                if safe:
                    for j in cover_indices[c]:
                        counts[j] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
            
            # Phase 2: Swaps
            if time.time() < deadline - 5.0:
                swap_attempts = 50
                for _ in range(swap_attempts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    if not code_set:
                        break
                    
                    c_old = random.choice(list(code_set))
                    c_new = random.randint(0, N-1)
                    
                    if c_new in code_set:
                        continue
                    
                    valid = True
                    for j in cover_indices[c_old]:
                        if counts[j] == 1:
                            if not ((covers[c_new] >> j) & 1):
                                valid = False
                                break
                    
                    if valid:
                        for j in cover_indices[c_old]:
                            counts[j] -= 1
                        for j in cover_indices[c_new]:
                            counts[j] += 1
                        
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        improved = True

        return best_code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search(code, deadline)
        if code:
            best_code = code
            best_size = len(code)
    
    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts * 12345)
        if not unc:
            code = local_search(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
    
    if best_code is None:
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result

def _construct_general(q, n, radius, deadline):
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            cnt = 0
            for j in uncovered:
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    if uncovered:
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
