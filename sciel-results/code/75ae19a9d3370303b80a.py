import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    return _direct_sum_construct(q, n, radius)

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
    ball_bitsets = [0] * N
    for i in range(N):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    # 1. Initial Construction: Greedy Set Cover
    covered_mask = 0
    all_mask = (1 << N) - 1
    codewords = []
    is_chosen = [False] * N
    
    # Greedy
    while covered_mask != all_mask:
        if time.time() > deadline - 30:
            # Fallback: just add random until covered
            while covered_mask != all_mask:
                i = random.randint(0, N - 1)
                if not is_chosen[i]:
                    codewords.append(i)
                    is_chosen[i] = True
                covered_mask |= ball_bitsets[i]
            break
            
        best_i = -1
        best_gain = 0
        for i in range(N):
            if is_chosen[i]:
                continue
            gain = (ball_bitsets[i] & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_i = i
                
        if best_i == -1:
            break
            
        codewords.append(best_i)
        is_chosen[best_i] = True
        covered_mask |= ball_bitsets[best_i]
        
    # 2. Local Search
    cover_count = np.zeros(N, dtype=np.int16)
    for cw in codewords:
        for j in neighbors[cw]:
            cover_count[j] += 1
            
    def try_remove(cw_idx, cw_val):
        for j in neighbors[cw_val]:
            if cover_count[j] == 1:
                return False
        for j in neighbors[cw_val]:
            cover_count[j] -= 1
        codewords.pop(cw_idx)
        is_chosen[cw_val] = False
        return True
        
    # Pass 1: Remove redundant codewords
    i = 0
    while i < len(codewords):
        if try_remove(i, codewords[i]):
            pass
        else:
            i += 1
        if time.time() > deadline - 10:
            break

    # Pass 2: Local Search (Swaps) - optimized with bit-parallel counting
    random.seed(42)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        random.shuffle(codewords)
        
        for idx in range(len(codewords)):
            if time.time() > deadline - 5:
                break
                
            cw_a = codewords[idx]
            
            # Find unique points (covered only by cw_a)
            # Use bitset to find points with cover_count == 1 that are in ball of cw_a
            # First, build a bitset of points with cover_count == 1
            unique_mask = 0
            for j in neighbors[cw_a]:
                if cover_count[j] == 1:
                    unique_mask |= (1 << j)
                    
            if unique_mask == 0:
                try_remove(idx, cw_a)
                improved = True
                break
            
            # Find candidate replacements that cover all unique points
            # A candidate covers all unique points if (ball_bitsets[candidate] & unique_mask) == unique_mask
            found_swap = False
            for cw_b in range(N):
                if is_chosen[cw_b]:
                    continue
                # Check if cw_b covers all unique points
                if (ball_bitsets[cw_b] & unique_mask) == unique_mask:
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
                break

    # Final cleanup
    i = 0
    while i < len(codewords):
        if try_remove(i, codewords[i]):
            pass
        else:
            i += 1

    result = [all_words[cw] for cw in codewords]
    return result
