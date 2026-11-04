import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute dominance masks (bitsets)
    # mask[i] has bit j set if dist(words[i], words[j]) <= radius
    dom_masks = [0] * N
    for i, w in enumerate(words):
        mask = 1 << i
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                mask |= (1 << idx)
                w_list[pos] = orig_val
        dom_masks[i] = mask
        
    # Precompute stars (neighbors) for B&B branching
    # stars[u] = list of c such that u is in dom_masks[c]
    # Which is the same as c is in dom_masks[u] due to symmetry
    stars = [[] for _ in range(N)]
    for u in range(N):
        mask = dom_masks[u]
        while mask:
            lsb = mask & -mask
            c = lsb.bit_length() - 1
            stars[u].append(c)
            mask ^= lsb

    # 3. Greedy Dominating Set
    undom_mask = (1 << N) - 1
    code = []
    
    while undom_mask != 0 and time.time() < deadline - 20:
        best_c = -1
        best_gain = -1
        for c in range(N):
            gain = (undom_mask & dom_masks[c]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain == 0:
            break
        code.append(best_c)
        undom_mask &= ~dom_masks[best_c]
        
    best_code = list(code)
    best_size = len(code)
    
    # 4. Local Search (Kick and Repair)
    
    def greedy_repair(code_list):
        undom = (1 << N) - 1
        for c in code_list:
            undom &= ~dom_masks[c]
            
        new_code = list(code_list)
        
        while undom != 0 and time.time() < deadline - 10:
            best_c = -1
            best_gain = -1
            for c in range(N):
                gain = (undom & dom_masks[c]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_gain == 0:
                break
            new_code.append(best_c)
            undom &= ~dom_masks[best_c]
            
        return new_code

    def prune_code(code_list):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                redundant = True
                mask_c = dom_masks[c]
                m = mask_c
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    covered = False
                    for c2 in code_list:
                        if c2 != c and ((dom_masks[c2] >> u) & 1):
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    for _ in range(100):
        if time.time() > deadline - 15:
            break
        current_code = list(best_code)
        k = random.randint(1, 3)
        if len(current_code) <= k:
            continue
        to_remove = random.sample(current_code, k)
        for w in to_remove:
            current_code.remove(w)
            
        current_code = greedy_repair(current_code)
        current_code = prune_code(current_code)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code

    # 5. Branch and Bound (Limited)
    sys.setrecursionlimit(10000)
    
    def lower_bound(undom_mask):
        if undom_mask == 0:
            return 0
        return (undom_mask.bit_count() + 15) // 16

    def bb_search(undom_mask, current_size, current_code):
        nonlocal best_size, best_code
        
        if current_size >= best_size:
            return
        if undom_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = list(current_code)
            return
        if current_size + lower_bound(undom_mask) >= best_size:
            return
        if time.time() > deadline - 5:
            return

        # Pick first undominated vertex
        u = (undom_mask & -undom_mask).bit_length() - 1
        
        # Branch on candidates covering u
        cands = stars[u]
        # Sort by gain
        cands.sort(key=lambda c: (undom_mask & dom_masks[c]).bit_count(), reverse=True)
        
        for c in cands:
            if time.time() > deadline - 5:
                return
            new_undom = undom_mask & ~dom_masks[c]
            current_code.append(c)
            bb_search(new_undom, current_size + 1, current_code)
            current_code.pop()

    if time.time() < deadline - 10:
        bb_search((1 << N) - 1, 0, [])

    return [list(words[i]) for i in best_code]
