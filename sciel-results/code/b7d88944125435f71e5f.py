import numpy as np
import time
import random
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Safety margin before 60s limit

    # Precompute all masks for distances 0 to MIN_DIST-1
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
    total_masks = len(all_masks)

    # 1. Find a linear code [17, 8, 6]
    k = 8
    linear_code = None
    np.random.seed(42)
    
    # Try to find a good linear code quickly
    max_linear_search_time = 2.0
    while time.time() - start_time < max_linear_search_time:
        G = np.random.randint(0, 2, size=(k, N))
        G_ints = []
        for i in range(k):
            val = 0
            for j in range(N):
                if G[i, j] == 1:
                    val |= (1 << j)
            G_ints.append(val)
        
        codewords = [0]
        for i in range(k):
            new_codewords = [c ^ G_ints[i] for c in codewords]
            codewords.extend(new_codewords)
        
        min_d = N + 1
        for c in codewords:
            if c == 0: continue
            w = bin(c).count('1')
            if w < min_d:
                min_d = w
                if min_d < MIN_DIST:
                    break
        if min_d >= MIN_DIST:
            linear_code = codewords
            break

    if linear_code is None:
        # Fallback: Greedy from 0
        blocked = bytearray(1 << N)
        code = []
        def add_word(w):
            if blocked[w]: return
            code.append(w)
            for d in range(MIN_DIST):
                for mask in masks_by_dist[d]:
                    blocked[w ^ mask] = 1
        add_word(0)
        for w in range(1, 1 << N):
            if time.time() - start_time > time_limit: break
            if not blocked[w]:
                add_word(w)
        return code

    # 2. Local Search
    code = list(linear_code)
    code_set = set(code)
    blocked = bytearray(1 << N)
    
    def block_word(w):
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1
                
    def unblock_word(w):
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 0

    for w in code:
        block_word(w)
        
    best_code = list(code)
    best_size = len(code)
    
    local_search_deadline = time_limit - 2
    
    # Try to add words first
    # Greedy scan
    for w in range(1 << N):
        if time.time() - start_time > local_search_deadline: break
        if not blocked[w] and w not in code_set:
            code.append(w)
            code_set.add(w)
            block_word(w)
            if len(code) > best_size:
                best_size = len(code)
                best_code = list(code)
                
    # If we couldn't add any, or if we want to try to improve further
    # Perform "Remove and Refill" local search
    # We try to remove a word and see if we can add more than 1 word
    
    # Shuffle indices for diversity
    indices = list(range(len(code)))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() - start_time > local_search_deadline: break
        
        removed_word = code[idx]
        # Remove from code
        code.pop(idx)
        code_set.remove(removed_word)
        unblock_word(removed_word)
        
        # Try to greedily add as many words as possible
        added_count = 0
        added_words = []
        
        # Scan for new words
        # Optimization: Only scan words that were blocked by removed_word?
        # No, other words might become unblocked if we add new words? No, adding blocks more.
        # So we just scan all unblocked words.
        # But scanning 131k words is fast.
        
        for w in range(1 << N):
            if time.time() - start_time > local_search_deadline: break
            if not blocked[w] and w not in code_set:
                code.append(w)
                code_set.add(w)
                block_word(w)
                added_count += 1
                added_words.append(w)
                
        # Check if we improved
        # We removed 1 word. We added `added_count` words.
        # Net change = added_count - 1
        if added_count > 1: # Improvement
            best_size = len(code)
            best_code = list(code)
            # Continue searching with this new code?
            # Yes, restart the scan or continue?
            # Let's restart the local search loop with the new code to find more improvements
            # But we need to update `blocked` correctly. It is already updated.
            # We can break and restart the outer loop or just continue.
            # Since we improved, we should probably continue trying to improve this new code.
            # But the `indices` list is now stale.
            # Let's just break and let the outer loop (if we had one) or just return.
            # Actually, let's do a few passes.
            
        else:
            # No improvement, restore
            code.insert(idx, removed_word)
            code_set.add(removed_word)
            block_word(removed_word)
            # Remove the added words
            for w in added_words:
                code.remove(w) # This is slow!
                code_set.remove(w)
                unblock_word(w)

    return best_code
