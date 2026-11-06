import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Precompute conflicts
    N = 255
    # indices 0..254 correspond to numbers 2..256
    # masks[i] is a bitset of j such that lcm(i+2, j+2) <= 256
    masks = [0] * N
    weights = [0.0] * N
    for i in range(N):
        a = i + 2
        weights[i] = 1.0 / a
        for j in range(N):
            if i == j: continue
            b = j + 2
            if lcm(a, b) <= 256:
                masks[i] |= (1 << j)
    
    best_score = 0.0
    best_mask = 0
    
    def get_score(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def is_compatible(mask, u):
        # Check if u can be added to set represented by mask
        # u must not be in mask
        if mask & (1 << u):
            return False
        # u must not conflict with any in mask
        # mask & masks[u] should be 0
        return (mask & masks[u]) == 0

    def local_search(mask):
        # Hill climbing with 1-add and 1-swap
        improved = True
        current_score = get_score(mask)
        while improved:
            improved = False
            # Try adding
            for u in range(N):
                if mask & (1 << u): continue
                if is_compatible(mask, u):
                    new_mask = mask | (1 << u)
                    new_score = current_score + weights[u]
                    if new_score > current_score:
                        mask = new_mask
                        current_score = new_score
                        improved = True
                        break # restart loop to check new state
            
            # Try swapping (remove v, add u)
            if not improved:
                # To make this faster, we can limit swaps or just do a few passes
                for v in range(N):
                    if not (mask & (1 << v)): continue
                    mask_without_v = mask ^ (1 << v)
                    # Check if any u can be added to mask_without_v
                    # Optimization: precompute compatible candidates?
                    for u in range(N):
                        if mask & (1 << u): continue
                        if is_compatible(mask_without_v, u):
                            new_score = current_score - weights[v] + weights[u]
                            if new_score > current_score:
                                mask = mask_without_v | (1 << u)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
        return mask, current_score

    # Multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Greedy initialization
        # Randomize order slightly? Or just standard greedy?
        # Standard greedy: sort by weight desc (index asc)
        order = list(range(N))
        random.shuffle(order) # Randomized restarts
        # Or better: sort by weight, but with random tie-breaking or slight noise?
        # Let's try: sort by weight descending, but start from a random subset?
        # Simpler: Randomized Greedy
        # Pick random node, add if compatible, repeat.
        
        # Actually, a good heuristic is:
        # 1. Start with empty set.
        # 2. While possible, pick the highest weight node compatible with current set.
        
        curr_mask = 0
        available = [1] * N
        # To speed up, we can just iterate N times
        for _ in range(N):
            best_u = -1
            best_w = 0.0
            for u in range(N):
                if not (curr_mask & (1 << u)) and is_compatible(curr_mask, u):
                    if weights[u] > best_w:
                        best_w = weights[u]
                        best_u = u
            if best_u == -1:
                break
            curr_mask |= (1 << best_u)
        
        # Local Search
        ls_mask, ls_score = local_search(curr_mask)
        
        if ls_score > best_score:
            best_score = ls_score
            best_mask = ls_mask
            
        restarts += 1
        
    return [i + 2 for i in range(N) if best_mask & (1 << i)]
