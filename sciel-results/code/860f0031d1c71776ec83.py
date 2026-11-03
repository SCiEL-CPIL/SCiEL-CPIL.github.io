import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for large spaces or general radius where exact B&B is too slow
    if total_words > 5000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            # Construct a good heuristic solution for larger n
            # Use a greedy approach with random restarts
            all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
            # Precompute balls
            balls = [0] * total_words
            for i, w in enumerate(all_words):
                mask = 0
                mask |= (1 << i)
                for pos in range(n):
                    orig = w[pos]
                    for new_val in range(q):
                        if new_val != orig:
                            neighbor_idx = 0
                            for k in range(n):
                                val = w[k] if k != pos else new_val
                                neighbor_idx = neighbor_idx * q + val
                            mask |= (1 << neighbor_idx)
                balls[i] = mask
            
            all_ones_mask = (1 << total_words) - 1
            best_size = total_words + 1
            best_code = None
            
            start_time = time.time()
            time_limit = 110.0
            
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                uncovered = all_ones_mask
                selected = []
                while uncovered:
                    best_idx = -1
                    max_cover = 0
                    for i in range(total_words):
                        c = (balls[i] & uncovered).bit_count()
                        if c > max_cover:
                            max_cover = c
                            best_idx = i
                    if best_idx == -1: break
                    selected.append(best_idx)
                    uncovered &= ~balls[best_idx]
                
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_code = [all_words[i] for i in selected]
            
            return best_code if best_code else [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Exact B&B for small spaces (3^7 = 2187)
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # Precompute balls
    balls = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    neighbor_idx = 0
                    for k in range(n):
                        val = w[k] if k != pos else new_val
                        neighbor_idx = neighbor_idx * q + val
                    mask |= (1 << neighbor_idx)
        balls[i] = mask

    # Precompute coverage lists: for each word, which codewords cover it
    # coverers[word_idx] = list of codeword indices that cover word_idx
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = balls[i]
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            coverers[idx].append(i)
            m ^= lsb

    all_ones_mask = (1 << total_words) - 1
    
    best_size = total_words + 1
    best_code = None
    
    # Greedy initial solution to set a tight bound
    uncovered = all_ones_mask
    selected_greedy = []
    while uncovered:
        best_idx = -1
        max_cover = 0
        for i in range(total_words):
            c = (balls[i] & uncovered).bit_count()
            if c > max_cover:
                max_cover = c
                best_idx = i
        if best_idx == -1: break
        selected_greedy.append(best_idx)
        uncovered &= ~balls[best_idx]
    
    if len(selected_greedy) < best_size:
        best_size = len(selected_greedy)
        best_code = [all_words[i] for i in selected_greedy]

    # B&B Search
    start_time = time.time()
    time_limit = 110.0
    
    # We need to track the current path
    # Using recursion with global variables for state is faster in Python for this depth
    
    current_selected = []
    
    def lower_bound(uncovered_mask):
        # Simple bound: ceil(popcount(uncovered) / max_ball_size)
        # max_ball_size is 1 + n(q-1)
        max_cover = 1 + n * (q - 1)
        return (uncovered_mask.bit_count() + max_cover - 1) // max_cover

    def search(uncovered_mask, depth):
        nonlocal best_size, best_code, current_selected
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered_mask == 0:
            if depth < best_size:
                best_size = depth
                best_code = [all_words[i] for i in current_selected]
            return
            
        if depth + lower_bound(uncovered_mask) >= best_size:
            return
            
        # Pick an uncovered word with the fewest candidates to minimize branching
        # Find first set bit
        m = uncovered_mask
        # Heuristic: pick the word with minimum len(coverers[idx]) intersected with available?
        # For now, pick the first one.
        lsb = m & -m
        u_idx = lsb.bit_length() - 1
        
        # Get candidates covering u_idx
        candidates = coverers[u_idx]
        
        # Filter candidates: must cover at least one word in uncovered_mask
        # And sort by coverage of uncovered_mask (descending)
        scored_candidates = []
        for c in candidates:
            cover_count = (balls[c] & uncovered_mask).bit_count()
            if cover_count > 0:
                scored_candidates.append((cover_count, c))
                
        scored_candidates.sort(reverse=True)
        
        # Dominance pruning: if candidate A covers a superset of candidate B within uncovered_mask, skip B
        # This is O(N^2) per node, might be too slow. Skip for now or implement carefully.
        
        for cover_count, c in scored_candidates:
            current_selected.append(c)
            new_uncovered = uncovered_mask & ~balls[c]
            search(new_uncovered, depth + 1)
            current_selected.pop()
            
            # If we found a solution of size 1 (impossible here) or improved, we might prune more
            # But standard B&B continues to find all solutions of size < best_size

    search(all_ones_mask, 0)
    
    if best_code is None:
        return [list(random.choices(range(q), k=n)) for _ in range(100)]
        
    return best_code
