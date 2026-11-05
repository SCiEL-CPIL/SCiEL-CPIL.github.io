import time
import random
import numpy as np

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    
    # Precompute all nonconstant 4-term progressions (APs)
    # An AP is (a, a+d, a+2d, a+3d) with d > 0 and all indices in [0, N)
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            # 0-indexed positions
            p = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(p)
            for idx in p:
                pos_to_aps[idx].append(len(aps) - 1)
    
    num_aps = len(aps)
    
    # Helper to count bad APs (those with < 3 distinct colors)
    def count_bad(colors):
        bad = 0
        for p in aps:
            c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
            # Check if number of distinct colors < 3
            # This is equivalent to: not (at least 3 distinct)
            # At least 3 distinct means NOT (all same) AND NOT (exactly 2 distinct)
            # Faster check: 
            if c0 == c1 == c2 == c3:
                bad += 1
            elif c0 == c1 and c2 == c3:
                bad += 1
            elif c0 == c2 and c1 == c3:
                bad += 1
            elif c0 == c3 and c1 == c2:
                bad += 1
            elif c0 == c1 and c0 == c2 and c0 != c3:
                bad += 1
            elif c0 == c1 and c0 == c3 and c0 != c2:
                bad += 1
            elif c0 == c2 and c0 == c3 and c0 != c1:
                bad += 1
            elif c1 == c2 and c1 == c3 and c1 != c0:
                bad += 1
            elif c0 == c2 and c1 == c3:
                bad += 1
            elif c0 == c3 and c1 == c2:
                bad += 1
            elif c0 == c1 and c2 == c3:
                bad += 1
            # Wait, the above is redundant and messy. Let's use a set or a cleaner logic.
            # Actually, for speed, let's just use the set approach but optimized.
            # Or better, just count distinct.
            # Let's revert to a clean, slightly slower but correct check for clarity in this helper,
            # as the main loop will use a more targeted approach.
            # Actually, let's just use the set. It's clear.
            # We'll optimize the inner loop in the main search instead.
        return bad
    
    # Better count_bad using numpy or just efficient python
    def count_bad_fast(colors):
        bad = 0
        for p in aps:
            c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
            # A 4-term AP is bad if it has 1 or 2 distinct colors.
            # It has 1 distinct color if c0==c1==c2==c3.
            # It has 2 distinct colors if exactly two values are present.
            # Let's just check: if len({c0,c1,c2,c3}) < 3: bad += 1
            # This is fast enough in Python for ~4000 APs.
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad

    best_colors = None
    best_bad = float('inf')
    
    # Try to find a valid coloring with K=3, then K=4, etc.
    for K in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
            
        # Use a randomized local search (Min-Conflicts style)
        # We represent colors as a numpy array of ints.
        
        # Multiple restarts
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random colors
            colors = np.random.randint(0, K, N)
            
            # Calculate initial bad count
            curr_bad = count_bad_fast(colors)
            
            if curr_bad < best_bad:
                best_bad = curr_bad
                best_colors = colors.copy()
                
            if curr_bad == 0:
                break
            
            # Local search: iteratively pick a bad AP, change one of its positions to a better color
            steps = 0
            max_steps = 20000  # Limit steps per restart to avoid getting stuck
            while curr_bad > 0 and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                    
                # Find a random bad AP
                # To be efficient, we could maintain a list of bad APs, but for now,
                # we just pick a random AP and check if it's bad.
                # If it's not bad, we might waste time.
                # Better: Pick a random position, look at its APs, if any are bad, pick one.
                # Or even better: maintain a set of bad AP indices.
                
                # For simplicity and speed in this implementation, let's just pick a random AP.
                # If it's not bad, skip.
                # This might be slow if bad APs are rare.
                
                # Let's try a different approach: pick a random position, 
                # find a bad AP involving it, then change that position.
                
                pos_idx = random.randint(0, N - 1)
                found_bad_ap = False
                for ap_idx in pos_to_aps[pos_idx]:
                    p = aps[ap_idx]
                    c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        # This AP is bad. Change pos_idx to a better color.
                        found_bad_ap = True
                        break
                
                if not found_bad_ap:
                    steps += 1
                    continue
                
                # Try all K colors for pos_idx and pick the one that minimizes local conflicts
                # Local conflicts: number of bad APs involving pos_idx
                old_c = colors[pos_idx]
                best_c = old_c
                min_local_bad = float('inf')
                
                for c in range(K):
                    if c == old_c:
                        continue
                    colors[pos_idx] = c
                    local_bad = 0
                    for ap_idx in pos_to_aps[pos_idx]:
                        p = aps[ap_idx]
                        c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
                        if len({c0, c1, c2, c3}) < 3:
                            local_bad += 1
                    if local_bad < min_local_bad:
                        min_local_bad = local_bad
                        best_c = c
                        if local_bad == 0:
                            break
                            
                colors[pos_idx] = best_c
                
                # Recalculate global bad count periodically or if we made progress
                # To save time, let's only recalculate every 10 steps or if we suspect improvement
                steps += 1
                if steps % 10 == 0 or min_local_bad == 0:
                    curr_bad = count_bad_fast(colors)
                    if curr_bad < best_bad:
                        best_bad = curr_bad
                        best_colors = colors.copy()
                    if curr_bad == 0:
                        break
                        
            if curr_bad == 0:
                break
                
        if best_bad == 0:
            break
            
    if best_colors is not None and best_bad == 0:
        return list(best_colors)
        
    # If no valid coloring found with K<=6, try a greedy construction with more colors
    # Or just return a valid coloring with many colors (e.g., K=8 or K=128)
    # Let's try a quick greedy with K=8
    if time.time() < deadline - 5:
        K = 8
        colors = np.zeros(N, dtype=int)
        for i in range(N):
            # Try colors 0..K-1, pick the first one that doesn't create a new bad AP
            # This is not guaranteed to work, but let's try.
            for c in range(K):
                colors[i] = c
                # Check all APs ending at i (i is the 4th term)
                valid = True
                for ap_idx in pos_to_aps[i]:
                    p = aps[ap_idx]
                    if p[3] != i:  # Only check APs where i is the last term
                        continue
                    c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    break
        # This greedy might fail. Let's just return a valid coloring with 128 colors.
        # Actually, a simple valid coloring is colors[i] = i % 4? No, that fails.
        # colors[i] = i % 5? Might fail.
        # The safest is to use distinct colors for positions that are close.
        # Or just return list(range(N)) which is always valid but uses 128 colors.
        # The task says "Minimize the number of labels actually used".
        # So we want the smallest K such that a valid coloring exists.
        # We tried K=3..6 and failed. Let's try K=7,8,9,10 with the same search.
        
    # Try K=7..10 with a faster search
    for K in [7, 8, 9, 10]:
        if time.time() > deadline - 5:
            break
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            colors = np.random.randint(0, K, N)
            curr_bad = count_bad_fast(colors)
            if curr_bad < best_bad:
                best_bad = curr_bad
                best_colors = colors.copy()
            if curr_bad == 0:
                break
            # Quick local search
            for _ in range(5000):
                if time.time() > deadline - 5:
                    break
                pos_idx = random.randint(0, N - 1)
                found_bad_ap = False
                for ap_idx in pos_to_aps[pos_idx]:
                    p = aps[ap_idx]
                    c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        found_bad_ap = True
                        break
                if not found_bad_ap:
                    continue
                old_c = colors[pos_idx]
                best_c = old_c
                min_local_bad = float('inf')
                for c in range(K):
                    if c == old_c:
                        continue
                    colors[pos_idx] = c
                    local_bad = 0
                    for ap_idx in pos_to_aps[pos_idx]:
                        p = aps[ap_idx]
                        c0, c1, c2, c3 = colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]
                        if len({c0, c1, c2, c3}) < 3:
                            local_bad += 1
                    if local_bad < min_local_bad:
                        min_local_bad = local_bad
                        best_c = c
                        if local_bad == 0:
                            break
                colors[pos_idx] = best_c
                if _ % 10 == 0:
                    curr_bad = count_bad_fast(colors)
                    if curr_bad < best_bad:
                        best_bad = curr_bad
                        best_colors = colors.copy()
                    if curr_bad == 0:
                        break
            if best_bad == 0:
                break
        if best_bad == 0:
            break
            
    if best_colors is not None:
        return list(best_colors)
    else:
        # Fallback: return a valid coloring with 128 colors
        return list(range(N))
