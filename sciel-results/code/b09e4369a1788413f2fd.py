import time
import random
import sys

N = 256

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # 1. Precompute all 4-APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            idx = len(aps)
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
            # Map AP index to positions
            for pos in (a, a + d, a + 2 * d, a + 3 * d):
                ap_indices[pos].append(idx)
                
    # 2. Initialization
    # Random initialization
    colors = [random.randint(0, 1) for _ in range(N)]
    
    # 3. Helper to count mono APs
    def count_mono():
        cnt = 0
        for a, b, c, d in aps:
            if colors[a] == colors[b] == colors[c] == colors[d]:
                cnt += 1
        return cnt

    current_score = count_mono()
    best_score = current_score
    best_colors = colors[:]
    
    # 4. Local Search
    # We will perform multiple passes until no improvement or time out
    # To escape local minima, we can add a small amount of random perturbation
    # or just do a single long run. Given the budget, a single run with 
    # random restarts if fast is good.
    
    # Let's do a loop of local search passes
    # If we get stuck, we might try a "kick" (random flips) and restart local search.
    
    # Optimization: Precompute the "state" of each AP? 
    # No, just check on the fly.
    
    def get_delta(idx):
        """Calculate change in score if we flip color at idx."""
        old_color = colors[idx]
        new_color = 1 - old_color
        delta = 0
        for ap_idx in ap_indices[idx]:
            a, b, c, d = aps[ap_idx]
            # Check if mono before
            is_mono_before = (colors[a] == colors[b] == colors[c] == colors[d])
            # Check if mono after
            # We need to be careful: if idx is one of a,b,c,d
            # We can just temporarily set, check, revert? 
            # Or logic:
            # If it was mono, and we flip one element, it becomes NOT mono (since 3 same, 1 diff).
            # If it was NOT mono:
            #   It becomes mono ONLY if the other 3 are same color as the NEW color of idx.
            
            if is_mono_before:
                # Flipping any element in a mono AP makes it non-mono
                delta -= 1
            else:
                # Check if the other 3 elements have the same color, and that color is new_color
                # Find the other 3 indices
                other_indices = [x for x in (a, b, c, d) if x != idx]
                c1, c2, c3 = colors[other_indices[0]], colors[other_indices[1]], colors[other_indices[2]]
                if c1 == c2 == c3 == new_color:
                    delta += 1
        return delta

    # Main search loop
    # We use a random order of indices to avoid bias
    order = list(range(N))
    random.shuffle(order)
    
    improved = True
    pass_count = 0
    max_passes = 100 # Safety cap, but time is the real limit
    
    while improved and pass_count < max_passes:
        if time.time() - start_time > time_limit:
            break
        improved = False
        pass_count += 1
        # Shuffle order each pass to break symmetries
        random.shuffle(order)
        
        for idx in order:
            if time.time() - start_time > time_limit:
                break
            delta = get_delta(idx)
            if delta < 0:
                # Flip
                colors[idx] = 1 - colors[idx]
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    
    # If time remains, we could try more restarts, but one good run is usually sufficient 
    # for this size. However, to be safe against bad random starts, 
    # we can do a few quick restarts if the first one finishes very fast.
    # But local search on 256 nodes is very fast. 
    # Let's add a simple restart mechanism if we have > 30s left.
    
    if time.time() - start_time < time_limit - 10:
        # Try one more restart from a different random seed
        # Actually, just running the local search again on the current best 
        # doesn't help. We need a new random start.
        
        # Let's try 2 more random restarts if time permits
        for restart in range(2):
            if time.time() - start_time > time_limit - 5:
                break
            colors = [random.randint(0, 1) for _ in range(N)]
            current_score = count_mono()
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
            
            improved = True
            pass_count = 0
            while improved and pass_count < max_passes:
                if time.time() - start_time > time_limit - 5:
                    break
                improved = False
                pass_count += 1
                random.shuffle(order)
                for idx in order:
                    if time.time() - start_time > time_limit - 5:
                        break
                    delta = get_delta(idx)
                    if delta < 0:
                        colors[idx] = 1 - colors[idx]
                        current_score += delta
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors[:]

    return best_colors
