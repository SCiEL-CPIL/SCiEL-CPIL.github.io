import time
import random
import numpy as np

N = 256

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 1..256 (0-indexed)."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    aps = _build_aps()
    num_aps = len(aps)
    
    # Precompute for each position, which APs it participates in
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
        
    def get_initial_state(colors):
        """
        Compute count0, count1 for each AP and the total mono score.
        colors: list of 0/1
        """
        count0 = np.zeros(num_aps, dtype=np.int8)
        count1 = np.zeros(num_aps, dtype=np.int8)
        
        # Vectorized initialization might be hard due to variable lengths, 
        # but we can loop over APs.
        # Actually, looping over positions is better to fill counts?
        # No, looping over APs is easier.
        for k, (a, b, c, d) in enumerate(aps):
            c0 = colors[a] == 0
            c1 = colors[b] == 0
            c2 = colors[c] == 0
            c3 = colors[d] == 0
            
            n0 = c0 + c1 + c2 + c3
            n1 = 4 - n0
            
            count0[k] = n0
            count1[k] = n1
            
        # Count mono: count0 == 4 or count1 == 4
        score = np.sum((count0 == 4) | (count1 == 4))
        
        return count0, count1, score

    def local_search(initial_colors, max_time_per_search=2.0):
        """
        Greedy local search with incremental updates.
        """
        colors = list(initial_colors)
        count0, count1, score = get_initial_state(colors)
        
        # Randomize order of positions to break symmetry
        order = list(range(N))
        random.shuffle(order)
        
        search_start = time.time()
        improved = True
        
        while improved and (time.time() - search_start) < max_time_per_search:
            improved = False
            # Shuffle order again? Or just iterate. 
            # Shuffling helps escape plateaus if we allow 0-delta moves, 
            # but for pure greedy, fixed order is deterministic.
            # Let's re-shuffle each pass to be safe against ordering bias.
            random.shuffle(order)
            
            for i in order:
                # Calculate delta for flipping i
                old_val = colors[i]
                new_val = 1 - old_val
                
                delta = 0
                for ap_idx in pos_to_aps[i]:
                    # Before flip
                    was_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                    
                    # Update counts
                    if old_val == 0:
                        count0[ap_idx] -= 1
                        count1[ap_idx] += 1
                    else:
                        count0[ap_idx] += 1
                        count1[ap_idx] -= 1
                    
                    # After flip
                    is_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                    
                if delta < 0:
                    # Accept flip
                    colors[i] = new_val
                    score += delta
                    improved = True
                else:
                    # Revert counts
                    if old_val == 0:
                        count0[ap_idx] += 1
                        count1[ap_idx] -= 1
                    else:
                        count0[ap_idx] -= 1
                        count1[ap_idx] += 1
                    # Note: ap_idx is the last one processed in the loop. 
                    # We need to revert ALL counts in pos_to_aps[i]
                    # This logic is flawed in the snippet above. 
                    # We must revert all.
                    
        # The revert logic above is buggy because it only reverts the last one.
        # Let's rewrite the inner loop to be safer.
        
        return colors, score

    # Corrected Local Search
    def local_search_safe(initial_colors, max_time_per_search=2.0):
        colors = list(initial_colors)
        count0, count1, score = get_initial_state(colors)
        
        search_start = time.time()
        improved = True
        
        while improved and (time.time() - search_start) < max_time_per_search:
            improved = False
            order = list(range(N))
            random.shuffle(order)
            
            for i in order:
                old_val = colors[i]
                new_val = 1 - old_val
                
                delta = 0
                affected_aps = pos_to_aps[i]
                
                # Phase 1: Calculate delta
                for ap_idx in affected_aps:
                    was_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                    
                    if old_val == 0:
                        new_c0 = count0[ap_idx] - 1
                        new_c1 = count1[ap_idx] + 1
                    else:
                        new_c0 = count0[ap_idx] + 1
                        new_c1 = count1[ap_idx] - 1
                        
                    is_mono = (new_c0 == 4) or (new_c1 == 4)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                        
                if delta < 0:
                    # Accept flip
                    colors[i] = new_val
                    score += delta
                    improved = True
                    
                    # Phase 2: Update counts
                    for ap_idx in affected_aps:
                        if old_val == 0:
                            count0[ap_idx] -= 1
                            count1[ap_idx] += 1
                        else:
                            count0[ap_idx] += 1
                            count1[ap_idx] -= 1

        return colors, score

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    candidates.append(thue_morse(N))
    
    # 2. Random
    random.seed(42)
    for _ in range(20):
        candidates.append([random.randint(0, 1) for _ in range(N)])
        
    # 3. Block patterns
    for bs in [2, 4, 8, 16, 32]:
        candidates.append([(i // bs) % 2 for i in range(N)])
        
    best_score = float('inf')
    best_colors = None
    
    # Run local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 10:
            break
            
        # Limit time per search to ensure we try many starts
        # If we have ~30 candidates, and 140s budget, we can spend ~4s each.
        # But local search converges fast. Let's use 2s per search.
        res_colors, res_score = local_search_safe(cand, max_time_per_search=2.0)
        
        if res_score < best_score:
            best_score = res_score
            best_colors = res_colors[:]
            
    # Final polish on the best found
    if best_colors is not None and time.time() < deadline - 5:
        final_colors, final_score = local_search_safe(best_colors, max_time_per_search=5.0)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors[:]
            
    return best_colors
