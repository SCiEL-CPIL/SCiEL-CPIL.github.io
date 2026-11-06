import time
import random

def construct():
    N = 128
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # Precompute 4-APs
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a-1, a-1+d, a-1+2*d, a-1+3*d) # 0-indexed
            aps.append(ap)
            
    # Precompute APs containing each index
    aps_by_idx = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            aps_by_idx[i].append(idx)
            
    def is_bad_ap(ap, colors):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        # Bad if at most 2 distinct colors
        # Quick check:
        if c0 == c1 == c2 == c3: return True
        if c0 == c1 and c2 == c3 and c0 != c2: return True
        if c0 == c2 and c1 == c3 and c0 != c1: return True
        if c0 == c3 and c1 == c2 and c0 != c1: return True
        # Check for exactly 2 colors in other patterns?
        # Actually, simpler:
        return len({c0, c1, c2, c3}) < 3

    best_valid_solution = None
    best_colors_used = N # Upper bound
    
    # Try k from 3 upwards
    for k in range(3, 8):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Multiple restarts for this k
        # Time budget per k
        budget_per_k = 20.0
        t_start_k = time.time()
        
        while time.time() - t_start_k < budget_per_k:
            # Initialize
            # Random assignment of k colors
            colors = [random.randint(0, k-1) for _ in range(N)]
            
            # Calculate initial bad count
            bad_count = 0
            for ap in aps:
                if is_bad_ap(ap, colors):
                    bad_count += 1
            
            # Local Search
            improved = True
            while improved and time.time() - t_start_k < budget_per_k:
                improved = False
                best_delta = 0
                best_move = None
                best_new_bad = bad_count
                
                # Try all moves
                for i in range(N):
                    old_c = colors[i]
                    # Try all other colors
                    for new_c in range(k):
                        if new_c == old_c: continue
                        
                        # Calculate change in bad_count
                        # Only APs containing i are affected
                        delta = 0
                        for ap_idx in aps_by_idx[i]:
                            ap = aps[ap_idx]
                            # Check if it was bad
                            was_bad = is_bad_ap(ap, colors)
                            # Simulate change
                            # We need to check the AP with new color at i
                            # Create a tuple or just check logic
                            # To avoid creating lists, we can write a helper or inline
                            
                            # Inline check for speed
                            c0 = colors[ap[0]] if ap[0] != i else new_c
                            c1 = colors[ap[1]] if ap[1] != i else new_c
                            c2 = colors[ap[2]] if ap[2] != i else new_c
                            c3 = colors[ap[3]] if ap[3] != i else new_c
                            
                            is_bad_now = len({c0, c1, c2, c3}) < 3
                            
                            if was_bad and not is_bad_now:
                                delta -= 1
                            elif not was_bad and is_bad_now:
                                delta += 1
                                
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, new_c)
                            best_new_bad = bad_count + delta
                            
                if best_move:
                    i, new_c = best_move
                    colors[i] = new_c
                    bad_count = best_new_bad
                    improved = True
                    if bad_count == 0:
                        # Valid solution found
                        num_colors = len(set(colors))
                        if num_colors < best_colors_used:
                            best_colors_used = num_colors
                            best_valid_solution = colors[:]
                        # We can try to reduce colors further by remapping?
                        # Or just break and try to find a better one?
                        # Since we iterate k, we found one with <=k colors.
                        # We should try to see if we can reduce the actual number used.
                        # But the loop structure tries k=3 first. If we find valid with k=3, 
                        # the actual colors used might be 3.
                        # If we find valid with k=4, actual might be 3 or 4.
                        # We want to minimize actual colors used.
                        # So we should keep searching even if bad_count=0, 
                        # trying to reduce the number of distinct colors in the set.
                        
                        # How? 
                        # If we have a valid coloring, we can try to change a color 
                        # to an existing other color to reduce the set size, 
                        # provided it doesn't create bad APs.
                        
                        # Let's add a "color reduction" phase.
                        pass
                else:
                    # No improvement found
                    # Try random restart or break
                    # Simple random kick
                    if random.random() < 0.1:
                        i = random.randint(0, N-1)
                        new_c = random.randint(0, k-1)
                        # Update bad count
                        for ap_idx in aps_by_idx[i]:
                             ap = aps[ap_idx]
                             was_bad = is_bad_ap(ap, colors)
                             c0 = colors[ap[0]] if ap[0] != i else new_c
                             c1 = colors[ap[1]] if ap[1] != i else new_c
                             c2 = colors[ap[2]] if ap[2] != i else new_c
                             c3 = colors[ap[3]] if ap[3] != i else new_c
                             is_bad_now = len({c0, c1, c2, c3}) < 3
                             if was_bad and not is_bad_now: bad_count -= 1
                             elif not was_bad and is_bad_now: bad_count += 1
                             colors[i] = new_c
                             # This is messy. Better to just restart the local search or accept the kick.
                             # Let's just restart the inner loop.
            if bad_count == 0:
                num_colors = len(set(colors))
                if num_colors < best_colors_used:
                    best_colors_used = num_colors
                    best_valid_solution = colors[:]
                
    if best_valid_solution:
        return best_valid_solution
    else:
        # Fallback: Greedy
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1)) \
                   and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1)):
                    colors.append(color)
                    break
        return colors
