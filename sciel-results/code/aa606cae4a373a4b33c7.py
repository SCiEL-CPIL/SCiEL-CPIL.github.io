import time
import random

def construct():
    N = 128
    # Precompute all 4-APs (0-indexed)
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def get_cost(coloring):
        bad_count = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad_count += 1
        return bad_count

    def search(K, time_limit):
        start_time = time.time()
        best_cost = float('inf')
        best_coloring = None
        
        # Multiple restarts
        while time.time() - start_time < time_limit:
            # Initial random coloring
            coloring = [random.randint(0, K - 1) for _ in range(N)]
            current_cost = get_cost(coloring)
            
            if current_cost == 0:
                return coloring
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = coloring[:]
                
            T = 5.0
            stuck_count = 0
            no_improve_steps = 0
            local_best_cost = current_cost
            local_best_coloring = coloring[:]
            
            while time.time() - start_time < time_limit:
                # Pick a random position
                i = random.randint(0, N - 1)
                old_c = coloring[i]
                
                # Pick a new color
                new_c = random.randint(0, K - 1)
                if new_c == old_c:
                    continue
                
                # Calculate delta
                delta = 0
                for ap_idx in pos_to_aps[i]:
                    ap = aps[ap_idx]
                    # Get current colors
                    c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
                    old_distinct = len({c0, c1, c2, c3})
                    was_bad = old_distinct < 3
                    
                    # Apply new color
                    if ap[0] == i:
                        c0 = new_c
                    elif ap[1] == i:
                        c1 = new_c
                    elif ap[2] == i:
                        c2 = new_c
                    elif ap[3] == i:
                        c3 = new_c
                        
                    new_distinct = len({c0, c1, c2, c3})
                    is_now_bad = new_distinct < 3
                    
                    if was_bad and not is_now_bad:
                        delta -= 1
                    elif not was_bad and is_now_bad:
                        delta += 1
                
                # Acceptance
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    if random.random() < (1.0 / (1.0 + delta * T)):
                        accept = True
                
                if accept:
                    coloring[i] = new_c
                    current_cost += delta
                    
                    if current_cost < local_best_cost:
                        local_best_cost = current_cost
                        local_best_coloring = coloring[:]
                        no_improve_steps = 0
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_coloring = coloring[:]
                            if best_cost == 0:
                                return best_coloring
                    else:
                        no_improve_steps += 1
                        
                    stuck_count += 1
                    if stuck_count > 10000:
                        break # Restart if too many steps without global improvement? 
                        # Actually, let's just restart if no improvement for a while.
                        
                    if no_improve_steps > 5000:
                        break # Restart local search
                    
                T *= 0.995
                if T < 0.01:
                    T = 0.01

        return best_coloring if best_cost == 0 else None

    # Try increasing K
    for K in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        result = search(K, 20.0)
        if result is not None:
            # Verify
            if get_cost(result) == 0:
                return result

    # Fallback: Greedy from Inspiration 1
    colors = []
    for x in range(1, 129):
        for color in range(128):
            # Check 4-APs ending at x
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                prev = [colors[x - 1 - j * d] for j in (1, 2, 3)]
                if len({color} | set(prev)) < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check 3-APs? No, task is 4-APs.
            # The condition "at least 3 distinct colors" for 4-APs.
            # Does the greedy need to check other constraints?
            # The condition in Inspiration 1 also had:
            # and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This checks 3-APs?
            # If a 3-AP (x-2d, x-d, x) is monochromatic, and we add x to a 4-AP...
            # Actually, the task ONLY constrains 4-APs.
            # However, if we don't constrain 3-APs, we might create a 4-AP that is bad later?
            # No, the greedy ensures that when the 4th element is added, the set size is >= 3.
            # So any 4-AP is valid.
            # The second condition in Inspiration 1 might be unnecessary for 4-APs, or it helps find a solution with fewer colors?
            # Let's stick to the 4-AP constraint only for the fallback to be safe and fast.
            # Actually, let's include the second condition to match Inspiration 1's behavior (13 colors).
            
            ok2 = True
            for d in range(1, (x - 1) // 2 + 1):
                if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                    # This would make a monochromatic 3-AP ending at x.
                    # Does this matter?
                    # If we have a monochromatic 3-AP, it doesn't violate 4-AP rules directly.
                    # But it might restrict future choices.
                    # Let's just use the 4-AP check.
                    pass
            
            if ok:
                colors.append(color)
                break
        else:
            # Should not happen if 128 colors available
            colors.append(127)
            
    return colors
