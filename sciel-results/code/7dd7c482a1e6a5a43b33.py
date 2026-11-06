import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    positions = list(range(1, n+1))
    
    # Precompute all nonconstant 4-APs
    ap_list = []
    for a in range(1, n+1):
        for d in range(1, n):
            if a + 3*d <= n:
                ap_list.append((a, a+d, a+2*d, a+3*d))
            else:
                break
    
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Map position to list of AP indices it belongs to
    pos_to_aps = {i: [] for i in positions}
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Convert to lists for faster iteration in Python loops if needed, or keep as np arrays
    # Using lists of ints is often faster for small counts in Python than numpy indexing for single elements
    pos_to_aps_list = {pos: pos_to_aps[pos] for pos in positions}

    def count_violations(colors):
        """
        colors: list of ints (0-indexed, length n)
        returns: number of violated APs
        """
        v_count = 0
        for ap in ap_list:
            c0 = colors[ap[0]-1]
            c1 = colors[ap[1]-1]
            c2 = colors[ap[2]-1]
            c3 = colors[ap[3]-1]
            if len({c0, c1, c2, c3}) < 3:
                v_count += 1
        return v_count

    def try_local_search(k, max_iters=20000, seed=0):
        rng = random.Random(seed)
        
        # Greedy initialization
        colors = [0] * n
        for p in range(1, n+1):
            best_c = 0
            best_v = float('inf')
            for c in range(k):
                colors[p-1] = c
                # Check only APs ending at p or containing p? 
                # Simpler: just count violations involving p
                v = 0
                for ap_idx in pos_to_aps_list[p]:
                    ap = ap_list[ap_idx]
                    c0 = colors[ap[0]-1]
                    c1 = colors[ap[1]-1]
                    c2 = colors[ap[2]-1]
                    c3 = colors[ap[3]-1]
                    if len({c0, c1, c2, c3}) < 3:
                        v += 1
                if v < best_v:
                    best_v = v
                    best_c = c
                if v == 0:
                    break
            colors[p-1] = best_c
        
        # Calculate initial violations
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors[:]
        
        # Precompute APs as tuples for fast access in inner loop
        # We need to update violations incrementally.
        # Maintain a set of violated AP indices? Or just recompute?
        # With ~2000 APs, recomputing all is ~2000 * 4 ops = 8000 ops.
        # In Python, this might be slow if done 20000 times.
        # 20000 * 8000 = 160,000,000 ops. Might be too slow in Python (10-20s+).
        # Let's use incremental updates.
        
        violated = np.zeros(num_aps, dtype=bool)
        for idx, ap in enumerate(ap_list):
            c0 = colors[ap[0]-1]
            c1 = colors[ap[1]-1]
            c2 = colors[ap[2]-1]
            c3 = colors[ap[3]-1]
            if len({c0, c1, c2, c3}) < 3:
                violated[idx] = True
        
        num_violations = int(np.sum(violated))
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if num_violations == 0:
                return colors[:]
            
            # Pick a random violated AP
            v_indices = np.where(violated)[0]
            if len(v_indices) == 0:
                return colors[:]
            
            ap_idx = int(v_indices[rng.randint(0, len(v_indices)-1)])
            ap = ap_list[ap_idx]
            
            # Try to fix by changing one position in this AP
            pos_order = [ap[0], ap[1], ap[2], ap[3]]
            rng.shuffle(pos_order)
            
            improved = False
            for pos in pos_order:
                old_color = colors[pos-1]
                for c in range(k):
                    if c == old_color:
                        continue
                    
                    colors[pos-1] = c
                    
                    # Update violations for all APs containing pos
                    delta = 0
                    for ap_i in pos_to_aps_list[pos]:
                        # Check if this AP was violated
                        was_v = violated[ap_i]
                        
                        # Check new status
                        a_t = ap_list[ap_i]
                        c0 = colors[a_t[0]-1]
                        c1 = colors[a_t[1]-1]
                        c2 = colors[a_t[2]-1]
                        c3 = colors[a_t[3]-1]
                        is_v = len({c0, c1, c2, c3}) < 3
                        
                        if was_v and not is_v:
                            delta -= 1
                        elif not was_v and is_v:
                            delta += 1
                        
                        violated[ap_i] = is_v
                    
                    new_num_violations = num_violations + delta
                    
                    if new_num_violations < num_violations:
                        num_violations = new_num_violations
                        improved = True
                        break
                    else:
                        # Revert
                        colors[pos-1] = old_color
                        # Revert violated status
                        for ap_i in pos_to_aps_list[pos]:
                            a_t = ap_list[ap_i]
                            c0 = colors[a_t[0]-1]
                            c1 = colors[a_t[1]-1]
                            c2 = colors[a_t[2]-1]
                            c3 = colors[a_t[3]-1]
                            violated[ap_i] = len({c0, c1, c2, c3}) < 3
                        break
                
                if improved:
                    break
            
            if not improved:
                # Kick: pick a random position from the violated AP and assign a random color
                pos = pos_order[rng.randint(0, len(pos_order)-1)]
                new_c = rng.randint(0, k-1)
                colors[pos-1] = new_c
                
                # Update violations
                for ap_i in pos_to_aps_list[pos]:
                    a_t = ap_list[ap_i]
                    c0 = colors[a_t[0]-1]
                    c1 = colors[a_t[1]-1]
                    c2 = colors[a_t[2]-1]
                    c3 = colors[a_t[3]-1]
                    violated[ap_i] = len({c0, c1, c2, c3}) < 3
                
                num_violations = int(np.sum(violated))
                
                if num_violations == 0:
                    return colors[:]

        return None

    # Try k=3
    best_result = None
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        result = try_local_search(3, max_iters=5000, seed=seed)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result

    # Try k=4
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        result = try_local_search(4, max_iters=5000, seed=seed)
        if result is not None:
            return result

    # Fallback: Return a known valid coloring if found in literature or simple construction
    # If search fails, we must return something. 
    # A random coloring with 128 colors is valid?
    # If all colors are distinct, any 4-AP has 4 distinct colors. Valid.
    # But score is 1/128 = 0.007.
    # Better to return the best invalid one? No, must be valid.
    # So we must find a valid one.
    
    # If local search fails, try a simple greedy that ensures validity by using many colors
    # Greedy: assign smallest color that doesn't create a violation with previous APs?
    # This might use many colors but ensures validity if we check all constraints.
    
    colors = []
    for x in range(1, n+1):
        found = False
        for c in range(128):
            # Check if assigning c to x creates any violation
            # We need to check all APs containing x.
            # Some positions in AP might be > x (future). 
            # If we only look at past, we might fail.
            # But if we use distinct colors for all, it works.
            # Let's just use distinct colors to guarantee validity as a last resort.
            # But we want to minimize.
            # The local search should have found 3 or 4.
            # If it didn't, something is wrong with the search.
            pass
        
    # Last resort: Distinct colors
    return list(range(128))
