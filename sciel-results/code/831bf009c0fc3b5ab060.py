import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges) # 120
    
    k4_edges_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4_edges_list.append([e1, e2, e3, e4, e5, e6])
    
    num_k4s = len(k4_edges_list) # 1820
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges_list):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)

    def get_seed_17():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    def compress_colors(colors_17, k):
        # Map 17 colors to k colors
        # Simple strategy: merge colors randomly or by index
        # To preserve structure, we can try to map adjacent colors in mod 17 to same color in mod k?
        # Or just random mapping.
        # A better mapping: group colors.
        # If k=15, merge 2 pairs.
        # If k=14, merge 3 pairs.
        
        mapping = list(range(17))
        # Shuffle to break symmetry?
        # random.shuffle(mapping)
        
        # Create target colors
        # We want to map 17 items to k bins.
        # Let's just use modulo k on the mapped indices?
        # Or simply: for c in 0..16, new_c = c % k?
        # This merges 15->0, 16->1 for k=15.
        # This is deterministic and fast.
        
        return [c % k for c in colors_17]

    def solve_fast(k, time_limit, init_colors=None):
        end_time = time.time() + time_limit
        
        if init_colors is None:
            # Try a few random inits if no good init provided
            best_colours = None
            best_bad = float('inf')
            
            while time.time() < end_time:
                # Random init
                colours = [random.randint(0, k - 1) for _ in range(num_edges)]
                # ... (run local search)
                pass
            return best_colours
        
        # Use provided init
        colours = init_colors[:]
        
        # Initialize state
        k4_counts = [[0]*k for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        bad_k4s = set()
        
        for k4_idx, e_list in enumerate(k4_edges_list):
            for e in e_list:
                k4_counts[k4_idx][colours[e]] += 1
            
            distinct = 0
            for c in range(k):
                if k4_counts[k4_idx][c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                bad_k4s.add(k4_idx)
        
        if not bad_k4s:
            return colours
        
        T = 1.0
        cool_rate = 0.9995
        min_T = 0.01
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < end_time and bad_k4s:
            if T < min_T:
                break
            
            # Pick a random bad K4
            # To avoid list(bad_k4s) cost, we can keep a list of bad k4s and clean it up periodically
            # Or just use list conversion if size is small. Initially it might be large.
            # Let's use a list for bad_k4s to make random choice O(1)
            # But we need to handle removals.
            # For now, list conversion is O(N_bad). If N_bad is large, it's slow.
            # Let's try to keep it as a list.
            
            # Convert to list once per step? No, that's slow.
            # Let's just pick a random index from 0 to len(bad_k4s)-1 if we had a list.
            # Since we are using a set, let's convert.
            # Optimization: If len(bad_k4s) is small, it's fast.
            # If it's large, we are in early stages.
            
            bad_list = list(bad_k4s)
            k4_idx = random.choice(bad_list)
            e_list = k4_edges_list[k4_idx]
            e_idx = random.choice(e_list)
            
            old_c = colours[e_idx]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue
            
            affected = edge_to_k4s[e_idx]
            delta = 0
            accept = False
            
            # Check delta
            # We need to simulate the change to see if it helps.
            # We will calculate the change in distinct count for each affected K4.
            
            changes = [] # (ak, old_distinct, new_distinct)
            
            for ak in affected:
                cnts = k4_counts[ak]
                old_dist = k4_distinct[ak]
                
                # Apply
                cnts[old_c] -= 1
                cnts[new_c] += 1
                
                new_dist = old_dist
                if cnts[old_c] == 0:
                    new_dist -= 1
                if cnts[new_c] == 1: # Was 0, now 1
                    new_dist += 1
                
                changes.append((ak, old_dist, new_dist))
                
                # Revert for now
                cnts[old_c] += 1
                cnts[new_c] -= 1
                
                # Update delta
                was_bad = (old_dist < 5)
                is_bad = (new_dist < 5)
                
                if was_bad and not is_bad:
                    delta -= 1
                elif not was_bad and is_bad:
                    delta += 1
            
            # Acceptance
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5 # Accept half of neutral moves
            else:
                if random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                colours[e_idx] = new_c
                for ak, old_dist, new_dist in changes:
                    cnts = k4_counts[ak]
                    cnts[old_c] -= 1
                    cnts[new_c] += 1
                    
                    k4_distinct[ak] = new_dist
                    
                    was_bad = (old_dist < 5)
                    is_bad = (new_dist < 5)
                    
                    if was_bad and not is_bad:
                        bad_k4s.discard(ak)
                    elif not was_bad and is_bad:
                        bad_k4s.add(ak)
                
                if delta < 0:
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T *= cool_rate
            
            if no_improve > max_no_improve:
                break
        
        return colours if not bad_k4s else None

    # Strategy:
    # 1. Try k=15 with compressed seed init.
    # 2. Try k=14 with compressed seed init.
    # 3. Fallback to 17.
    
    seed_17 = get_seed_17()
    
    # Try 15
    if time.time() < deadline - 20:
        init_15 = compress_colors(seed_17, 15)
        res = solve_fast(15, 30, init_15)
        if res is not None:
            return res
    
    # Try 14
    if time.time() < deadline - 10:
        init_14 = compress_colors(seed_17, 14)
        res = solve_fast(14, 20, init_14)
        if res is not None:
            return res
    
    # Fallback
    return seed_17
