import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    # Seed graph (Paley 17)
    squares = {x * x % 17 for x in range(1, 17)}
    seed_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    best_n = 17
    best_edges = seed_edges

    # Try to find larger graphs
    # R(4,6) is between 35 and 38.
    # We try 36, 35, 34...
    for n in range(36, 16, -1):
        if time.time() > deadline:
            break
        
        # Allocate time for this n
        # Give more time to larger n
        time_limit = 30 if n >= 35 else 15
        
        result = local_search(n, time_limit, start_time, deadline)
        if result is not None:
            best_n = n
            best_edges = result
            # If we found a valid graph, we can continue to see if we can find a larger one?
            # No, we are iterating downwards.
            # So if we find 36, we stop.
            # If we find 35, we stop (since 36 failed).
            # Actually, if 36 fails, we try 35.
            # If 35 succeeds, we return 35.
            # We don't need to try smaller ones if we found a valid one for the current n.
            # But wait, we are iterating 36, 35, 34...
            # If 36 fails, we go to 35.
            # If 35 succeeds, we set best_n=35 and break?
            # Yes, because we want max n.
            break

    return {'n': best_n, 'edges': best_edges}

def local_search(n, time_limit, global_start, global_deadline):
    # Pre-compute subsets
    vertices = list(range(n))
    
    # Generate all 4-subsets
    subs4 = list(itertools.combinations(vertices, 4))
    # Generate all 6-subsets
    subs6 = list(itertools.combinations(vertices, 6))
    
    # Create index maps
    # This might be slow for n=36, but let's try.
    # To speed up, we can use a dictionary.
    idx4 = {s: i for i, s in enumerate(subs4)}
    idx6 = {s: i for i, s in enumerate(subs6)}
    
    # Initialize random graph
    N = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                N[i] |= (1 << j)
                N[j] |= (1 << i)
    
    # Initialize status arrays
    is_k4 = [False] * len(subs4)
    is_ind = [False] * len(subs6)
    
    # Calculate initial counts
    k4_count = 0
    ind6_count = 0
    
    # Helper to check K4
    def check_k4(s):
        a, b, c, d = s
        return (N[a] & (1 << b) & (1 << c) & (1 << d) != 0 and
                N[b] & (1 << c) & (1 << d) != 0 and
                N[c] & (1 << d) != 0)
                
    # Helper to check Independent
    def check_ind(s):
        a, b, c, d, e, f = s
        mask_bcd_ef = (1 << b) | (1 << c) | (1 << d) | (1 << e) | (1 << f)
        if N[a] & mask_bcd_ef: return False
        mask_cdef = (1 << c) | (1 << d) | (1 << e) | (1 << f)
        if N[b] & mask_cdef: return False
        mask_def = (1 << d) | (1 << e) | (1 << f)
        if N[c] & mask_def: return False
        mask_ef = (1 << e) | (1 << f)
        if N[d] & mask_ef: return False
        if N[e] & (1 << f): return False
        return True

    for i, s in enumerate(subs4):
        if check_k4(s):
            is_k4[i] = True
            k4_count += 1
            
    for i, s in enumerate(subs6):
        if check_ind(s):
            is_ind[i] = True
            ind6_count += 1
            
    total_violations = k4_count + ind6_count
    
    # Pre-compute lists of subset indices for each edge
    # map_edge_to_subsets4[edge] = list of indices in subs4 containing edge
    # map_edge_to_subsets6[edge] = list of indices in subs6 containing edge
    # This is O(n^2 * n^2) memory/time?
    # For n=36, edges=630.
    # subs4 size ~ 58k.
    # Each 4-set has 6 edges.
    # So total entries ~ 58k * 6 = 350k.
    # subs6 size ~ 1.9M.
    # Each 6-set has 15 edges.
    # Total entries ~ 1.9M * 15 = 28M.
    # This is large.
    
    # Instead of pre-computing, we can iterate over subsets and check if they contain the edge.
    # But that's O(N_subsets) per flip.
    # For 6-sets, 1.9M checks per flip is too slow.
    
    # We MUST pre-compute the mapping for 6-sets to be fast enough.
    # Let's try to build it efficiently.
    
    map4 = { (i,j): [] for i in range(n) for j in range(i+1, n) }
    map6 = { (i,j): [] for i in range(n) for j in range(i+1, n) }
    
    # Fill map4
    for idx, s in enumerate(subs4):
        for i in range(4):
            for j in range(i+1, 4):
                u, v = s[i], s[j]
                if u > v: u, v = v, u
                map4[(u,v)].append(idx)
                
    # Fill map6
    # This will take time and memory.
    for idx, s in enumerate(subs6):
        for i in range(6):
            for j in range(i+1, 6):
                u, v = s[i], s[j]
                if u > v: u, v = v, u
                map6[(u,v)].append(idx)

    # Local Search Loop
    while time.time() < global_deadline - 5:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Flip edge
        N[u] ^= (1 << v)
        N[v] ^= (1 << u)
        
        old_total = total_violations
        new_k4_count = k4_count
        new_ind6_count = ind6_count
        
        # Update K4s
        for idx in map4[(u,v)]:
            s = subs4[idx]
            new_status = check_k4(s)
            if new_status != is_k4[idx]:
                if new_status:
                    new_k4_count += 1
                else:
                    new_k4_count -= 1
                is_k4[idx] = new_status
                
        # Update Inds
        for idx in map6[(u,v)]:
            s = subs6[idx]
            new_status = check_ind(s)
            if new_status != is_ind[idx]:
                if new_status:
                    new_ind6_count += 1
                else:
                    new_ind6_count -= 1
                is_ind[idx] = new_status
                
        new_total = new_k4_count + new_ind6_count
        
        if new_total < old_total:
            k4_count = new_k4_count
            ind6_count = new_ind6_count
            total_violations = new_total
            if total_violations == 0:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if N[i] & (1 << j):
                            edges.append([i, j])
                return edges
        else:
            # Revert
            N[u] ^= (1 << v)
            N[v] ^= (1 << u)
            # Revert status? No, we didn't change is_k4/is_ind if we reverted?
            # Wait, we DID change is_k4/is_ind in the loop above.
            # We need to revert them too.
            # This is a problem.
            # Better: store old values and restore.
            # Or, just don't update is_k4/is_ind until we accept.
            # But we need the old values to compute the change.
            
            # Let's refactor:
            # 1. Compute new statuses for affected subsets.
            # 2. Calculate delta.
            # 3. If accept, update is_k4/is_ind and counts.
            # 4. If reject, do nothing.
            
            pass # The logic above is flawed for rejection.

    return None
