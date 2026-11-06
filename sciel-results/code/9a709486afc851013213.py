import time
import random

def _has_k222(colors, n):
    """
    Check if the coloring contains a monochromatic K2,2,2.
    colors: list of 0/1 for edges (i,j) with i<j.
    n: number of vertices.
    Returns True if a monochromatic K2,2,2 exists.
    """
    # Build adjacency bitmasks for each color
    # adj_c[c][u] is an integer bitmask of vertices v such that edge (u,v) has color c
    adj_c = [[0] * n for _ in range(2)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj_c[c][i] |= (1 << j)
            adj_c[c][j] |= (1 << i)
            idx += 1
            
    # Check for K2,2,2
    # Iterate over all pairs (u, v)
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                # Common neighbors in color c (bitmask)
                common = adj_c[c][u] & adj_c[c][v]
                # We need to find a K2,2 in the subgraph induced by 'common' using color c
                # K2,2 has parts {a,b} and {x,y}. Edges are a-x, a-y, b-x, b-y.
                # So we need two disjoint pairs {a,b} and {x,y} in 'common'
                # such that all 4 edges between them are color c.
                
                # Iterate over pairs {a, b} in common
                # For each pair, check if there are at least 2 other vertices in common
                # that are connected to both a and b in color c.
                
                # To iterate over pairs in a bitmask efficiently:
                # We can just iterate over all a in common, then all b > a in common.
                
                # Optimization: If the number of bits in common is small, skip.
                # bit_count is available in Python 3.10+
                if common.bit_count() < 4:
                    continue
                    
                # Iterate a in common
                a_mask = common
                while a_mask:
                    a = (a_mask & -a_mask).bit_length() - 1
                    a_mask &= a_mask - 1
                    
                    # Iterate b in common, b > a
                    b_mask = common & (~((1 << (a + 1)) - 1))
                    while b_mask:
                        b = (b_mask & -b_mask).bit_length() - 1
                        b_mask &= b_mask - 1
                        
                        # Find vertices in common connected to both a and b in color c
                        # N_a_in_common = adj_c[c][a] & common
                        # N_b_in_common = adj_c[c][b] & common
                        # Common2 = N_a_in_common & N_b_in_common
                        # Remove a and b from Common2
                        # If bit_count(Common2) >= 2, return True
                        
                        n_a = adj_c[c][a] & common
                        n_b = adj_c[c][b] & common
                        common2 = n_a & n_b
                        # Exclude a and b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        
                        if common2.bit_count() >= 2:
                            return True
    return False

def _paley_coloring(n):
    """
    Generate Paley coloring for prime n = 4k+1.
    """
    if n % 4 != 1:
        return None
    squares = {x*x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _count_violations_fast(colors, n):
    """
    Count the number of monochromatic K2,2,2 subgraphs (or a proxy metric).
    Returns a count. 0 means valid.
    """
    adj_c = [[0] * n for _ in range(2)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj_c[c][i] |= (1 << j)
            adj_c[c][j] |= (1 << i)
            idx += 1
            
    count = 0
    for u in range(n):
        for v in range(u+1, n):
            for c in [0, 1]:
                common = adj_c[c][u] & adj_c[c][v]
                if common.bit_count() < 4:
                    continue
                
                a_mask = common
                while a_mask:
                    a = (a_mask & -a_mask).bit_length() - 1
                    a_mask &= a_mask - 1
                    
                    b_mask = common & (~((1 << (a + 1)) - 1))
                    while b_mask:
                        b = (b_mask & -b_mask).bit_length() - 1
                        b_mask &= b_mask - 1
                        
                        n_a = adj_c[c][a] & common
                        n_b = adj_c[c][b] & common
                        common2 = n_a & n_b
                        common2 &= ~(1 << a)
                        common2 &= ~(1 << b)
                        
                        cnt = common2.bit_count()
                        if cnt >= 2:
                            count += 1
                            break
                    if count > 0: # Early exit? No, we want a count for gradient.
                        # But for speed, if we just want to know if it's 0, we can stop.
                        # However, for SA, a count is better.
                        # Let's continue counting but maybe cap it?
                        pass
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try n=29 first
    n = 29
    colors = _paley_coloring(n)
    
    if colors is not None:
        if not _has_k222(colors, n):
            return {'n': n, 'colors': colors}
            
    # If Paley 29 fails, try local search on 29
    # Start from Paley 29 or random
    best_colors = colors
    best_cost = _count_violations_fast(colors, n)
    
    current_colors = colors[:]
    current_cost = best_cost
    
    if current_cost == 0:
        return {'n': n, 'colors': current_colors}
        
    # Local Search
    # Simulated Annealing or Simple Hill Climbing with Random Restarts
    # Since n=29 is small, we can afford many steps.
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        if restart == 0:
            curr = current_colors[:]
            curr_cost = current_cost
        else:
            # Random restart
            curr = [random.choice([0,1]) for _ in range(len(current_colors))]
            curr_cost = _count_violations_fast(curr, n)
            
        if curr_cost == 0:
            return {'n': n, 'colors': curr}
            
        # If this restart's cost is better than best, update best
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_colors = curr[:]
            
        temp = 1.0
        max_steps = 20000 # Increase steps due to faster cost function
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if curr_cost == 0:
                return {'n': n, 'colors': curr}
                
            # Pick random edge
            edge_idx = random.randint(0, len(curr)-1)
            old_c = curr[edge_idx]
            new_c = 1 - old_c
            curr[edge_idx] = new_c
            
            new_cost = _count_violations_fast(curr, n)
            
            if new_cost < curr_cost:
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_colors = curr[:]
            else:
                # Accept with probability exp(-delta/temp)
                delta = new_cost - curr_cost
                if delta == 0:
                    curr_cost = new_cost
                else:
                    prob = pow(2.71828, -delta/temp)
                    if random.random() > prob:
                        # Reject
                        curr[edge_idx] = old_c
                    else:
                        curr_cost = new_cost
            
            temp *= 0.999 # Slower cooling for longer search
            
    # If we found a valid one in search, best_cost would be 0.
    # But we return early if curr_cost becomes 0.
    # So if we exit the loop, best_cost > 0.
    
    # Check if best_colors is valid (just in case)
    if not _has_k222(best_colors, n):
        return {'n': n, 'colors': best_colors}
        
    # If 29 failed, try 28?
    # But 29 is the record, so it should work.
    # If we run out of time, return the best we have?
    # The problem asks to maximize n.
    # If we can't get 29, we should return a valid smaller n.
    # But we don't have a valid smaller n guaranteed.
    # We can construct a valid n=17 (Paley).
    # Let's fall back to n=17 if 29 fails.
    
    colors_17 = _paley_coloring(17)
    if colors_17 and not _has_k222(colors_17, 17):
        return {'n': 17, 'colors': colors_17}
        
    # Fallback: n=1
    return {'n': 1, 'colors': []}
