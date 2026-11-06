import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a valid coloring for n=26, 25, 24
    for n in [26, 25, 24]:
        if time.time() > deadline:
            break
        
        result = local_search(n, time_limit=10)
        if result is not None:
            return result
            
    # Fallback to a known valid small n
    # n=11 is valid with the provided formula
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search(n, time_limit=10):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge indices
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize neighbor masks
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    def calc_violations():
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        violations += k * (k - 1) // 2
        return violations

    current_violations = calc_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.99
    
    while time.time() < start_time + time_limit:
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta in violations
        delta = 0
        
        # 1. Remove from old_color
        c = old_color
        # For each v in N_c(j) \ {i}, j is a common neighbor of i and v.
        # Removing j from N_c(i) reduces k_iv by 1.
        # Change in binom(k,2) is -(k-1).
        
        mask_j = neighbors[c][j]
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # For each v in N_c(i) \ {j}, i is a common neighbor of j and v.
        # Removing i from N_c(j) reduces k_jv by 1.
        # Change in binom(k,2) is -(k-1).
        
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta -= (k - 1)
            m >>= 1
            v += 1
            
        # Apply removal to masks
        neighbors[c][i] &= ~(1 << j)
        neighbors[c][j] &= ~(1 << i)
        
        # 2. Add to new_color
        c = new_color
        # For each v in N_c(j) \ {i}, j becomes a common neighbor of i and v.
        # k_iv increases by 1.
        # Change in binom(k,2) is k.
        
        mask_j = neighbors[c][j]
        v = 0
        m = mask_j
        while m:
            if m & 1:
                if v != i:
                    k = (neighbors[c][i] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # For each v in N_c(i) \ {j}, i becomes a common neighbor of j and v.
        # k_jv increases by 1.
        # Change in binom(k,2) is k.
        
        mask_i = neighbors[c][i]
        v = 0
        m = mask_i
        while m:
            if m & 1:
                if v != j:
                    k = (neighbors[c][j] & neighbors[c][v]).bit_count()
                    delta += k
            m >>= 1
            v += 1
            
        # Apply addition to masks
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
        # Update colors array
        colors[e_idx] = new_color
        
        new_violations = current_violations + delta
        
        # Accept if better, or with probability exp(-delta / temperature)
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            neighbors[c][i] &= ~(1 << j)
            neighbors[c][j] &= ~(1 << i)
            
            c = old_color
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    # If we didn't find 0 violations, return None to try smaller n
    # Or return the best found? The task says "Maximize n".
    # If we don't find a valid one for n, we should try n-1.
    return None
