import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to find a valid coloring for n=12 first (improvement over seed 11)
    # Then try larger n if time permits.
    # The seed is 11. We want to beat it quickly.
    
    # Order of attempts: 12, 13, 14, ..., up to time limit
    # If we can't find 12, we return seed.
    
    for n in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
        if time.time() > deadline - 10:
            break
        
        # Try a few times for each n
        # For small n, 1-2 tries are often enough. For large n, more.
        max_attempts = 3 if n <= 15 else 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time()
            if remaining < 2:
                break
                
            # Allocate time for this attempt
            # For n=12, we can spend a lot of time. For n=26, less per attempt.
            time_limit = min(remaining, 20.0) if n <= 15 else min(remaining, 5.0)
            
            result = fast_local_search(n, time_limit)
            if result is not None:
                return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def fast_local_search(n, time_limit=10.0):
    """
    Optimized local search for small n.
    Uses bitmasks for neighbors.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors
    # A structured initialization often helps
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        # Simple modular coloring
        colors[idx] = (i + j) % 5
        
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    # Build initial neighbors
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Count violations
    # A violation exists if for any color c, and any pair (u, v), 
    # they share >= 2 common neighbors in color c.
    # We can maintain a count of "bad pairs" or just total violations.
    # Let's maintain `violations` as the sum over all c, u, v of C(common(c,u,v), 2).
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                # Only check v > u to avoid double counting? 
                # C(k,2) is symmetric. 
                # Summing over all u,v where u!=v counts each pair twice.
                # Let's just sum C(k,2) for all u, v.
                for v in range(n):
                    if u == v:
                        continue
                    common = (neighbors[c][u] & neighbors[c][v]).bit_count()
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2 # Divide by 2 because (u,v) and (v,u) are same pair
        
    current_violations = count_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    start_time = time.time()
    best_violations = current_violations
    best_colors = colors[:]
    
    # Simulated Annealing or Hill Climbing
    temperature = 1.0
    cooling_rate = 0.99
    min_temperature = 0.1
    
    # Precompute list of edge indices for random selection
    # random.randint is fast enough
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        # Pick a new color different from old
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta in violations
        # We need to update k[c][u][v] implicitly.
        # Instead of storing k, we can compute the change in violations
        # for pairs involving i or j.
        
        # The violation count changes only for pairs (u, v) where 
        # the common neighbor count in some color changes.
        # Changing edge (i, j) from old_c to new_c affects:
        # 1. Pairs (i, w) and (j, w) for all w.
        # 2. Specifically, the common neighbors of (i, w) in old_c decrease by 1 if j was a common neighbor.
        #    j is a common neighbor of (i, w) in old_c if w is a neighbor of i in old_c AND w is a neighbor of j in old_c?
        #    No. Common neighbors of (i, w) in color c are vertices x such that (i,x) is c and (w,x) is c.
        #    If we change (i,j) from old_c to new_c:
        #    - In old_c: j is no longer a neighbor of i.
        #      So for any w, if j was a common neighbor of (i, w) in old_c, the count drops.
        #      j is a common neighbor of (i, w) in old_c if (i,j) is old_c (true before change) and (w,j) is old_c.
        #      So we need to check neighbors of j in old_c.
        #    - In new_c: j is now a neighbor of i.
        #      So for any w, if j becomes a common neighbor of (i, w) in new_c, the count increases.
        #      This happens if (w,j) is new_c.
        
        # This logic is tricky to get right quickly. 
        # For small n, we can just recompute the local violation count for pairs involving i or j.
        
        # Let's define a function to get the violation contribution of a pair (u, v)
        def pair_violation(u, v, c):
            common = (neighbors[c][u] & neighbors[c][v]).bit_count()
            if common >= 2:
                return common * (common - 1) // 2
            return 0
        
        # Calculate old contribution for pairs involving i or j
        old_contrib = 0
        affected_pairs = set()
        for c in [old_color, new_color]:
            # Pairs (i, w) and (j, w) for all w
            for w in range(n):
                if w == i or w == j:
                    continue
                affected_pairs.add((min(i, w), max(i, w)))
                affected_pairs.add((min(j, w), max(j, w)))
                
        # Note: pair (i,j) itself doesn't have common neighbors in the same way?
        # The definition is: no two vertices with two common neighbours in one colour.
        # So we look at pairs of vertices (u,v) and check if they have >=2 common neighbors in color c.
        # The edge (i,j) color change affects the neighbor lists of i and j.
        # This affects the common neighbor counts for:
        # - Pairs (i, w) for all w
        # - Pairs (j, w) for all w
        # - Pair (i, j) itself? 
        #   Common neighbors of (i,j) in color c are vertices x such that (i,x) is c and (j,x) is c.
        #   Changing (i,j) color does NOT change (i,x) or (j,x) for x != i,j.
        #   So the common neighbors of (i,j) are unchanged.
        
        for u, v in affected_pairs:
            for c in [old_color, new_color]:
                old_contrib += pair_violation(u, v, c)
                
        # Apply change
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        colors[e_idx] = new_color
        
        new_contrib = 0
        for u, v in affected_pairs:
            for c in [old_color, new_color]:
                new_contrib += pair_violation(u, v, c)
                
        delta = new_contrib - old_contrib
        
        # Accept or reject
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_violations += delta
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert
            neighbors[new_color][i] &= ~(1 << j)
            neighbors[new_color][j] &= ~(1 << i)
            neighbors[old_color][i] |= (1 << j)
            neighbors[old_color][j] |= (1 << i)
            colors[e_idx] = old_color
            
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
            
    return None
