import time
import random
from itertools import combinations

def _has_k222_fast(adj, n):
    """
    Check if graph with adjacency bitmask list adj on n vertices contains K2,2,2.
    Returns True if found.
    """
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            for c in range(b+1, n):
                if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                    continue
                for d in range(c+1, n):
                    if not (adj[a] >> d) & 1 or not (adj[b] >> d) & 1:
                        continue
                    cn_mask = adj[a] & adj[b] & adj[c] & adj[d]
                    cn_mask &= ~(1 << a)
                    cn_mask &= ~(1 << b)
                    cn_mask &= ~(1 << c)
                    cn_mask &= ~(1 << d)
                    if cn_mask.bit_count() >= 2:
                        return True
    return False

def _is_valid_coloring(adj, n):
    """Check if neither color class contains K2,2,2."""
    if _has_k222_fast(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222_fast(comp, n):
        return False
    return True

def _build_paley29():
    """Build Paley graph on 29 vertices."""
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    
    n = q
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_violations_extended(adj, n, new_v):
    """
    Count K2,2,2 subgraphs involving new_v in either color.
    adj is the full n-vertex adjacency list.
    """
    violations = 0
    # Check Red
    # new_v is in Part A.
    # a in A (not connected to new_v in Red)
    # b1, b2 in B (connected to new_v in Red)
    # c1, c2 in C (connected to new_v in Red)
    
    red_neighbors = []
    non_red_neighbors = []
    for i in range(n-1):
        if (adj[new_v] >> i) & 1:
            red_neighbors.append(i)
        else:
            non_red_neighbors.append(i)
            
    # Precompute adjacency for speed
    # We need to check edges between these specific vertices
    
    for a in non_red_neighbors:
        # a is in A with new_v.
        # We need b1, b2 in red_neighbors, c1, c2 in red_neighbors
        # such that:
        # 1. (a, b1) in R, (a, b2) in R
        # 2. (a, c1) in R, (a, c2) in R
        # 3. (b1, c1) in R, (b1, c2) in R, (b2, c1) in R, (b2, c2) in R
        # 4. (b1, b2) NOT in R
        # 5. (c1, c2) NOT in R
        
        # Filter red_neighbors that are connected to a in Red
        candidates_b = [b for b in red_neighbors if (adj[a] >> b) & 1]
        candidates_c = [c for c in red_neighbors if (adj[a] >> c) & 1]
        
        # We need to pick 2 from candidates_b and 2 from candidates_c
        # Note: b and c are distinct vertices.
        
        for b1, b2 in combinations(candidates_b, 2):
            if (adj[b1] >> b2) & 1: # (b1, b2) must NOT be in R
                continue
                
            # Now find c1, c2 from candidates_c
            # c1, c2 must be connected to b1 and b2 in R
            # And c1, c2 must NOT be connected in R
            
            potential_c = []
            for c in candidates_c:
                if c == b1 or c == b2:
                    continue
                if (adj[b1] >> c) & 1 and (adj[b2] >> c) & 1:
                    potential_c.append(c)
            
            for c1, c2 in combinations(potential_c, 2):
                if (adj[c1] >> c2) & 1: # (c1, c2) must NOT be in R
                    continue
                violations += 1
                
    # Check Blue
    # In Blue graph, "edge" means "not in Red"
    # new_v in A (Blue).
    # a in A (Blue) => (new_v, a) NOT in Blue => (new_v, a) IN Red.
    # So a is in red_neighbors.
    # b1, b2 in B (Blue) => (new_v, b1) IN Blue => (new_v, b1) NOT IN Red.
    # So b1, b2 are in non_red_neighbors.
    # c1, c2 in C (Blue) => (new_v, c1) IN Blue => (new_v, c1) NOT IN Red.
    # So c1, c2 are in non_red_neighbors.
    
    for a in red_neighbors:
        # a is in A with new_v in Blue.
        # b1, b2, c1, c2 are in non_red_neighbors.
        
        # Conditions in Blue:
        # (a, b1) in Blue => (a, b1) NOT IN Red
        # (a, b2) in Blue => (a, b2) NOT IN Red
        # (a, c1) in Blue => (a, c1) NOT IN Red
        # (a, c2) in Blue => (a, c2) NOT IN Red
        # (b1, c1) in Blue => (b1, c1) NOT IN Red
        # ...
        # (b1, b2) NOT IN Blue => (b1, b2) IN Red
        # (c1, c2) NOT IN Blue => (c1, c2) IN Red
        
        candidates_b = [b for b in non_red_neighbors if not ((adj[a] >> b) & 1)]
        candidates_c = [c for c in non_red_neighbors if not ((adj[a] >> c) & 1)]
        
        for b1, b2 in combinations(candidates_b, 2):
            if not ((adj[b1] >> b2) & 1): # (b1, b2) MUST be in Red
                continue
                
            potential_c = []
            for c in candidates_c:
                if c == b1 or c == b2:
                    continue
                # (b1, c) NOT IN Red and (b2, c) NOT IN Red
                if not ((adj[b1] >> c) & 1) and not ((adj[b2] >> c) & 1):
                    potential_c.append(c)
            
            for c1, c2 in combinations(potential_c, 2):
                if not ((adj[c1] >> c2) & 1): # (c1, c2) MUST be in Red
                    continue
                violations += 1
                
    return violations

def _try_extend_to_30(base_adj, deadline):
    n_base = 29
    n_target = 30
    random.seed(42)
    
    best_x = None
    best_violations = float('inf')
    
    # Precompute base_adj for speed
    # adj[i] is bitmask of neighbors in Red for i in 0..28
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Initialize x
        # Strategy: Random, but biased towards degrees that might work
        x = [random.randint(0, 1) for _ in range(n_base)]
        
        # Build adj for 30 vertices
        adj = [0] * n_target
        for i in range(n_base):
            adj[i] = base_adj[i]
        for i in range(n_base):
            if x[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
                
        v = _count_violations_extended(adj, n_target, 29)
        
        if v == 0:
            return x[:]
            
        if v < best_violations:
            best_violations = v
            best_x = x[:]
            
        # Local Search (Simulated Annealing)
        current_x = x[:]
        current_v = v
        temperature = 10.0
        cooling_rate = 0.995
        
        steps = 0
        max_steps = 2000
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            i = random.randint(0, n_base - 1)
            
            # Flip i
            old_val = current_x[i]
            current_x[i] = 1 - old_val
            
            # Update adj
            if current_x[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
            else:
                adj[29] &= ~(1 << i)
                adj[i] &= ~(1 << 29)
                
            new_v = _count_violations_extended(adj, n_target, 29)
            
            delta = new_v - current_v
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (random.random() < pow(2.718, -delta / temperature)):
                current_v = new_v
                if current_v == 0:
                    return current_x[:]
                if current_v < best_violations:
                    best_violations = current_v
                    best_x = current_x[:]
            else:
                # Revert
                current_x[i] = old_val
                if old_val == 1:
                    adj[29] |= (1 << i)
                    adj[i] |= (1 << 29)
                else:
                    adj[29] &= ~(1 << i)
                    adj[i] &= ~(1 << 29)
                    
            temperature *= cooling_rate
            steps += 1

    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    base_adj = _build_paley29()
    
    if not _is_valid_coloring(base_adj, 29):
        n = 29
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(1 if (base_adj[i] >> j) & 1 else 0)
        return {'n': n, 'colors': colors}
        
    result_edges = _try_extend_to_30(base_adj, deadline)
    
    if result_edges is not None:
        n = 30
        adj = [0] * n
        for i in range(29):
            adj[i] = base_adj[i]
        for i in range(29):
            if result_edges[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
                
        if _is_valid_coloring(adj, n):
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(1 if (adj[i] >> j) & 1 else 0)
            return {'n': n, 'colors': colors}
            
    n = 29
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(1 if (base_adj[i] >> j) & 1 else 0)
    return {'n': n, 'colors': colors}
