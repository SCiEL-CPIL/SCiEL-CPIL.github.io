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

def _count_violations_29(adj, n, base_adj):
    """
    Count the number of K2,2,2 subgraphs involving vertex n-1 (the new vertex)
    in either the red or blue graph.
    """
    new_v = n - 1
    violations = 0
    
    # Check Red graph violations involving new_v
    # Neighbors of new_v in Red
    red_neighbors = []
    for i in range(n-1):
        if (adj[new_v] >> i) & 1:
            red_neighbors.append(i)
    
    # For each 4-subset of red_neighbors, check if they form a K2,2,2 - v with new_v
    # Structure: new_v is in Part A. Neighbors are in Parts B and C.
    # We need to partition the 4 neighbors into two pairs {b1, b2} and {c1, c2}
    # such that:
    # 1. No edge between b1, b2 in Red
    # 2. No edge between c1, c2 in Red
    # 3. All 4 edges between {b1, b2} and {c1, c2} exist in Red
    
    if len(red_neighbors) >= 4:
        for quad in combinations(red_neighbors, 4):
            u, v, w, x = quad
            # Possible pairings: (u,v) & (w,x), (u,w) & (v,x), (u,x) & (v,w)
            # Check pairing 1: {u,v} and {w,x}
            if not ((adj[u] >> v) & 1) and not ((adj[w] >> x) & 1):
                # Check cross edges
                if ((adj[u] >> w) & 1) and ((adj[u] >> x) & 1) and \
                   ((adj[v] >> w) & 1) and ((adj[v] >> x) & 1):
                    violations += 1
            
            # Check pairing 2: {u,w} and {v,x}
            if not ((adj[u] >> w) & 1) and not ((adj[v] >> x) & 1):
                if ((adj[u] >> v) & 1) and ((adj[u] >> x) & 1) and \
                   ((adj[w] >> v) & 1) and ((adj[w] >> x) & 1):
                    violations += 1
            
            # Check pairing 3: {u,x} and {v,w}
            if not ((adj[u] >> x) & 1) and not ((adj[v] >> w) & 1):
                if ((adj[u] >> v) & 1) and ((adj[u] >> w) & 1) and \
                   ((adj[x] >> v) & 1) and ((adj[x] >> w) & 1):
                    violations += 1

    # Check Blue graph violations involving new_v
    # Neighbors of new_v in Blue
    blue_neighbors = []
    for i in range(n-1):
        if not ((adj[new_v] >> i) & 1):
            blue_neighbors.append(i)
            
    # The complement graph logic applies similarly
    # In Blue graph, edges are where adj has 0.
    # So we check for K2,2,2 in the complement of adj restricted to blue_neighbors.
    
    if len(blue_neighbors) >= 4:
        for quad in combinations(blue_neighbors, 4):
            u, v, w, x = quad
            # Check pairing 1: {u,v} and {w,x}
            # In Blue, "no edge" means "edge in adj"
            # In Blue, "edge" means "no edge in adj"
            
            # Condition:
            # 1. u,v connected in Blue? NO (they are in same part B). So u,v MUST be connected in Red (adj).
            # 2. w,x connected in Blue? NO. So w,x MUST be connected in Red (adj).
            # 3. u,w connected in Blue? YES. So u,w MUST NOT be connected in Red (adj).
            
            if ((adj[u] >> v) & 1) and ((adj[w] >> x) & 1):
                if not ((adj[u] >> w) & 1) and not ((adj[u] >> x) & 1) and \
                   not ((adj[v] >> w) & 1) and not ((adj[v] >> x) & 1):
                    violations += 1
            
            # Check pairing 2: {u,w} and {v,x}
            if ((adj[u] >> w) & 1) and ((adj[v] >> x) & 1):
                if not ((adj[u] >> v) & 1) and not ((adj[u] >> x) & 1) and \
                   not ((adj[w] >> v) & 1) and not ((adj[w] >> x) & 1):
                    violations += 1
            
            # Check pairing 3: {u,x} and {v,w}
            if ((adj[u] >> x) & 1) and ((adj[v] >> w) & 1):
                if not ((adj[u] >> v) & 1) and not ((adj[u] >> w) & 1) and \
                   not ((adj[x] >> v) & 1) and not ((adj[x] >> w) & 1):
                    violations += 1

    return violations

def _compute_violation_delta(base_adj, n_base, flip_idx, current_x):
    """
    Compute the change in violations when flipping vertex flip_idx from current_x.
    Returns delta_violations (new_violations - old_violations).
    """
    new_v = n_base  # vertex 29
    
    # Old state: x[flip_idx] determines edge to new_v
    old_is_red = current_x[flip_idx]
    
    # Precompute: for each other vertex i, whether it's connected to new_v in old state
    # and whether flip_idx is connected to i in base graph
    
    # We need to count violations that involve flip_idx and new_v
    # A violation involves 4 neighbors of new_v (in red or blue) that form a K2,2,2 with new_v
    
    # Let's compute violations involving flip_idx in the old and new states
    
    # Old violations involving flip_idx:
    # flip_idx is a neighbor of new_v in color old_color
    # We need to find 3 other neighbors of new_v in the same color as flip_idx
    # such that together with flip_idx they form a valid K2,2,2 partition
    
    old_violations = 0
    new_violations = 0
    
    # In old state, flip_idx has color old_is_red (1=red, 0=blue)
    # In new state, flip_idx has color 1-old_is_red
    
    # For old state:
    if old_is_red:
        # flip_idx is red neighbor of new_v
        # Find other red neighbors of new_v (excluding flip_idx)
        other_red = [i for i in range(n_base) if i != flip_idx and current_x[i] == 1]
        # For each triple of other_red, check if {flip_idx} + triple forms K2,2,2
        if len(other_red) >= 3:
            for tri in combinations(other_red, 3):
                a, b, c = tri
                # 4 vertices: flip_idx, a, b, c
                # Need to partition into two pairs with no intra-pair edges and all cross edges
                # Pairing 1: {flip_idx, a} and {b, c}
                if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[b] >> c) & 1):
                    if ((base_adj[flip_idx] >> b) & 1) and ((base_adj[flip_idx] >> c) & 1) and \
                       ((base_adj[a] >> b) & 1) and ((base_adj[a] >> c) & 1):
                        old_violations += 1
                # Pairing 2: {flip_idx, b} and {a, c}
                if not ((base_adj[flip_idx] >> b) & 1) and not ((base_adj[a] >> c) & 1):
                    if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[flip_idx] >> c) & 1) and \
                       ((base_adj[b] >> a) & 1) and ((base_adj[b] >> c) & 1):
                        old_violations += 1
                # Pairing 3: {flip_idx, c} and {a, b}
                if not ((base_adj[flip_idx] >> c) & 1) and not ((base_adj[a] >> b) & 1):
                    if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[flip_idx] >> b) & 1) and \
                       ((base_adj[c] >> a) & 1) and ((base_adj[c] >> b) & 1):
                        old_violations += 1
    else:
        # flip_idx is blue neighbor of new_v
        other_blue = [i for i in range(n_base) if i != flip_idx and current_x[i] == 0]
        if len(other_blue) >= 3:
            for tri in combinations(other_blue, 3):
                a, b, c = tri
                # In blue graph, "no edge" means edge in base_adj, "edge" means no edge in base_adj
                # Pairing 1: {flip_idx, a} and {b, c}
                if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[b] >> c) & 1):
                    if not ((base_adj[flip_idx] >> b) & 1) and not ((base_adj[flip_idx] >> c) & 1) and \
                       not ((base_adj[a] >> b) & 1) and not ((base_adj[a] >> c) & 1):
                        old_violations += 1
                # Pairing 2: {flip_idx, b} and {a, c}
                if ((base_adj[flip_idx] >> b) & 1) and ((base_adj[a] >> c) & 1):
                    if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[flip_idx] >> c) & 1) and \
                       not ((base_adj[b] >> a) & 1) and not ((base_adj[b] >> c) & 1):
                        old_violations += 1
                # Pairing 3: {flip_idx, c} and {a, b}
                if ((base_adj[flip_idx] >> c) & 1) and ((base_adj[a] >> b) & 1):
                    if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[flip_idx] >> b) & 1) and \
                       not ((base_adj[c] >> a) & 1) and not ((base_adj[c] >> b) & 1):
                        old_violations += 1
    
    # For new state:
    new_is_red = 1 - old_is_red
    if new_is_red:
        other_red = [i for i in range(n_base) if i != flip_idx and current_x[i] == 1]
        if len(other_red) >= 3:
            for tri in combinations(other_red, 3):
                a, b, c = tri
                if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[b] >> c) & 1):
                    if ((base_adj[flip_idx] >> b) & 1) and ((base_adj[flip_idx] >> c) & 1) and \
                       ((base_adj[a] >> b) & 1) and ((base_adj[a] >> c) & 1):
                        new_violations += 1
                if not ((base_adj[flip_idx] >> b) & 1) and not ((base_adj[a] >> c) & 1):
                    if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[flip_idx] >> c) & 1) and \
                       ((base_adj[b] >> a) & 1) and ((base_adj[b] >> c) & 1):
                        new_violations += 1
                if not ((base_adj[flip_idx] >> c) & 1) and not ((base_adj[a] >> b) & 1):
                    if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[flip_idx] >> b) & 1) and \
                       ((base_adj[c] >> a) & 1) and ((base_adj[c] >> b) & 1):
                        new_violations += 1
    else:
        other_blue = [i for i in range(n_base) if i != flip_idx and current_x[i] == 0]
        if len(other_blue) >= 3:
            for tri in combinations(other_blue, 3):
                a, b, c = tri
                if ((base_adj[flip_idx] >> a) & 1) and ((base_adj[b] >> c) & 1):
                    if not ((base_adj[flip_idx] >> b) & 1) and not ((base_adj[flip_idx] >> c) & 1) and \
                       not ((base_adj[a] >> b) & 1) and not ((base_adj[a] >> c) & 1):
                        new_violations += 1
                if ((base_adj[flip_idx] >> b) & 1) and ((base_adj[a] >> c) & 1):
                    if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[flip_idx] >> c) & 1) and \
                       not ((base_adj[b] >> a) & 1) and not ((base_adj[b] >> c) & 1):
                        new_violations += 1
                if ((base_adj[flip_idx] >> c) & 1) and ((base_adj[a] >> b) & 1):
                    if not ((base_adj[flip_idx] >> a) & 1) and not ((base_adj[flip_idx] >> b) & 1) and \
                       not ((base_adj[c] >> a) & 1) and not ((base_adj[c] >> b) & 1):
                        new_violations += 1
    
    return new_violations - old_violations

def _try_extend_to_30(base_adj, deadline):
    """
    Try to extend the 29-vertex base to 30 vertices using local search minimizing violations.
    Uses incremental violation updates for speed.
    """
    n_base = 29
    n_target = 30
    random.seed(42)
    
    best_x = None
    best_violations = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        x = [random.randint(0, 1) for _ in range(n_base)]
        
        # Build adj for this x
        adj = [0] * n_target
        for i in range(n_base):
            adj[i] = base_adj[i]
        for i in range(n_base):
            if x[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
                
        v = _count_violations_29(adj, n_target, base_adj)
        
        if v == 0:
            return x[:]
            
        if v < best_violations:
            best_violations = v
            best_x = x[:]
            
        # Local search with simulated annealing
        current_x = x[:]
        current_v = v
        temperature = 1.0
        cooling_rate = 0.995
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            i = random.randint(0, n_base - 1)
            
            # Compute delta violations incrementally
            delta_v = _compute_violation_delta(base_adj, n_base, i, current_x)
            new_v = current_v + delta_v
            
            if new_v < current_v:
                current_x[i] = 1 - current_x[i]
                current_v = new_v
                if current_v == 0:
                    return current_x[:]
                if current_v < best_violations:
                    best_violations = current_v
                    best_x = current_x[:]
            else:
                # Simulated annealing: accept with probability exp(-delta/T)
                if delta_v > 0 and random.random() < pow(2.718281828, -delta_v / temperature):
                    current_x[i] = 1 - current_x[i]
                    current_v = new_v
                    if current_v < best_violations:
                        best_violations = current_v
                        best_x = current_x[:]
            
            temperature *= cooling_rate

    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Build and Validate n=29 base
    base_adj = _build_paley29()
    
    # 2. Try to extend to n=30
    search_deadline = start_time + 115
    
    result_edges = _try_extend_to_30(base_adj, search_deadline)
    
    if result_edges is not None:
        # Build the 30-vertex graph
        n = 30
        adj = [0] * n
        for i in range(29):
            adj[i] = base_adj[i]
        for i in range(29):
            if result_edges[i] == 1:
                adj[29] |= (1 << i)
                adj[i] |= (1 << 29)
        
        # Final verification before returning
        if _is_valid_coloring(adj, n):
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(1 if (adj[i] >> j) & 1 else 0)
            return {'n': n, 'colors': colors}
            
    # 3. Fallback: Return n=29
    n = 29
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(1 if (base_adj[i] >> j) & 1 else 0)
            
    return {'n': n, 'colors': colors}
