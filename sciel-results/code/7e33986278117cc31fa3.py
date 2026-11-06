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

def _precompute_violation_data(base_adj, n_base):
    """
    Precompute data needed to quickly evaluate violations when adding vertex 29.
    For each subset of 4 neighbors, precompute whether they form a K2,2,2 - v structure.
    """
    # Precompute for each 4-tuple (i,j,k,l) with i<j<k<l < n_base:
    # red_violation[i,j,k,l] = True if in red graph, {i,j,k,l} with new_v as part A
    #   forms K2,2,2 (new_v connected to all 4, and the 4 form two independent pairs with complete bipartite between them)
    # blue_violation[i,j,k,l] = True if in blue graph, similar structure
    
    n = n_base
    red_viol = {}
    blue_viol = {}
    
    for quad in combinations(range(n), 4):
        u, v, w, x = quad
        # Check red violations: new_v in Part A, neighbors in Parts B,C
        # Pairing 1: {u,v} and {w,x}
        if not ((base_adj[u] >> v) & 1) and not ((base_adj[w] >> x) & 1):
            if ((base_adj[u] >> w) & 1) and ((base_adj[u] >> x) & 1) and \
               ((base_adj[v] >> w) & 1) and ((base_adj[v] >> x) & 1):
                red_viol[quad] = True
        
        # Pairing 2: {u,w} and {v,x}
        if not ((base_adj[u] >> w) & 1) and not ((base_adj[v] >> x) & 1):
            if ((base_adj[u] >> v) & 1) and ((base_adj[u] >> x) & 1) and \
               ((base_adj[w] >> v) & 1) and ((base_adj[w] >> x) & 1):
                red_viol[quad] = True
        
        # Pairing 3: {u,x} and {v,w}
        if not ((base_adj[u] >> x) & 1) and not ((base_adj[v] >> w) & 1):
            if ((base_adj[u] >> v) & 1) and ((base_adj[u] >> w) & 1) and \
               ((base_adj[x] >> v) & 1) and ((base_adj[x] >> w) & 1):
                red_viol[quad] = True
        
        # Check blue violations: new_v in Part A, neighbors in Parts B,C
        # In blue graph, edges are where base_adj has 0
        # Pairing 1: {u,v} and {w,x}
        if ((base_adj[u] >> v) & 1) and ((base_adj[w] >> x) & 1):
            if not ((base_adj[u] >> w) & 1) and not ((base_adj[u] >> x) & 1) and \
               not ((base_adj[v] >> w) & 1) and not ((base_adj[v] >> x) & 1):
                blue_viol[quad] = True
        
        # Pairing 2: {u,w} and {v,x}
        if ((base_adj[u] >> w) & 1) and ((base_adj[v] >> x) & 1):
            if not ((base_adj[u] >> v) & 1) and not ((base_adj[u] >> x) & 1) and \
               not ((base_adj[w] >> v) & 1) and not ((base_adj[w] >> x) & 1):
                blue_viol[quad] = True
        
        # Pairing 3: {u,x} and {v,w}
        if ((base_adj[u] >> x) & 1) and ((base_adj[v] >> w) & 1):
            if not ((base_adj[u] >> v) & 1) and not ((base_adj[u] >> w) & 1) and \
               not ((base_adj[x] >> v) & 1) and not ((base_adj[x] >> w) & 1):
                blue_viol[quad] = True
    
    return red_viol, blue_viol

def _count_violations_fast(x, base_adj, red_viol, blue_viol, n_base):
    """
    Count violations given assignment x (list of 0/1 for vertices 0..n_base-1).
    x[i]=1 means vertex i is connected to new vertex 29 in red.
    x[i]=0 means vertex i is connected to new vertex 29 in blue.
    """
    violations = 0
    
    # Red neighbors of new vertex
    red_neigh = [i for i in range(n_base) if x[i] == 1]
    # Blue neighbors of new vertex
    blue_neigh = [i for i in range(n_base) if x[i] == 0]
    
    # Count red violations
    for quad in combinations(red_neigh, 4):
        if quad in red_viol:
            violations += 1
    
    # Count blue violations
    for quad in combinations(blue_neigh, 4):
        if quad in blue_viol:
            violations += 1
    
    return violations

def _try_extend_to_30(base_adj, deadline):
    """
    Try to extend the 29-vertex base to 30 vertices using local search minimizing violations.
    """
    n_base = 29
    n_target = 30
    random.seed(42)
    
    # Precompute violation data
    red_viol, blue_viol = _precompute_violation_data(base_adj, n_base)
    
    best_x = None
    best_violations = float('inf')
    
    # Try several random restarts with local search
    for restart in range(30):
        if time.time() > deadline:
            break
            
        # Random initial assignment
        x = [random.randint(0, 1) for _ in range(n_base)]
        
        v = _count_violations_fast(x, base_adj, red_viol, blue_viol, n_base)
        
        if v == 0:
            return x[:]
            
        if v < best_violations:
            best_violations = v
            best_x = x[:]
            
        # Local search: try all single flips, keep the best improvement
        current_x = x[:]
        current_v = v
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip = -1
            best_new_v = current_v
            for i in range(n_base):
                new_x = current_x[:]
                new_x[i] = 1 - new_x[i]
                new_v = _count_violations_fast(new_x, base_adj, red_viol, blue_viol, n_base)
                if new_v < best_new_v:
                    best_new_v = new_v
                    best_flip = i
            if best_flip >= 0:
                current_x[best_flip] = 1 - current_x[best_flip]
                current_v = best_new_v
                improved = True
                if current_v == 0:
                    return current_x[:]
                if current_v < best_violations:
                    best_violations = current_v
                    best_x = current_x[:]
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Build and Validate n=29 base
    base_adj = _build_paley29()
    
    # 2. Try to extend to n=30
    search_deadline = start_time + 110
    
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
