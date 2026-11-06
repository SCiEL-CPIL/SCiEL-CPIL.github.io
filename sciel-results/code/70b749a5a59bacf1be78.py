import time
import random
import itertools

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _max_independent_set_size_bitmask(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound with better pruning.
    """
    # Precompute degree for ordering
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree descending for better branching
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map old vertex index to new position in order
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    # Rebuild adj in new order
    new_adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (adj[i] >> j) & 1:
                new_adj[i] |= (1 << pos_in_order[j])
    
    # Now use branch and bound on reordered graph
    def max_is(vertices_mask, remaining, current_size):
        if remaining <= 0:
            return current_size
        if not vertices_mask:
            return current_size
        
        # Upper bound
        count = bin(vertices_mask).count('1')
        if count + current_size < target:
            return current_size
        if count < remaining:
            return current_size + count
        
        # Pick vertex with highest degree in remaining subgraph
        # Use first set bit for simplicity, but try a few heuristics
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        new_mask = vertices_mask & ~(1 << v) & ~new_adj[v]
        include_val = max_is(new_mask, remaining - 1, current_size + 1)
        
        if include_val >= target:
            return include_val
            
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining, current_size)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target, 0)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    # Using standard dot product
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _find_independent_set(adj, n, size):
    """Find an independent set of given size in the graph."""
    # Greedy approach with backtracking
    def backtrack(available_mask, chosen, needed):
        if needed == 0:
            return chosen[:]
        if not available_mask:
            return None
        if bin(available_mask).count('1') < needed:
            return None
        
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Try including v
        new_mask = available_mask & ~(1 << v) & ~adj[v]
        result = backtrack(new_mask, chosen + [v], needed - 1)
        if result is not None:
            return result
        
        # Try excluding v
        return backtrack(available_mask ^ (1 << v), chosen, needed)
    
    all_vertices = (1 << n) - 1
    return backtrack(all_vertices, [], size)

def _try_add_vertex(adj, n, edges, deadline):
    """
    Try to add a new vertex to the graph while maintaining C4-freeness.
    Returns (success, new_adj, new_edges) or (False, None, None).
    """
    if time.time() > deadline:
        return False, None, None
    
    new_v = n
    n_new = n + 1
    new_adj = adj[:]
    new_adj.append(0)
    new_edges = edges[:]
    
    # Find a set of vertices to connect to, maintaining C4-freeness
    # We need to choose neighbors such that no two neighbors have a common neighbor
    # that is also a neighbor of the new vertex (which would create a C4)
    
    # Strategy: greedily add edges while checking C4-freeness
    # Sort candidates by degree (prefer lower degree to avoid C4s)
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    candidates = sorted(range(n), key=lambda x: degrees[x])
    
    chosen = []
    available = set(range(n))
    
    for v in candidates:
        if time.time() > deadline:
            return False, None, None
        if v not in available:
            continue
        
        # Check if we can add edge (new_v, v)
        # This would create a C4 if there exists another neighbor u of new_v
        # such that u and v have a common neighbor
        # Actually, adding (new_v, v) creates a C4 if there exist u, w such that
        # new_v is connected to u, v is connected to w, and u is connected to w
        # But more precisely: adding (new_v, v) creates a C4 if there exists
        # u in chosen such that u and v have a common neighbor
        
        can_add = True
        for u in chosen:
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                can_add = False
                break
        
        if can_add:
            chosen.append(v)
            new_adj[new_v] |= (1 << v)
            new_adj[v] |= (1 << new_v)
            new_edges.append([new_v, v])
    
    # Verify C4-freeness of the new graph
    if _has_c4_bitmask(n_new, new_adj):
        return False, None, None
    
    return True, new_adj, new_edges

def _construct_large_graph(deadline):
    """
    Try to construct a graph with n > 31 by iteratively adding vertices.
    """
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    best_n = n_base
    best_adj = adj_base
    best_edges = edges_base
    
    # Try to add vertices one by one
    current_n = n_base
    current_adj = adj_base
    current_edges = edges_base
    
    for target_n in range(n_base + 1, 44):
        if time.time() > deadline:
            break
        
        # Try multiple random orders for adding edges
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + target_n)
            
            trial_adj = current_adj[:]
            trial_edges = current_edges[:]
            
            # Try to add vertex with randomized edge selection
            new_v = current_n
            new_adj = trial_adj[:]
            new_adj.append(0)
            new_edges = trial_edges[:]
            
            # Randomly order candidates
            candidates = list(range(current_n))
            random.shuffle(candidates)
            
            chosen = []
            for v in candidates:
                if time.time() > deadline:
                    break
                
                # Check if we can add edge (new_v, v)
                can_add = True
                for u in chosen:
                    common = trial_adj[u] & trial_adj[v]
                    if common & (common - 1):  # at least two bits set
                        can_add = False
                        break
                
                if can_add:
                    chosen.append(v)
                    new_adj[new_v] |= (1 << v)
                    new_adj[v] |= (1 << new_v)
                    new_edges.append([new_v, v])
            
            # Verify C4-freeness
            if not _has_c4_bitmask(current_n + 1, new_adj):
                # Check independence number
                if _max_independent_set_size_bitmask(current_n + 1, new_adj, 11) < 11:
                    best_n = current_n + 1
                    best_adj = new_adj
                    best_edges = new_edges
                    current_n = best_n
                    current_adj = new_adj
                    current_edges = new_edges
                    found = True
                    break
            else:
                # Even without checking independence, we need C4-free
                pass
        
        if not found:
            # Try a few more attempts with different strategies
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                random.seed(attempt * 104729 + target_n * 3)
                
                trial_adj = current_adj[:]
                trial_edges = current_edges[:]
                
                new_v = current_n
                new_adj = trial_adj[:]
                new_adj.append(0)
                new_edges = trial_edges[:]
                
                # Try to add edges in a more constrained way
                # Only connect to vertices that don't create C4s
                # And limit the number of edges to maintain independence constraint
                
                candidates = list(range(current_n))
                random.shuffle(candidates)
                
                chosen = []
                max_edges = random.randint(3, 8)
                
                for v in candidates:
                    if len(chosen) >= max_edges:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Check if we can add edge (new_v, v)
                    can_add = True
                    for u in chosen:
                        common = trial_adj[u] & trial_adj[v]
                        if common & (common - 1):  # at least two bits set
                            can_add = False
                            break
                    
                    if can_add:
                        chosen.append(v)
                        new_adj[new_v] |= (1 << v)
                        new_adj[v] |= (1 << new_v)
                        new_edges.append([new_v, v])
                
                # Verify C4-freeness
                if not _has_c4_bitmask(current_n + 1, new_adj):
                    # Check independence number
                    if _max_independent_set_size_bitmask(current_n + 1, new_adj, 11) < 11:
                        best_n = current_n + 1
                        best_adj = new_adj
                        best_edges = new_edges
                        current_n = best_n
                        current_adj = new_adj
                        current_edges = new_edges
                        found = True
                        break
            
            if not found:
                break
    
    return {'n': best_n, 'edges': best_edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with the polarity graph of PG(2,5)
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Try to extend the graph
    if time.time() < deadline:
        extended = _construct_large_graph(deadline)
        if extended['n'] > best_result['n']:
            best_result = extended
    
    # Verify the best result
    if _verify_graph(best_result['n'], best_result['edges']):
        return best_result
    
    # Fallback: return the base graph
    return {'n': n_base, 'edges': edges_base}

# EVOLVE-BLOCK-END
