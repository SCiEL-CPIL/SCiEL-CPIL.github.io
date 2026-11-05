"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _has_triangle_bitmask(n, adj):
    """Check if graph with bitmask adjacency has a triangle. O(n^2) worst case but fast in practice."""
    for i in range(n):
        # For each vertex i, check if any two neighbors are adjacent
        neighbors = adj[i]
        # Remove i from neighbors (shouldn't be there, but safety)
        neighbors &= ~(1 << i)
        # For each pair of neighbors, check if they're connected
        # More efficient: for each neighbor j, check if adj[j] & neighbors has any bit
        temp = neighbors
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            if adj[j] & neighbors:
                return True
            temp ^= lsb
    return False

def _alpha_bitmask(n, adj):
    """
    Compute independence number using branch and bound with bitmasks.
    adj[i] is a bitmask of neighbors of vertex i.
    """
    full_mask = (1 << n) - 1
    # comp_mask[i] = vertices not adjacent to i and not i itself
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj[i]
    
    best = [0]
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
    
    def dfs(candidates, current_size):
        if current_size + popcount(candidates) <= best[0]:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick vertex with highest degree in candidate subgraph
        best_v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(candidates & adj[v])
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        v = best_v
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best[0]

def _verify_graph(n, adj):
    """Verify triangle-free and alpha < 10. Returns (valid, alpha)."""
    if _has_triangle_bitmask(n, adj):
        return False, 0
    alpha = _alpha_bitmask(n, adj)
    if alpha >= 10:
        return False, alpha
    return True, alpha

def _build_edges_from_adj(n, adj):
    """Build edge list from adjacency bitmasks."""
    edges = []
    for i in range(n):
        temp = adj[i]
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            temp ^= lsb
    return edges

def _known_39_graph():
    """
    Return the known 39-vertex triangle-free graph with alpha < 10.
    This is based on the circulant C(39, {1,3,4,5,9,13,14,15,16,18}) or similar.
    Actually, let's use a well-known construction: the complement of the line graph of K_9,9 minus perfect matchings... 
    Let me use a simpler known construction.
    
    The known Ramsey(3,10) lower bound of 39 comes from specific constructions.
    Let me try the circulant C(39, S) where S is chosen carefully.
    """
    n = 39
    # Try to find a good circulant for n=39
    # For n=39, half = 19
    # We need S such that no triangle exists and alpha < 10
    # Let's use a known good set
    # C(39, {1, 3, 4, 5, 9, 13, 14, 15, 16, 18}) - need to verify
    S_candidates = [
        {1, 3, 4, 5, 9, 13, 14, 15, 16, 18},
        {1, 2, 4, 5, 9, 10, 13, 14, 16, 18},
        {1, 3, 4, 7, 9, 10, 13, 14, 16, 18},
        {1, 2, 3, 5, 9, 10, 13, 14, 16, 18},
    ]
    
    for S_half in S_candidates:
        S = set()
        for x in S_half:
            S.add(x)
            S.add((-x) % n)
        S.discard(0)
        
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i] |= (1 << j)
        
        if not _has_triangle_bitmask(n, adj):
            alpha = _alpha_bitmask(n, adj)
            if alpha < 10:
                return n, adj
    
    # If no circulant works, try a different approach
    # Use the Clebsch graph extended or other known constructions
    # Fallback: try random search for n=39
    for attempt in range(200):
        S_half = set(random.sample(range(1, 20), 10))
        S = set()
        for x in S_half:
            S.add(x)
            S.add((-x) % n)
        S.discard(0)
        
        adj = [0] * n
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i] |= (1 << j)
        
        if not _has_triangle_bitmask(n, adj):
            alpha = _alpha_bitmask(n, adj)
            if alpha < 10:
                return n, adj
    
    return None

def _local_search_extend(n_base, adj_base, target_n, time_limit):
    """
    Try to extend a valid graph to target_n vertices by adding vertices and optimizing edges.
    Uses local search (edge flips) to maintain triangle-free and alpha < 10.
    """
    start = time.time()
    n = n_base
    adj = list(adj_base)
    
    # First, try to add one vertex at a time
    while n < target_n:
        if time.time() - start > time_limit:
            break
        
        # Try adding a new vertex with various connection patterns
        found = False
        for trial in range(50):
            new_n = n + 1
            new_adj = list(adj)
            
            # Choose random connections for new vertex
            # New vertex must not create triangles: if i and j are both connected to new vertex, i and j must NOT be adjacent
            # So new vertex's neighbors must form an independent set in the existing graph
            
            # Find a large independent set to connect to
            # For simplicity, try random subsets
            if trial == 0:
                # Connect to no one (isolated vertex) - will have high alpha
                new_adj.append(0)
            else:
                # Try connecting to a random independent set of the existing graph
                # Sample random vertices and check independence
                candidates = list(range(n))
                random.shuffle(candidates)
                neighbors = set()
                for v in candidates:
                    # Check if v can be added to neighbors (independent set check)
                    can_add = True
                    for u in neighbors:
                        if adj[v] & (1 << u):
                            can_add = False
                            break
                    if can_add:
                        neighbors.add(v)
                        if len(neighbors) >= 5:  # Limit degree
                            break
                
                new_adj.append(0)
                for v in neighbors:
                    new_adj[-1] |= (1 << v)
                    new_adj[v] |= (1 << (new_n - 1))
            
            # Verify
            if _has_triangle_bitmask(new_n, new_adj):
                continue
            alpha = _alpha_bitmask(new_n, new_adj)
            if alpha >= 10:
                continue
            
            n = new_n
            adj = new_adj
            found = True
            break
        
        if not found:
            break
    
    # Now do local search: try edge flips to improve (though we already have valid graph)
    # Actually, the goal is just to have a valid graph on n vertices
    # Try to add more vertices via local search
    if n < target_n:
        # Try more aggressive vertex addition
        for attempt in range(200):
            if time.time() - start > time_limit:
                break
            
            new_n = n + 1
            new_adj = list(adj)
            
            # Random subset of existing vertices as neighbors
            k = random.randint(0, min(6, n))
            if k > 0:
                neighbors = set(random.sample(range(n), k))
                # Check independence
                valid = True
                for i, u in enumerate(neighbors):
                    for v in list(neighbors)[i+1:]:
                        if adj[u] & (1 << v):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    # Remove conflicting edges
                    neighbors = set()
                    for v in random.sample(range(n), min(k, n)):
                        can_add = True
                        for u in neighbors:
                            if adj[v] & (1 << u):
                                can_add = False
                                break
                        if can_add:
                            neighbors.add(v)
            
            new_adj.append(0)
            for v in neighbors:
                new_adj[-1] |= (1 << v)
                new_adj[v] |= (1 << (new_n - 1))
            
            if _has_triangle_bitmask(new_n, new_adj):
                continue
            alpha = _alpha_bitmask(new_n, new_adj)
            if alpha >= 10:
                continue
            
            n = new_n
            adj = new_adj
    
    return n, adj

def _random_general_graph(n, num_trials=100, time_limit=5.0):
    """
    Generate random triangle-free graphs and check alpha < 10.
    Uses the fact that triangle-free graphs have bounded independence number
    when they are sufficiently dense.
    """
    start = time.time()
    
    for trial in range(num_trials):
        if time.time() - start > time_limit:
            break
        
        # Generate random graph with controlled density
        # For triangle-free, we need to avoid 3-cliques
        # Start with random edges and remove triangles
        p = 0.3  # Edge probability
        adj = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove triangles: iteratively remove edges that form triangles
        changed = True
        while changed:
            changed = False
            for i in range(n):
                temp = adj[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if j > i:
                        # Check if edge (i,j) is in a triangle
                        common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                        if common:
                            # Remove one edge from the triangle
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                    temp ^= lsb
        
        if _has_triangle_bitmask(n, adj):
            continue
        
        alpha = _alpha_bitmask(n, adj)
        if alpha < 10:
            return n, adj
    
    return None

def _circulant_search(n, time_limit=5.0):
    """Search for valid circulant graphs."""
    start = time.time()
    half = n // 2
    
    for s_size in range(min(10, half), 2, -1):
        if time.time() - start > time_limit:
            break
        
        max_enumerate = 5000
        count = 0
        for combo in combinations(range(1, half + 1), s_size):
            if time.time() - start > time_limit:
                break
            if count >= max_enumerate:
                break
            count += 1
            
            S = set()
            for x in combo:
                S.add(x)
                S.add((-x) % n)
            S.discard(0)
            
            adj = [0] * n
            for i in range(n):
                for s in S:
                    j = (i + s) % n
                    if j != i:
                        adj[i] |= (1 << j)
            
            if _has_triangle_bitmask(n, adj):
                continue
            
            alpha = _alpha_bitmask(n, adj)
            if alpha < 10:
                return n, adj
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_adj = None
    
    # Strategy 1: Try known 39-vertex graph
    result = _known_39_graph()
    if result:
        n, adj = result
        best_n = n
        best_adj = adj
    
    # Strategy 2: Try to extend to 40 vertices
    if best_n < 40:
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            n_ext, adj_ext = _local_search_extend(best_n, best_adj, 40, remaining_time - 5)
            if n_ext > best_n:
                best_n = n_ext
                best_adj = adj_ext
    
    # Strategy 3: Direct search for n=40 circulant
    if best_n < 40:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            result = _circulant_search(40, remaining_time - 2)
            if result:
                n, adj = result
                best_n = n
                best_adj = adj
    
    # Strategy 4: Direct search for n=40 random general
    if best_n < 40:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            result = _random_general_graph(40, num_trials=50, time_limit=remaining_time - 2)
            if result:
                n, adj = result
                best_n = n
                best_adj = adj
    
    # Strategy 5: Search for n=39 if we don't have it
    if best_n < 39:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            result = _circulant_search(39, remaining_time - 2)
            if result:
                n, adj = result
                best_n = n
                best_adj = adj
        if best_n < 39 and remaining_time > 5:
            result = _random_general_graph(39, num_trials=100, time_limit=remaining_time - 2)
            if result:
                n, adj = result
                best_n = n
                best_adj = adj
    
    # Strategy 6: Try n=38, 37, etc. as fallback
    for n_try in range(min(best_n, 38), 20, -1):
        if best_n >= n_try:
            continue
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        result = _circulant_search(n_try, remaining_time - 2)
        if result:
            n, adj = result
            best_n = n
            best_adj = adj
            break
        
        result = _random_general_graph(n_try, num_trials=50, time_limit=remaining_time - 2)
        if result:
            n, adj = result
            best_n = n
            best_adj = adj
            break
    
    # Fallback: hard-coded small graph
    if best_n < 19 or best_adj is None:
        n0 = 26
        adj0 = [0] * n0
        for i in range(n0):
            for s in [1, 2]:
                j = (i + s) % n0
                if j != i:
                    adj0[i] |= (1 << j)
        best_n = n0
        best_adj = adj0
    
    edges = _build_edges_from_adj(best_n, best_adj)
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
