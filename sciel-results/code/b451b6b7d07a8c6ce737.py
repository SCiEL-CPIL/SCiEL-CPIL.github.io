import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with various strategies
        # We use a mix of greedy heuristics and local search refinements.
        
        # Strategy 1: Greedy with randomized ordering and degree balancing
        # Strategy 2: Start from a sparse seed and grow
        # Strategy 3: Local search (edge swaps) to fix min-degree violations
        
        # We will try a batch of attempts.
        # Each attempt consists of:
        # 1. Initial construction (greedy)
        # 2. Local search refinement (swap edges to increase min degree or fix violations)
        
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = _try_construct_hybrid(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If we couldn't find a solution for this n, continue to n-1
        # But we must ensure we don't exceed deadline.
        
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4 in the graph defined by adj.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x,y) such that x is in N(u)\{v} and y is in N(v)\{u}.
    """
    # Vectorized approach
    # We need to check if there exists x, y such that:
    # adj[u,x] and adj[x,y] and adj[y,v] and x!=v and y!=u and x!=y
    # Note: since adj is symmetric and we are adding (u,v), adj[u,v] is currently False.
    # So x != v is naturally handled if we exclude v from N(u) candidates?
    # No, x is a neighbor of u. If x==v, then v is a neighbor of u, which implies adj[u,v] is True.
    # But we are checking if we *can* add it, so adj[u,v] is False. Thus x != v is always true for x in N(u).
    # Similarly y != u is always true.
    # However, x and y must be distinct.
    
    # Efficient check:
    # Let N_u = set of neighbors of u
    # Let N_v = set of neighbors of v
    # We want to know if there is an edge between N_u and N_v.
    # Since adj is a boolean matrix, we can do:
    # sub_adj = adj[np.ix_(N_u, N_v)]
    # return np.any(sub_adj)
    # But we must ensure x != y.
    # If there is an edge (x,y) with x in N_u and y in N_v, and x != y, then C4 exists.
    # If x == y, then u-x-v is a path of length 2. Adding u-v creates a triangle, not a C4.
    # So we need to check if there is an edge (x,y) with x != y.
    
    idx_u = np.nonzero(adj[u])[0]
    if len(idx_u) == 0:
        return False
    idx_v = np.nonzero(adj[v])[0]
    if len(idx_v) == 0:
        return False
        
    # Check submatrix
    # We need to be careful: if x==y, adj[x,x] is 0 (no self-loops), so it won't contribute to np.any.
    # So np.any(adj[np.ix_(idx_u, idx_v)]) is sufficient.
    return np.any(adj[np.ix_(idx_u, idx_v)])

def _try_construct_hybrid(n, required_min_deg, deadline, seed):
    """
    Hybrid construction:
    1. Greedy phase to build a C4-free graph.
    2. Local search phase to try to increase min degree.
    """
    rng = np.random.RandomState(seed)
    
    adj = np.zeros((n, n), dtype=bool)
    degrees = np.zeros(n, dtype=int)
    
    # Phase 1: Greedy Construction
    # We try to add edges in a way that balances degrees.
    # We iterate until no more edges can be added or we run out of time.
    
    # To introduce diversity, we can shuffle the order in which we consider vertices.
    # Or we can use a priority queue.
    
    # Simple approach:
    # While min_deg < required_min_deg:
    #   Find vertex u with min degree.
    #   Try to add an edge (u, v) where v is a non-neighbor.
    #   Choose v to maximize the minimum degree of the resulting graph (or just random/low-degree).
    
    # Let's implement a greedy that picks the most constrained vertex.
    
    # To avoid getting stuck, we allow "random" choices among valid candidates.
    
    max_greedy_steps = n * (n - 1) // 2
    for _ in range(max_greedy_steps):
        if time.time() > deadline:
            return None
            
        if degrees.min() >= required_min_deg:
            break
            
        # Find the vertex with the minimum degree
        # If there are multiple, pick one randomly to add diversity
        min_deg_val = degrees.min()
        min_vertices = np.where(degrees == min_deg_val)[0]
        u = min_vertices[rng.randint(0, len(min_vertices))]
        
        # Find candidates for u: non-neighbors
        candidates = np.where(~adj[u])[0]
        candidates = candidates[candidates != u]
        
        if len(candidates) == 0:
            # u is connected to everyone, but degree < required?
            # This means required_min_deg > n-1, which is impossible.
            # Or we are stuck.
            break
            
        # Sort candidates by degree (ascending) to balance the graph
        cand_degrees = degrees[candidates]
        sorted_indices = np.argsort(cand_degrees)
        
        # Try candidates in sorted order.
        # To add randomness, we can shuffle the top K candidates.
        # For now, let's try the best one (lowest degree).
        # If that fails, try the next best.
        
        added = False
        # We can also add a bit of randomness: pick from the top 3 candidates randomly?
        # Let's stick to deterministic best-first for the greedy phase, as the seed changes the initial state/ordering implicitly via ties?
        # Actually, if degrees are equal, np.argsort is stable but deterministic.
        # Let's add randomness by shuffling candidates with equal degree?
        
        # Simpler: just iterate.
        for idx in sorted_indices:
            v = candidates[idx]
            
            if _check_c4_add(adj, u, v):
                continue
                
            # Safe to add
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1
            added = True
            break
            
        if not added:
            # Could not add any edge to u without creating C4.
            # Try a random other vertex?
            # If we can't add to the min-degree vertex, we might be stuck.
            # Let's try a few random vertices to see if any edge can be added.
            success = False
            for _ in range(5):
                u2 = rng.randint(0, n)
                candidates2 = np.where(~adj[u2])[0]
                candidates2 = candidates2[candidates2 != u2]
                if len(candidates2) == 0:
                    continue
                cand_degrees2 = degrees[candidates2]
                sorted_indices2 = np.argsort(cand_degrees2)
                
                for idx2 in sorted_indices2:
                    v2 = candidates2[idx2]
                    if not _check_c4_add(adj, u2, v2):
                        adj[u2, v2] = True
                        adj[v2, u2] = True
                        degrees[u2] += 1
                        degrees[v2] += 1
                        success = True
                        break
                if success:
                    break
                    
            if not success:
                # Stuck in greedy phase.
                # Proceed to local search?
                # Local search might help if we are close.
                pass

    # Phase 2: Local Search
    # Goal: Increase min degree if it's below required.
    # Move: Swap edges.
    # Remove edge (a, b) and add edge (c, d) such that:
    # 1. (a, b) is present.
    # 2. (c, d) is absent.
    # 3. Adding (c, d) does not create C4 (considering (a, b) is removed).
    # 4. The min degree does not decrease. Ideally increases.
    
    # This is complex. A simpler local search:
    # If min_deg < required, find a vertex u with low degree.
    # Find a vertex v with high degree.
    # Try to "move" an edge from v to u.
    # i.e., find a neighbor w of v such that (u, w) is not an edge and adding (u, w) is C4-safe,
    # and removing (v, w) is safe (it always is, as we are removing an edge).
    # But we must ensure that removing (v, w) doesn't drop v's degree below required.
    # And adding (u, w) doesn't create C4.
    
    # Let's implement this "edge shifting" local search.
    
    if degrees.min() < required_min_deg:
        # Run local search for a limited number of steps
        local_search_steps = 500
        for _ in range(local_search_steps):
            if time.time() > deadline:
                break
                
            if degrees.min() >= required_min_deg:
                break
                
            # Find u with min degree
            min_deg_val = degrees.min()
            min_vertices = np.where(degrees == min_deg_val)[0]
            u = min_vertices[rng.randint(0, len(min_vertices))]
            
            # Find v with max degree (potential donor)
            max_deg_val = degrees.max()
            max_vertices = np.where(degrees == max_deg_val)[0]
            v = max_vertices[rng.randint(0, len(max_vertices))]
            
            if u == v:
                continue
                
            # We want to move an edge from v to u.
            # Find a neighbor w of v such that:
            # 1. (u, w) is not an edge.
            # 2. Adding (u, w) is C4-safe.
            # 3. degrees[v] > required_min_deg (so we can afford to lose one).
            
            if degrees[v] <= required_min_deg:
                continue
                
            # Candidates for w: neighbors of v that are not neighbors of u
            # N(v) = np.where(adj[v])[0]
            # N(u) = np.where(adj[u])[0]
            # w must be in N(v) \ (N(u) U {u})
            
            # Efficiently:
            # mask_w = adj[v].copy()
            # mask_w[u] = False
            # mask_w[adj[u]] = False
            
            # But we also need to check C4.
            
            # Let's iterate through neighbors of v
            neighbors_v = np.where(adj[v])[0]
            # Shuffle to add randomness
            rng.shuffle(neighbors_v)
            
            moved = False
            for w in neighbors_v:
                if w == u:
                    continue
                if adj[u, w]:
                    continue
                
                # Check if adding (u, w) creates C4
                # Note: (v, w) is currently present.
                # When we check C4 for (u, w), we assume (v, w) is still present?
                # No, we are planning to remove (v, w).
                # So the check should be: does (u, w) create C4 in the graph G \ {(v, w)}?
                # This is tricky.
                # Approximation: Check if (u, w) creates C4 in G.
                # If it does, maybe removing (v, w) breaks the C4?
                # A C4 involving (u, w) would be u-w-x-y-u.
                # If x=v and y is a neighbor of w and u?
                # If the C4 is u-w-v-... no, v is not connected to u (we assumed adj[u,v] is False? No, u and v are distinct vertices, they might be connected).
                # Wait, u and v are just two vertices.
                # If u and v are connected, and we remove (v, w), and add (u, w).
                # Let's just do the check on the current graph G.
                # If _check_c4_add(adj, u, w) is False, it's definitely safe.
                # If True, it might be safe after removing (v, w), but let's be conservative.
                
                if _check_c4_add(adj, u, w):
                    continue
                
                # Check if removing (v, w) and adding (u, w) maintains C4-free.
                # We already checked (u, w) is safe in G.
                # Is it possible that removing (v, w) creates a C4? No, removing edges never creates C4s.
                # So if (u, w) is safe in G, it is safe in G \ {(v, w)}.
                
                # Perform the move
                adj[v, w] = False
                adj[w, v] = False
                adj[u, w] = True
                adj[w, u] = True
                
                degrees[v] -= 1
                degrees[w] -= 1
                degrees[u] += 1
                degrees[w] += 1
                
                moved = True
                break
                
            if not moved:
                # Try a different u or v?
                # For now, just break or continue to next step.
                # If we can't move, we are stuck.
                # Let's try a few more random pairs.
                success = False
                for _ in range(5):
                    u2 = rng.randint(0, n)
                    v2 = rng.randint(0, n)
                    if u2 == v2:
                        continue
                    if degrees[v2] <= required_min_deg:
                        continue
                    neighbors_v2 = np.where(adj[v2])[0]
                    rng.shuffle(neighbors_v2)
                    for w in neighbors_v2:
                        if w == u2:
                            continue
                        if adj[u2, w]:
                            continue
                        if _check_c4_add(adj, u2, w):
                            continue
                        adj[v2, w] = False
                        adj[w, v2] = False
                        adj[u2, w] = True
                        adj[w, u2] = True
                        degrees[v2] -= 1
                        degrees[w] -= 1
                        degrees[u2] += 1
                        degrees[w] += 1
                        success = True
                        break
                    if success:
                        break
                if not success:
                    break

    # Check final result
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
