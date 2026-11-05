import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s safety margin
    
    best_result = {'n': 39, 'edges': []}
    
    # Try n=47 first (best case), then 46, etc.
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _fast_greedy_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
        # If fast greedy fails, try local search with swaps for this n
        if n >= 43 and time.time() < deadline:
            result = _local_search_construct(n, required_min_deg, deadline)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}

    # Fallback: return a valid small graph
    return best_result

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency matrix adj creates a C4.
    Vectorized: C4 exists iff there's x in N(u)\{v} and y in N(v)\{u} with x!=y and (x,y) in E.
    """
    # N(u) excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    # N(v) excluding u  
    mask_v = adj[v].copy()
    mask_v[u] = False
    
    # For each x in N(u)\{v}, check if N(x) intersects N(v)\{u}\{x}
    # adj[x] & mask_v gives for each x, the intersection of N(x) with N(v)\{u}
    # Since adj[x][x] = 0, x won't be in adj[x] & mask_v
    intersections = np.sum(adj & mask_v[None, :], axis=1)
    
    # Check if any x in N(u)\{v} has intersection > 0
    return np.any(intersections[mask_u] > 0)

def _extract_edges(adj):
    """Extract edge list from adjacency matrix."""
    edges = []
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def _fast_greedy_construct(n, required_min_deg, deadline):
    """
    Attempts to build a C4-free graph with min degree >= required_min_deg.
    Uses multiple random restarts with a greedy strategy prioritizing low-degree vertices.
    """
    rng = np.random.default_rng()
    
    max_restarts = 300
    
    for attempt in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj = np.zeros((n, n), dtype=bool)
        
        success = False
        while time.time() < deadline:
            degrees = adj.sum(axis=1)
            min_deg = int(np.min(degrees))
            
            if min_deg >= required_min_deg:
                success = True
                break
                
            # Find vertices with minimum degree
            min_vertices = np.where(degrees == min_deg)[0]
            if len(min_vertices) == 0:
                break
            
            # Try each min-degree vertex, pick one that can be improved
            shuffled = min_vertices.copy()
            rng.shuffle(shuffled)
            
            u = shuffled[0]
            
            # Get non-neighbors of u
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            
            if len(non_neighbors) == 0:
                break
            
            # Shuffle to try random order
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if _check_c4_add(adj, u, v):
                    continue
                # Safe to add
                adj[u, v] = True
                adj[v, u] = True
                added = True
                break  # Restart loop to re-evaluate degrees
            
            if not added:
                # Could not add any edge to u without creating C4
                # Try a different vertex or restart
                break
                
        if success:
            edges = _extract_edges(adj)
            return (n, edges)
            
    return None

def _local_search_construct(n, required_min_deg, deadline):
    """
    Fallback: Local search with edge swaps to fix stuck states.
    """
    # First try greedy to get a starting point
    result = _fast_greedy_construct(n, required_min_deg, deadline)
    if result is not None:
        return result
        
    # If greedy fails, try edge-swap local search
    # Start from a random C4-free graph and try to increase min degree
    for seed in range(30):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed)
        adj = np.zeros((n, n), dtype=bool)
        
        # Build a random C4-free graph
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if not _check_c4_add(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                
        degrees = adj.sum(axis=1)
        if np.min(degrees) >= required_min_deg:
            edges = _extract_edges(adj)
            return (n, edges)
        
        # Local search: try edge swaps to increase min degree
        # For each vertex with low degree, try to swap an edge to raise its degree
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            degrees = adj.sum(axis=1)
            min_deg = int(np.min(degrees))
            if min_deg >= required_min_deg:
                edges = _extract_edges(adj)
                return (n, edges)
            
            # Find vertex with minimum degree
            min_vertices = np.where(degrees == min_deg)[0]
            u = rng.choice(min_vertices)
            
            # Try to add an edge to u
            non_neighbors = np.where(~adj[u])[0]
            non_neighbors = non_neighbors[non_neighbors != u]
            
            if len(non_neighbors) == 0:
                continue
            
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if not _check_c4_add(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    added = True
                    break
            
            if not added:
                # Try edge swap: remove an edge from a high-degree vertex, add to u
                # Find a high-degree vertex w
                max_deg = int(np.max(degrees))
                high_vertices = np.where(degrees == max_deg)[0]
                high_vertices = high_vertices[high_vertices != u]
                
                if len(high_vertices) > 0:
                    w = rng.choice(high_vertices)
                    # Find a neighbor x of w that is not u
                    neighbors_w = np.where(adj[w])[0]
                    neighbors_w = neighbors_w[neighbors_w != u]
                    if len(neighbors_w) > 1:
                        rng.shuffle(neighbors_w)
                        x = neighbors_w[0]
                        # Try to remove edge (w, x) and add (u, x) or (u, w)
                        # First try adding (u, x)
                        if not adj[u, x] and not _check_c4_add(adj, u, x):
                            # Check if removing (w, x) keeps C4-free
                            adj[w, x] = False
                            adj[x, w] = False
                            # Verify no C4 was created by removal (removal can't create C4)
                            # But we need to ensure the graph is still valid
                            adj[u, x] = True
                            adj[x, u] = True
                            added = True
                        
                        if not added:
                            # Try adding (u, w)
                            if not adj[u, w] and not _check_c4_add(adj, u, w):
                                adj[w, x] = False
                                adj[x, w] = False
                                adj[u, w] = True
                                adj[w, u] = True
                                added = True
                        
                        if not added:
                            # Undo
                            adj[w, x] = True
                            adj[x, w] = True
    
    return None

# EVOLVE-BLOCK-END
