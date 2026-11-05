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
    
    # We know n=47 is possible (R(C4, K1,39) = 47).
    # Try 47 first.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        # Try to construct for this n
        result = _robust_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
    # Fallback: return a valid small graph
    # n=39, min_deg=0 is always valid (empty graph)
    return {'n': 39, 'edges': []}

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency matrix adj creates a C4.
    Returns True if a C4 is created, False otherwise.
    """
    # N(u) excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    # N(v) excluding u  
    mask_v = adj[v].copy()
    mask_v[u] = False
    
    # Check if there exists x in N(u)\{v} and y in N(v)\{u} such that x!=y and (x,y) in E.
    # This is equivalent to: (adj & mask_v[None, :]) has any True in rows where mask_u is True,
    # excluding the case where x==y (which is impossible since adj[x,x]=0).
    
    # adj[x] & mask_v gives for each x, the intersection of N(x) with N(v)\{u}
    intersections = np.sum(adj & mask_v[None, :], axis=1)
    
    # Check if any x in N(u)\{v} has intersection > 0
    return bool(np.any(intersections[mask_u] > 0))

def _find_blocking_edges(adj, u, v):
    """
    If adding (u,v) creates a C4, find the edges (x,y) that are part of such C4s.
    Returns a list of (x, y) pairs.
    """
    neighbors_u = np.where(adj[u])[0]
    neighbors_u = neighbors_u[neighbors_u != v]
    
    neighbors_v = np.where(adj[v])[0]
    neighbors_v = neighbors_v[neighbors_v != u]
    
    blocking = []
    for x in neighbors_u:
        # Check neighbors of x that are also in neighbors_v
        common = np.intersect1d(np.where(adj[x])[0], neighbors_v)
        for y in common:
            if x != y:
                blocking.append((x, y))
    return blocking

def _greedy_fill(adj, n, target, deadline):
    """
    Greedily add edges to increase min degree.
    Returns True if target min degree is reached.
    """
    while time.time() < deadline:
        degrees = adj.sum(axis=1)
        min_deg = np.min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        # Pick one randomly to break symmetry
        u = np.random.choice(min_vertices)
        
        # Get non-neighbors of u
        non_neighbors = np.where(~adj[u])[0]
        non_neighbors = non_neighbors[non_neighbors != u]
        
        if len(non_neighbors) == 0:
            return False
            
        np.random.shuffle(non_neighbors)
        
        added = False
        for v in non_neighbors:
            if not _check_c4_add(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                added = True
                break  # Restart loop to re-evaluate degrees
        
        if not added:
            # Stuck in greedy phase
            return False
            
    return False

def _local_search_swaps(adj, n, target, deadline):
    """
    Local search using edge swaps to increase min degree.
    Assumes adj is C4-free.
    Returns True if target min degree is reached.
    """
    # We try to improve the graph from its current state
    for _ in range(1000):  # Max iterations per restart
        if time.time() > deadline:
            return False
            
        degrees = adj.sum(axis=1)
        min_deg = np.min(degrees)
        
        if min_deg >= target:
            return True
            
        # Find vertex with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        u = np.random.choice(min_vertices)
        
        # Get non-neighbors of u
        non_neighbors = np.where(~adj[u])[0]
        non_neighbors = non_neighbors[non_neighbors != u]
        
        if len(non_neighbors) == 0:
            return False
            
        np.random.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if _check_c4_add(adj, u, v):
                # Try to find a swap
                blockers = _find_blocking_edges(adj, u, v)
                for x, y in blockers:
                    # Check if removing (x,y) and adding (u,v) is safe
                    # 1. Removing (x,y) never creates a C4.
                    # 2. Adding (u,v) after removing (x,y) must not create a C4.
                    #    Since (x,y) was a blocker, removing it breaks that specific C4.
                    #    Could it create a NEW C4? No, because we are just removing an edge.
                    #    The new edge (u,v) is added to a graph that is C4-free (after removal).
                    #    We must ensure that (u,v) doesn't create a C4 with OTHER edges.
                    #    The _check_c4_add on the MODIFIED graph is the correct check.
                    
                    # Perform tentative swap
                    adj[x, y] = False
                    adj[y, x] = False
                    adj[u, v] = True
                    adj[v, u] = True
                    
                    # Check if the new graph is C4-free
                    # We know it was C4-free before removal. Removal keeps it C4-free.
                    # Adding (u,v) might create a C4.
                    # We can just check if the graph is C4-free globally? Too slow.
                    # We can check if adding (u,v) to the graph-without-(x,y) creates a C4.
                    # This is exactly what _check_c4_add does, but we already modified adj.
                    # Let's revert and check properly?
                    # Actually, if (x,y) was the ONLY blocker, then it's safe.
                    # If there were other blockers, it's not safe.
                    # So we must check if (u,v) creates a C4 in the graph without (x,y).
                    
                    # To be safe, let's verify the whole graph is C4-free?
                    # For n=47, checking whole graph is O(n^4) ~ 5M ops. Too slow for every step.
                    # But we can check locally.
                    # The only new C4s must involve (u,v).
                    # So we just need to check if (u,v) creates a C4 in the current adj (with x,y removed).
                    # We can do this by temporarily setting adj[u,v]=False, checking, then setting True.
                    # Or simpler: just check if the current adj (with swap) is C4-free.
                    
                    # Efficient check:
                    # Is the current graph C4-free?
                    # We can check if any C4 exists.
                    # Since we only added one edge (u,v), any new C4 must contain (u,v).
                    # So we just need to check if (u,v) creates a C4 in the graph where (x,y) is removed.
                    # Let's call a helper _check_c4_add_on_graph(adj, u, v) which assumes adj[u,v] is False.
                    # But adj[u,v] is True now.
                    
                    # Let's revert (u,v) for the check
                    adj[u, v] = False
                    adj[v, u] = False
                    
                    if not _check_c4_add(adj, u, v):
                        # Safe to keep the swap
                        adj[u, v] = True
                        adj[v, u] = True
                        improved = True
                        break # Restart local search loop
                    else:
                        # Not safe, revert everything
                        adj[x, y] = True
                        adj[y, x] = True
                        # (u,v) is already False
                        continue
                        
                if improved:
                    break
            else:
                # No C4, just add
                adj[u, v] = True
                adj[v, u] = True
                improved = True
                break
                
        if not improved:
            # Could not improve u
            # Try a random restart of the local search state?
            # Or just break and let the outer loop restart.
            break
            
    return False

def _robust_construct(n, target, deadline):
    """
    Combines greedy and local search with multiple restarts.
    """
    # Try a few random greedy starts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        adj = np.zeros((n, n), dtype=bool)
        
        # 1. Greedy Phase
        if _greedy_fill(adj, n, target, deadline):
            return (n, _to_edges(adj))
            
        # 2. Local Search Phase
        # The greedy phase might have left the graph C4-free but not meeting the degree target.
        # Try to fix it with swaps.
        if _local_search_swaps(adj, n, target, deadline):
            return (n, _to_edges(adj))
            
        # 3. If that fails, try a different random initialization
        # Randomly add some edges first, then greedy?
        # Or just restart.
        
    return None

def _to_edges(adj):
    edges = []
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
