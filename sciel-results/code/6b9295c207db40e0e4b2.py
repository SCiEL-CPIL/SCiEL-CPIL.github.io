import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_quadratic_residue(a, p):
    """Check if a is a quadratic residue mod p."""
    if a == 0:
        return False
    return pow(a, (p - 1) // 2, p) == 1

def _paley_graph(n):
    """
    Construct a Paley graph on n vertices (n must be prime, n ≡ 3 mod 4).
    Vertex i is connected to vertex j if (j-i) mod n is a quadratic residue.
    This graph is C4-free for certain primes and has high minimum degree.
    """
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if _is_quadratic_residue(diff, n):
                adj[i, j] = True
                adj[j, i] = True
    return adj

def _check_c4_full(adj):
    """Check if graph is C4-free. Returns True if C4-free."""
    n = adj.shape[0]
    # For each pair of non-adjacent vertices, check if they have 2+ common neighbors
    # Actually, C4 exists iff there exist 4 distinct vertices a,b,c,d with edges ab, bc, cd, da
    # Equivalently: for any two non-adjacent vertices u,v, |N(u) ∩ N(v)| <= 1
    # And for any two adjacent vertices u,v, |N(u) ∩ N(v)| = 0 (otherwise triangle, but we check C4)
    # Actually C4: a-b-c-d-a. So a and c are non-adjacent (or could be adjacent in K4, but C4 specifically)
    # Let me just check: for all pairs u,v, if u!=v, count common neighbors. If >=2 and u,v not both endpoints of a C4...
    # Simpler: iterate over all 4-tuples? Too slow.
    # Better: for each pair (u,v), compute |N(u) ∩ N(v)|. If u and v are not adjacent and |N(u)∩N(v)| >= 2, 
    #   then there exist x,y in N(u)∩N(v), x≠y, giving u-x-v-y-u? No, that's not right.
    # C4: u-x-v-y-u requires edges ux, xv, vy, yu. So x in N(u)∩N(v) and y in N(u)∩N(v) with x≠y.
    # But we also need x≠y and the four vertices distinct.
    # So: if u,v are non-adjacent and |N(u)∩N(v)| >= 2, pick two distinct x,y from intersection.
    # Then u-x-v-y-u is a C4 if x≠y and all four distinct. Since u,v non-adjacent, x,y distinct from u,v.
    # x,y distinct from each other. So yes, C4 exists.
    # If u,v are adjacent, a C4 through edge uv would be u-x-v-y-u, but that requires x in N(u)∩N(v) and y in N(u)∩N(v), x≠y.
    # But wait, that's the same condition. Actually for adjacent u,v, a C4 containing edge uv is u-x-v-y-u where x,y are common neighbors.
    # So the condition is: for ANY pair u,v (adjacent or not), if |N(u)∩N(v)| >= 2, then a C4 exists?
    # No! If u,v are adjacent and have common neighbors x,y, then u-x-v-y-u is a 4-cycle only if x≠y and the edges exist.
    # Edges: ux (yes, x in N(u)), xv (yes, x in N(v)), vy (yes, y in N(v)), yu (yes, y in N(u)).
    # And u,v adjacent. So u-x-v-y-u is indeed a C4 (possibly with chord uv, but it's still a C4 subgraph).
    # So the check is: for all pairs u≠v, |N(u)∩N(v)| <= 1.
    for u in range(n):
        for v in range(u + 1, n):
            common = np.sum(adj[u] & adj[v])
            if common >= 2:
                return False
    return True

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) to graph defined by adjacency matrix adj creates a C4.
    Returns True if a C4 is created, False otherwise.
    Assumes adj[u,v] is currently False.
    """
    # N(u) excluding v
    mask_u = adj[u].copy()
    mask_u[v] = False
    # N(v) excluding u  
    mask_v = adj[v].copy()
    mask_v[u] = False
    
    # Check if there exists x in N(u)\{v} and y in N(v)\{u} such that x!=y and (x,y) in E.
    intersections = np.sum(adj & mask_v[None, :], axis=1)
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
                break
        
        if not added:
            return False
            
    return False

def _local_search_swaps(adj, n, target, deadline):
    """
    Local search using edge swaps to increase min degree.
    Assumes adj is C4-free.
    Returns True if target min degree is reached.
    """
    for _ in range(2000):
        if time.time() > deadline:
            return False
            
        degrees = adj.sum(axis=1)
        min_deg = np.min(degrees)
        
        if min_deg >= target:
            return True
            
        min_vertices = np.where(degrees == min_deg)[0]
        u = np.random.choice(min_vertices)
        
        non_neighbors = np.where(~adj[u])[0]
        non_neighbors = non_neighbors[non_neighbors != u]
        
        if len(non_neighbors) == 0:
            return False
            
        np.random.shuffle(non_neighbors)
        
        improved = False
        for v in non_neighbors:
            if _check_c4_add(adj, u, v):
                blockers = _find_blocking_edges(adj, u, v)
                np.random.shuffle(blockers)
                for x, y in blockers:
                    adj[x, y] = False
                    adj[y, x] = False
                    
                    adj[u, v] = False
                    adj[v, u] = False
                    
                    if not _check_c4_add(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        improved = True
                        break
                    else:
                        adj[x, y] = True
                        adj[y, x] = True
                        continue
                        
                if improved:
                    break
            else:
                adj[u, v] = True
                adj[v, u] = True
                improved = True
                break
                
        if not improved:
            break
            
    return False

def _to_edges(adj):
    edges = []
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def _verify(adj, n, target):
    """Verify graph is C4-free and has min degree >= target."""
    if not _check_c4_full(adj):
        return False
    if np.min(adj.sum(axis=1)) < target:
        return False
    return True

def _robust_construct(n, target, deadline):
    """
    Try multiple strategies to construct a C4-free graph with min degree >= target.
    """
    # Strategy 1: Paley graph (if n is prime)
    if n in [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3]:
        adj = _paley_graph(n)
        if _verify(adj, n, target):
            return (n, _to_edges(adj))
        
        # Try to boost the Paley graph
        adj2 = adj.copy()
        if _greedy_fill(adj2, n, target, deadline):
            if _verify(adj2, n, target):
                return (n, _to_edges(adj2))
        
        adj3 = adj.copy()
        if _local_search_swaps(adj3, n, target, deadline):
            if _verify(adj3, n, target):
                return (n, _to_edges(adj3))
    
    # Strategy 2: Random greedy
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        adj = np.zeros((n, n), dtype=bool)
        
        if _greedy_fill(adj, n, target, deadline):
            if _verify(adj, n, target):
                return (n, _to_edges(adj))
            
        if _local_search_swaps(adj, n, target, deadline):
            if _verify(adj, n, target):
                return (n, _to_edges(adj))
        
    return None

def construct():
    """
    Construct a C4-free graph on n vertices with min degree >= n - 39.
    Maximize n.
    """
    deadline = time.time() + 145
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        required_min_deg = n - 39
        
        result = _robust_construct(n, required_min_deg, deadline)
        
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
            
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
