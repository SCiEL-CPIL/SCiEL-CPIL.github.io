import random
import time
import sys

def has_c4(adj, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 is created."""
    nu = adj[u]
    nv = adj[v]
    
    if len(nv) < len(nu):
        for w in nv:
            if w == u: continue
            for x in adj[w]:
                if x in nu and x != v:
                    return True
    else:
        for x in nu:
            if x == v: continue
            for w in adj[x]:
                if w in nv and w != u:
                    return True
                    
    return False

def find_blocking_edge(adj, u, v):
    """
    If adding (u,v) creates a C4, find an edge (w,x) that is part of the C4 
    such that removing (w,x) might allow adding (u,v).
    Returns (w, x) or None.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We need w in N(v), x in N(u), and (w,x) is an edge.
    # Also w != u, x != v, w != x.
    
    if len(nv) < len(nu):
        for w in nv:
            if w == u: continue
            for x in adj[w]:
                if x in nu and x != v and x != w:
                    # (w,x) is the blocking edge in the C4 u-w-x-v-u
                    return (w, x)
    else:
        for x in nu:
            if x == v: continue
            for w in adj[x]:
                if w in nv and w != u and w != x:
                    return (w, x)
                    
    return None

def construct():
    target_n = 46
    target_deg = target_n - 39 # 7
    
    start_time = time.time()
    time_limit = 100
    
    # Helper to convert adj to edges list
    def adj_to_edges(adj):
        edges = []
        for u in range(len(adj)):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # 1. Greedy Initialization
    def greedy_init(n, target_edges):
        adj = [set() for _ in range(n)]
        candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(candidates)
        added = 0
        for u, v in candidates:
            if added >= target_edges:
                break
            if not has_c4(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
        return adj

    # 2. Local Search with Blocking Edge Removal
    def local_search(adj, n, target_deg, start_time, time_limit):
        degrees = [len(adj[i]) for i in range(n)]
        
        # We want to increase min degree
        # Iterate until time limit or success
        
        # Precompute list of edges for random removal? 
        # Or just pick random u, v.
        
        while time.time() - start_time < time_limit:
            curr_min = min(degrees)
            if curr_min >= target_deg:
                return adj
            
            # Identify low degree vertices
            low_verts = [i for i in range(n) if degrees[i] < target_deg]
            if not low_verts:
                return adj
            
            # Pick a random low degree vertex u
            u = random.choice(low_verts)
            
            # Pick a random non-neighbor v of u
            non_neighbors = [v for v in range(n) if v != u and v not in adj[u]]
            if not non_neighbors:
                continue
            
            v = random.choice(non_neighbors)
            
            # Check if we can add (u,v)
            if not has_c4(adj, u, v):
                # Safe to add
                adj[u].add(v)
                adj[v].add(u)
                degrees[u] += 1
                degrees[v] += 1
                continue
            
            # Blocked. Find blocking edge (w,x)
            block = find_blocking_edge(adj, u, v)
            if block:
                w, x = block
                # Try to remove (w,x) and add (u,v)
                # Check if removing (w,x) is "safe" (it is, removing edges never creates C4)
                # But we need to ensure that adding (u,v) after removing (w,x) is safe.
                # It might be that (w,x) was the *only* blocker, or there are others.
                # Let's simulate the move.
                
                # Temporarily remove (w,x)
                adj[w].remove(x)
                adj[x].remove(w)
                degrees[w] -= 1
                degrees[x] -= 1
                
                # Check if adding (u,v) is now safe
                if not has_c4(adj, u, v):
                    # Safe. Commit the move.
                    adj[u].add(v)
                    adj[v].add(u)
                    degrees[u] += 1
                    degrees[v] += 1
                    # Check if this move improved min degree
                    new_min = min(degrees)
                    # If new_min < curr_min, we might have hurt the min degree elsewhere (w or x dropped)
                    # But u and v increased. 
                    # If w or x were the unique min-degree vertices, and they dropped below the new min of u/v, 
                    # then min degree decreased.
                    # We should only accept if min degree did not decrease.
                    if new_min >= curr_min:
                        # Accept
                        pass
                    else:
                        # Reject: Revert
                        adj[u].remove(v)
                        adj[v].remove(u)
                        degrees[u] -= 1
                        degrees[v] -= 1
                        adj[w].add(x)
                        adj[x].add(w)
                        degrees[w] += 1
                        degrees[x] += 1
                else:
                    # Still blocked, revert removal
                    adj[w].add(x)
                    adj[x].add(w)
                    degrees[w] += 1
                    degrees[x] += 1
            else:
                # Should not happen if has_c4 returned True
                pass
                
        return adj

    # Main Loop
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Init
        # Try to get a dense graph
        adj = greedy_init(target_n, 200)
        
        # Check if valid
        min_deg = min(len(adj[i]) for i in range(target_n))
        if min_deg >= target_deg:
            return {'n': target_n, 'edges': adj_to_edges(adj)}
        
        # Refine with Local Search
        adj = local_search(adj, target_n, target_deg, start_time, time_limit)
        
        min_deg = min(len(adj[i]) for i in range(target_n))
        if min_deg >= target_deg:
            return {'n': target_n, 'edges': adj_to_edges(adj)}
            
    # Fallback
    return {'n': 39, 'edges': []}
