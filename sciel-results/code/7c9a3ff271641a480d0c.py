import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    target_edges = 91
    
    def creates_c4(u, v, adj, edge_set):
        # Check if adding (u, v) creates a C4
        # This happens if there is a path of length 3: u-w-x-v
        # w in N(u), x in N(v), and (w, x) is an edge
        for w in adj[u]:
            for x in adj[v]:
                if (w, x) in edge_set or (x, w) in edge_set:
                    return True
        return False

    def find_c4_free_90():
        for attempt in range(100):
            if time.time() > deadline - 30: break
            
            edges = set()
            adj = {i: set() for i in range(n)}
            
            # Add 90 random edges
            while len(edges) < 90:
                u, v = random.sample(range(n), 2)
                if u > v: u, v = v, u
                if (u, v) not in edges:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                    
            # Remove edges to break C4s
            # We need to ensure no C4 exists.
            # A C4 exists if there is a pair (u, v) with >= 2 common neighbors.
            # Or simpler: iterate and remove edges from found C4s.
            
            changed = True
            while changed and len(edges) >= 90:
                changed = False
                # Find a C4
                # Check all pairs for common neighbors >= 2
                for u in range(n):
                    for v in range(u + 1, n):
                        common = adj[u].intersection(adj[v])
                        if len(common) >= 2:
                            # Found a C4. Remove an edge.
                            # Pick a common neighbor w
                            w = next(iter(common))
                            # Edge (u, w) is part of C4 u-w-v-...
                            # Actually u-w and w-v are edges.
                            # Remove (u, w)
                            if (u, w) in edges:
                                edges.remove((u, w))
                                adj[u].remove(w)
                                adj[w].remove(u)
                                changed = True
                                break
                    if changed: break
                
                # If we removed an edge, try to add one back
                if len(edges) < 90:
                    added = False
                    for _ in range(200):
                        u, v = random.sample(range(n), 2)
                        if u > v: u, v = v, u
                        if (u, v) not in edges:
                            if not creates_c4(u, v, adj, edges):
                                edges.add((u, v))
                                adj[u].add(v)
                                adj[v].add(u)
                                added = True
                                break
                    if not added:
                        break # Stuck, restart
            
            if len(edges) == 90:
                # Verify no C4
                has_c4 = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if len(adj[u].intersection(adj[v])) >= 2:
                            has_c4 = True
                            break
                    if has_c4: break
                if not has_c4:
                    return list(edges), adj
        return None, None

    base_edges, adj = find_c4_free_90()
    
    if base_edges:
        edge_set = set(base_edges)
        best_edge = None
        min_common = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    common = len(adj[u].intersection(adj[v]))
                    if common < min_common:
                        min_common = common
                        best_edge = (u, v)
                        
        if best_edge:
            final_edges = base_edges + [list(best_edge)]
            return final_edges

    # Fallback
    edges = set()
    adj = {i: set() for i in range(n)}
    while len(edges) < 91:
        u, v = random.sample(range(n), 2)
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
    return [[u, v] for u, v in edges]
