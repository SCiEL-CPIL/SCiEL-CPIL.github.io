import random
import itertools

def is_valid(edges, n):
    # Build adjacency list and degree
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Check all pairs with same degree
    for s in range(n):
        for t in range(s + 1, n):
            if degrees[s] != degrees[t]:
                continue
            
            # Check if there is a simple path of length 3 between s and t
            # s - u - v - t
            found = False
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[u]:
                    if v == s or v == t or v == u:
                        continue
                    if t in adj[v]:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def find_conflict(edges, n):
    """Returns a tuple (s, u, v, t) representing a conflict path, or None."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    for s in range(n):
        for t in range(s + 1, n):
            if degrees[s] != degrees[t]:
                continue
            
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[u]:
                    if v == s or v == t or v == u:
                        continue
                    if t in adj[v]:
                        return (s, u, v, t)
    return None

def construct():
    n = 15
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start with K_7,8 (Known valid, 56 edges)
    edges_k78 = []
    for u in range(7):
        for v in range(7, 15):
            edges_k78.append([u, v])
    if is_valid(edges_k78, n):
        best_count = len(edges_k78)
        best_edges = list(edges_k78)

    # Strategy 2: Start with K_15 and remove edges to fix conflicts
    # This explores a different region of the search space
    full_edges = [[u, v] for u in range(n) for v in range(u + 1, n)]
    
    # We will do a few random restarts of the removal process
    for restart in range(20):
        current_edges = list(full_edges)
        random.shuffle(current_edges)
        
        # Iterate until valid or stuck
        # To avoid long loops, limit iterations
        for _ in range(1000):
            conflict = find_conflict(current_edges, n)
            if conflict is None:
                # Valid
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                break
            
            s, u, v, t = conflict
            # Remove one of the edges in the path: (s,u), (u,v), (v,t)
            # Randomly choose which one to remove to explore different solutions
            path_edges = [
                (min(s, u), max(s, u)),
                (min(u, v), max(u, v)),
                (min(v, t), max(v, t))
            ]
            edge_to_remove = random.choice(path_edges)
            
            # Remove it
            # Find and remove from list
            for i, e in enumerate(current_edges):
                if (min(e[0], e[1]), max(e[0], e[1])) == edge_to_remove:
                    current_edges.pop(i)
                    break
        
        # After removal, try to add back edges (Local Search)
        if len(current_edges) > best_count:
             # Try to add edges to improve
             edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
             improved = True
             while improved:
                 improved = False
                 for u in range(n):
                     for v in range(u+1, n):
                         if (u,v) not in edge_set:
                             new_edges = current_edges + [[u,v]]
                             if is_valid(new_edges, n):
                                 current_edges.append([u,v])
                                 edge_set.add((u,v))
                                 improved = True
                                 if len(current_edges) > best_count:
                                     best_count = len(current_edges)
                                     best_edges = list(current_edges)

    return best_edges
