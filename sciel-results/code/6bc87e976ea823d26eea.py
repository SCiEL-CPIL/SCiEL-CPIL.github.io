"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(frozenset(e))
    
    # Build adjacency for base
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # All possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u+1, n):
            if frozenset((u,v)) not in base_edges:
                all_possible.append((u,v))
    
    def check_validity(added_edges):
        """Check if adding these edges keeps triangle-free, connected, diameter <= 4."""
        # Build new adjacency
        new_adj = [set() for _ in range(n)]
        for e in P['edges']:
            new_adj[e[0]].add(e[1])
            new_adj[e[1]].add(e[0])
        
        # Check for triangles and add edges
        for u, v in added_edges:
            # Check triangle: if any common neighbor, triangle forms
            if new_adj[u] & new_adj[v]:
                return False
            new_adj[u].add(v)
            new_adj[v].add(u)
        
        # Check connectedness and diameter via BFS from each node
        # Actually, just BFS from one node if connected, then check max distance
        # First check connectedness
        visited = [False] * n
        queue = [0]
        visited[0] = True
        dist = [-1] * n
        dist[0] = 0
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in new_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        if not all(visited):
            return False
        
        # Check diameter: max distance from any node to any other
        # Since graph is connected, do BFS from each node
        for src in range(n):
            d = [-1] * n
            d[src] = 0
            q = [src]
            qi2 = 0
            while qi2 < len(q):
                u = q[qi2]
                qi2 += 1
                for v in new_adj[u]:
                    if d[v] == -1:
                        d[v] = d[u] + 1
                        q.append(v)
            if max(d) > 4:
                return False
        
        return True
    
    # Try increasing number of edges
    for k in range(1, 15):
        if time.time() > deadline:
            break
        # Try to find a valid set of k edges
        # Use greedy/heuristic: try promising edge combinations
        # Heuristic: edges that connect far-apart vertices are more useful for diameter reduction
        
        # Sort possible edges by "usefulness" - prefer edges that connect vertices far apart in the path
        # Distance in path: |u-v|
        sorted_edges = sorted(all_possible, key=lambda e: abs(e[0]-e[1]), reverse=True)
        
        # Try combinations from the top candidates
        # For small k, we can try more combinations
        if k <= 3:
            candidates = sorted_edges[:50]
        elif k <= 5:
            candidates = sorted_edges[:80]
        else:
            candidates = sorted_edges[:120]
        
        found = False
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            if check_validity(list(combo)):
                return [list(e) for e in combo]
        
        if time.time() > deadline:
            break
    
    # Fallback: return a known valid solution (if we can find one quickly)
    # Try a specific structure: connect center vertices to many others
    # Let's try a bipartite hub structure
    # Hubs: 6, 12, 18 (evenly spaced)
    # Connect each hub to vertices at distance 2-4 from it, avoiding triangles
    
    # Actually, let's just return a valid solution we construct manually
    # For diameter 4 on 24 vertices, we can use:
    # Connect 0-4, 0-5, 1-5, 1-6, ... etc. But need to be careful about triangles.
    
    # Let's try a simple valid solution: add edges to create a "ladder" like structure
    # Or just return the first valid one we can find with a broader search
    
    # Broader search with more time
    for k in range(1, 12):
        if time.time() > deadline:
            break
        if k <= 4:
            candidates = sorted_edges[:100]
        else:
            candidates = sorted_edges[:150]
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            if check_validity(list(combo)):
                return [list(e) for e in combo]
    
    # Last resort: return empty (invalid, but better than timeout)
    return []
# EVOLVE-BLOCK-END
