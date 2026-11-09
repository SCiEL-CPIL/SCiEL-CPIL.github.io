"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u, v), max(u, v)))
    
    # Generate candidate edges: must be bipartite (different parity) to ensure triangle-free
    # and not in base edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if u % 2 == v % 2:
                continue
            candidates.append((u, v))
            
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
        
    def get_diameter(added_edges):
        adj = build_adj(added_edges)
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                cur = q.popleft()
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        q.append(nb)
            if -1 in dist:
                return -1
            max_diam = max(max_diam, max(dist))
        return max_diam

    # Greedy construction
    current = []
    # We want to minimize edges, so we start empty and add only when necessary?
    # Or start with a superset and remove?
    # Given the small N, let's try to build a valid solution first, then optimize.
    
    # Strategy: Add edges that maximize the "reach" or reduce the longest path.
    # We'll iterate until diameter <= 4.
    
    # To speed up, we can sort candidates by length descending? 
    # Longer edges reduce diameter faster.
    candidates_sorted = sorted(candidates, key=lambda x: x[1]-x[0], reverse=True)
    
    # Greedy Add
    while True:
        diam = get_diameter(current)
        if diam <= 4:
            break
        if diam == -1:
             # Not connected? Path is connected, so this shouldn't happen unless we messed up.
             break
             
        best_edge = None
        best_diam = diam + 1 # We want to minimize new diameter
        
        # Try each candidate not in current
        for edge in candidates_sorted:
            if edge in current:
                continue
            trial = current + [edge]
            new_diam = get_diameter(trial)
            if new_diam == -1:
                continue
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = edge
            if new_diam <= 4:
                # Found an edge that achieves target
                # We might find a better one (fewer total edges later), but for now, grab it?
                # Actually, greedy might pick a suboptimal edge if it just barely hits 4.
                # But let's just take the one that minimizes diameter.
                pass
        
        if best_edge is None:
            # No edge improves diameter? Should not happen if graph is connected and diam > 4
            # unless we are stuck in a local minimum of diameter?
            # Actually, adding any edge to a connected graph keeps it connected.
            # Diameter can only decrease or stay same.
            # If no edge decreases diameter, we are stuck?
            # But we need to reach 4.
            # If current diam is 5, and no edge makes it 4, we might need multiple edges.
            # But greedy adds one at a time.
            # If best_edge is None, it means no single edge reduces diameter?
            # That's unlikely for a path.
            # Let's just pick the longest edge not used as a fallback.
            for edge in candidates_sorted:
                if edge not in current:
                    best_edge = edge
                    break
            
        if best_edge is None:
            break
            
        current.append(best_edge)
        
    # Now we have a valid solution. Try to remove edges to minimize count.
    def try_remove_edges(added_edges):
        removed = []
        changed = True
        while changed:
            changed = False
            for i in range(len(added_edges)):
                trial = added_edges[:i] + added_edges[i+1:]
                if get_diameter(trial) <= 4:
                    added_edges = trial
                    changed = True
                    break
        return added_edges

    current = try_remove_edges(current)
    
    return [[int(u), int(v)] for u, v in current]
# EVOLVE-BLOCK-END
