"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def _verify(edges):
    """Verify the graph satisfies all constraints. Returns True if valid."""
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
    
    m = len(edges)
    
    # Check diameter is exactly 2 using BFS
    # For each pair (u, v) not adjacent, there must be a common neighbor
    # And for adjacent pairs, distance is 1 (so diameter <= 2 requires all non-adjacent pairs have distance 2)
    # Also need diameter >= 2, i.e., not all pairs are adjacent (which would be K_n)
    
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                # Must have a common neighbor
                if not (adj[u] & adj[v]):
                    return False
    
    # Check edge-criticality: deleting each edge must increase diameter above 2 or disconnect
    for e_idx, (u, v) in enumerate(edges):
        # Build graph without this edge
        adj2 = [set() for _ in range(n)]
        for i, j in edge_set:
            if (i, j) == (min(u, v), max(u, v)):
                continue
            adj2[i].add(j)
            adj2[j].add(i)
        
        # Check if graph is disconnected or diameter > 2
        disconnected_or_large = False
        for a in range(n):
            # BFS from a
            visited = {a}
            queue = [a]
            dist = {a: 0}
            while queue:
                cur = queue.pop(0)
                for nb in adj2[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            if len(visited) < n:
                disconnected_or_large = True
                break
            for b in range(n):
                if a != b and b in visited:
                    if dist[b] > 2:
                        disconnected_or_large = True
                        break
            if disconnected_or_large:
                break
        if not disconnected_or_large:
            return False
    
    return True

def construct():
    """Construct K_{8,8} complete bipartite graph on vertices 0..15."""
    n = P['n']
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
