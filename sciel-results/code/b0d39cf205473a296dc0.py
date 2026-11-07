"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    # Bipartite partition: even and odd vertices
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    
    # Precompute base path distances
    def base_dist(a, b):
        return abs(a - b)
    
    # We'll try to find a small set of cross-partity edges
    # such that the augmented graph has diameter <= 4
    
    # Strategy: Try different hub configurations
    # A hub vertex h (odd) connected to even vertices gives short paths
    
    # Let's try to minimize by being selective about which even vertices
    # each odd hub connects to
    
    # Key insight: For diameter <= 4, we need every pair (u,v) to have
    # a path of length <= 4. The base path gives distance |u-v|.
    # Added edges can shortcut this.
    
    # With one odd hub h connected to a set S of even vertices:
    # dist(u,v) <= min(|u-v|, dist_to_hub(u)+1+dist_to_hub(v))
    # where dist_to_hub(x) = 1 if x in S and x != h, else base_dist(x,h)
    
    # Actually for the hub approach: if h is odd and connected to even e,
    # then for any even e1 in S and even e2 in S: dist(e1,e2) <= 2 (e1-h-e2)
    # For odd o1 not in {h}: dist(o1, e) where e in S: dist(o1,h)+1 <= base_dist(o1,h)+1
    # For odd o1, o2: dist(o1,o2) <= dist(o1,h)+1+dist(h,o2) if both can reach h via S
    
    # Let's try a two-hub approach: two odd vertices as hubs
    # This might allow fewer total edges
    
    # Try all pairs of odd hubs and find minimum edges needed
    
    def check_diameter(hubs, hub_connections):
        """Check if the graph with given hub connections has diameter <= 4."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for h, evens_list in hub_connections.items():
            for e in evens_list:
                adj[h].add(e)
                adj[e].add(h)
        
        # BFS from each vertex to check max distance
        from collections import deque
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for i in range(n):
                if dist[i] == -1 or dist[i] > 4:
                    return False
        return True
    
    # Try single hub approach first with selective connections
    best_edges = None
    best_count = 999
    
    for hub in odds:
        # For each even vertex, determine if connecting it to hub helps
        # We need to ensure all pairs have distance <= 4
        # Start with all evens except neighbors, then try to remove
        
        # Which evens MUST be connected?
        # An even vertex e not connected to hub has dist(e, hub) = |e-hub|
        # For pair (e, f) where both are far from hub, we might need both connected
        
        # Greedy: start with all non-neighbor evens, try to remove one at a time
        candidate = set()
        for e in evens:
            if e != hub - 1 and e != hub + 1:
                candidate.add(e)
        
        # Try to remove edges while maintaining diameter <= 4
        removed = set()
        for e in sorted(candidate, key=lambda x: abs(x - hub), reverse=True):
            trial = candidate - {e}
            if not trial:
                continue
            if check_diameter({hub: list(trial)}, {}):
                candidate = trial
        
        if len(candidate) < best_count:
            best_count = len(candidate)
            best_edges = [[hub, e] for e in sorted(candidate)]
    
    # Also try two-hub approach
    for h1 in odds:
        for h2 in odds:
            if h1 >= h2:
                continue
            # Each hub connects to some evens
            # Try: h1 connects to evens closer to h1, h2 to evens closer to h2
            c1 = set()
            c2 = set()
            for e in evens:
                if e == h1 - 1 or e == h1 + 1:
                    continue
                if e == h2 - 1 or e == h2 + 1:
                    continue
                d1 = abs(e - h1)
                d2 = abs(e - h2)
                if d1 < d2:
                    c1.add(e)
                elif d2 < d1:
                    c2.add(e)
                else:
                    c1.add(e)  # tie-break
            
            if check_diameter({h1: list(c1), h2: list(c2)}, {}):
                total = len(c1) + len(c2)
                if total < best_count:
                    best_count = total
                    best_edges = [[h1, e] for e in sorted(c1)] + [[h2, e] for e in sorted(c2)]
    
    if best_edges is None:
        # Fallback: original approach
        hub = 11
        best_edges = []
        for e in evens:
            if e != hub - 1 and e != hub + 1:
                best_edges.append([hub, e])
    
    return best_edges
# EVOLVE-BLOCK-END
