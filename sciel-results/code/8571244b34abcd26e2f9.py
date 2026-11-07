"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _validate_graph(edges, n=16, deadline=None):
    """
    Check if the graph satisfies:
    1. Diameter exactly 2
    2. Every edge is essential (removing it increases diameter above 2 or disconnects)
    """
    m = len(edges)
    if m == 0:
        return False

    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v or u >= n or v >= n:
            return False
        u, v = min(u, v), max(u, v)
        if (u, v) in edge_set:
            return False
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)

    # Check diameter is exactly 2
    def bfs_dist(start):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        for qi in range(len(queue)):
            u = queue[qi]
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    for s in range(n):
        d = bfs_dist(s)
        if -1 in d:
            return False
        max_d = max(d)
        if max_d > 2:
            return False
    # Diameter must be exactly 2 (not 1)
    # If diameter were 1, graph would be complete, but we have n=16, m<120
    # For diameter to be exactly 2, there must exist at least one pair at distance 2
    has_dist_2 = False
    for s in range(n):
        d = bfs_dist(s)
        if 2 in d:
            has_dist_2 = True
            break
    if not has_dist_2:
        return False

    # Check edge essentiality
    for i, (u, v) in enumerate(edges):
        u, v = min(u, v), max(u, v)
        # Build new adjacency without this edge
        adj2 = [set() for _ in range(n)]
        for a, b in edges:
            if a == u and b == v:
                continue
            if a == v and b == u:
                continue
            a, b = min(a, b), max(a, b)
            adj2[a].add(b)
            adj2[b].add(a)
        
        # Check if removing edge (u,v) disconnects or increases diameter above 2
        # First check if u and v are still connected
        dist = [-1] * n
        dist[u] = 0
        queue = [u]
        for qi in range(len(queue)):
            x = queue[qi]
            for y in adj2[x]:
                if dist[y] == -1:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        
        if dist[v] == -1:
            # Disconnected -> essential
            continue
        
        if dist[v] > 2:
            # Distance increased above 2 -> essential
            continue
        
        # If dist[v] <= 2, we need to check if ANY other pair has distance > 2
        # For speed, we can just check if diameter > 2
        # BFS from all nodes is expensive, but for 16 nodes it's fine
        diameter_after = 0
        for s in range(n):
            d2 = [-1] * n
            d2[s] = 0
            q = [s]
            for qi in range(len(q)):
                x = q[qi]
                for y in adj2[x]:
                    if d2[y] == -1:
                        d2[y] = d2[x] + 1
                        q.append(y)
            if -1 in d2:
                # Disconnected
                diameter_after = float('inf')
                break
            max_d = max(d2)
            if max_d > diameter_after:
                diameter_after = max_d
            if diameter_after > 2:
                break
        
        if diameter_after > 2:
            # Essential
            continue
        else:
            # Not essential
            return False

    return True


def construct():
    start_time = time.time()
    deadline = 90.0  # 90s wall-clock limit, use margin
    
    n = 16
    best_edges = []
    best_count = 0

    # Start with K_{8,8} as baseline (64 edges)
    base_edges = []
    for u in range(8):
        for v in range(8, 16):
            base_edges.append([u, v])
    
    if _validate_graph(base_edges, n, deadline):
        best_edges = list(base_edges)
        best_count = 64
    
    # Try to add edges within the same partition
    # Part A: 0..7, Part B: 8..15
    # Currently no intra-part edges
    # We can try adding some intra-part edges and removing some inter-part edges
    
    # Strategy: Start from K_{8,8}, try to swap: remove one inter-part edge, add one intra-part edge
    # This keeps edge count at 64 but might allow more edges if we can add multiple intra-part edges
    
    # Actually, let's try a different approach:
    # Start from empty, add edges greedily
    # Or better: use local search starting from K_{8,8}
    
    # Let's try: for each possible intra-part edge, see if we can add it and still satisfy constraints
    # If adding an intra-part edge breaks essentiality of some edge, we need to remove that edge
    # Net change: +1 intra, -1 inter = 0 change, or if we can add multiple intra without removing, +k
    
    # Let's try adding intra-part edges one by one to the base
    all_intra = []
    for i in range(8):
        for j in range(i+1, 8):
            all_intra.append([i, j])
            all_intra.append([i+8, j+8])
    
    # For each subset of intra edges (up to size 3 for now), check if graph is valid
    # This is 2^16 * 2^16 which is too much. Let's be smarter.
    
    # Try adding a single intra edge to K_{8,8}
    for ie in all_intra:
        candidate = list(base_edges)
        candidate.append(ie)
        if len(candidate) > 60 and time.time() > deadline - 5.0:
            break
        if _validate_graph(candidate, n, deadline):
            if len(candidate) > best_count:
                best_edges = list(candidate)
                best_count = len(candidate)
            if time.time() > deadline - 2.0:
                break
    
    # Try adding two intra edges
    if time.time() < deadline - 10.0:
        for i in range(len(all_intra)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, len(all_intra)):
                candidate = list(base_edges)
                candidate.append(all_intra[i])
                candidate.append(all_intra[j])
                if _validate_graph(candidate, n, deadline):
                    if len(candidate) > best_count:
                        best_edges = list(candidate)
                        best_count = len(candidate)
                    if time.time() > deadline - 2.0:
                        break
    
    # Now try local search: swap inter-part edges with intra-part edges
    # Remove one inter-part edge, add one intra-part edge
    if time.time() < deadline - 10.0:
        inter_edges = list(base_edges)
        for ie_idx, ie in enumerate(all_intra):
            if time.time() > deadline - 5.0:
                break
            for re_idx, re in enumerate(inter_edges):
                # Skip if already trying
                candidate = []
                for e in base_edges:
                    if e[0] == re[0] and e[1] == re[1]:
                        continue
                    if e[0] == re[1] and e[1] == re[0]:
                        continue
                    candidate.append(e)
                candidate.append(ie)
                if _validate_graph(candidate, n, deadline):
                    if len(candidate) > best_count:
                        best_edges = list(candidate)
                        best_count = len(candidate)
                if time.time() > deadline - 2.0:
                    break
    
    # Final cleanup: ensure types are pure Python ints and lists
    cleaned = []
    for e in best_edges:
        u, v = int(e[0]), int(e[1])
        cleaned.append([u, v])
    
    if time.time() > deadline - 1.0:
        return cleaned
        
    return cleaned

# EVOLVE-BLOCK-END
