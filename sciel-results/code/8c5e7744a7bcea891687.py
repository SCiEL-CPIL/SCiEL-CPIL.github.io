"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _check_validity(added_edges):
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Triangle-free check
    for u in range(n):
        neighbors = adj[u]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    return False
    
    # Connected check
    visited = [False] * n
    visited[0] = True
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    if not all(visited):
        return False
    
    # Diameter <= 4 check
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = [src]
        while queue:
            node = queue.pop(0)
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    # Fallback: single hub at 12 connected to all odd vertices (10 edges)
    fallback = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]

    # Try two-hub configurations to reduce edge count
    # Hub must be even vertex (bipartite: even-odd only)
    # Even vertices: 0,2,4,6,8,10,12,14,16,18,20,22
    # Odd vertices: 1,3,5,7,9,11,13,15,17,19,21,23
    
    best = fallback
    best_len = len(fallback)
    
    # Precompute: for each even vertex, which odd vertices can it connect to?
    # (must not be adjacent on path, i.e., |u-v| != 1)
    even_verts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    odd_verts = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    # For each even vertex, the set of odd vertices it can connect to
    # (not path-adjacent)
    hub_candidates = {}
    for h in even_verts:
        cands = []
        for o in odd_verts:
            if abs(h - o) != 1:
                cands.append(o)
        hub_candidates[h] = set(cands)
    
    # Two-hub search: try all pairs of even hubs, assign odd vertices to minimize total edges
    for i in range(len(even_verts)):
        if time.time() > deadline - 2:
            break
        h1 = even_verts[i]
        for j in range(i + 1, len(even_verts)):
            h2 = even_verts[j]
            s1 = hub_candidates[h1]
            s2 = hub_candidates[h2]
            
            # Odd vertices that can only be covered by h1
            only1 = s1 - s2
            # Odd vertices that can only be covered by h2
            only2 = s2 - s1
            # Odd vertices that can be covered by either
            both = s1 & s2
            # Odd vertices that can't be covered by either
            neither = set(odd_verts) - s1 - s2
            
            if neither:
                continue
            
            # We must connect all 'only1' to h1 and all 'only2' to h2
            # For 'both', we choose assignment to minimize total
            # Total = len(only1) + len(only2) + |both| (each 'both' vertex needs exactly one edge)
            total_edges = len(only1) + len(only2) + len(both)
            
            if total_edges < best_len:
                # Build the edge list
                edges = []
                for o in only1:
                    edges.append([h1, o])
                for o in only2:
                    edges.append([h2, o])
                for o in both:
                    # Assign to whichever hub - try both and pick valid one
                    # First try assigning to h1
                    edges.append([h1, o])
                
                if _check_validity(edges):
                    best = edges
                    best_len = total_edges
                else:
                    # Try assigning 'both' to h2 instead
                    edges2 = []
                    for o in only1:
                        edges2.append([h1, o])
                    for o in only2:
                        edges2.append([h2, o])
                    for o in both:
                        edges2.append([h2, o])
                    
                    if _check_validity(edges2):
                        best = edges2
                        best_len = total_edges
                    else:
                        # Try mixed assignment - greedy
                        edges3 = []
                        for o in only1:
                            edges3.append([h1, o])
                        for o in only2:
                            edges3.append([h2, o])
                        for o in both:
                            edges3.append([h1, o])
                        # This is same as first try, so skip
                        pass
    
    # Also try single hub at different positions
    for h in even_verts:
        if time.time() > deadline - 2:
            break
        s = hub_candidates[h]
        neither = set(odd_verts) - s
        if neither:
            continue
        total_edges = len(s)
        if total_edges < best_len:
            edges = [[h, o] for o in s]
            if _check_validity(edges):
                best = edges
                best_len = total_edges
    
    # Try 3-hub configurations if best is still 10
    if best_len >= 10:
        for i in range(len(even_verts)):
            if time.time() > deadline - 2:
                break
            h1 = even_verts[i]
            for j in range(i + 1, len(even_verts)):
                if time.time() > deadline - 2:
                    break
                h2 = even_verts[j]
                for k in range(j + 1, len(even_verts)):
                    if time.time() > deadline - 2:
                        break
                    h3 = even_verts[k]
                    s1 = hub_candidates[h1]
                    s2 = hub_candidates[h2]
                    s3 = hub_candidates[h3]
                    
                    # Each odd vertex must be covered by at least one hub
                    uncovered = set(odd_verts) - s1 - s2 - s3
                    if uncovered:
                        continue
                    
                    # Minimum edges = number of odd vertices (each needs at least one edge)
                    # but we might need extra if some can only be covered by one hub
                    # Actually each odd vertex needs exactly one edge (to one of its covering hubs)
                    # So total = 12 (all odd vertices)
                    # But wait - we only need to cover odd vertices that are "far" from path
                    # Actually in this model, every odd vertex needs an edge to a hub
                    total_edges = 12  # all 12 odd vertices
                    if total_edges < best_len:
                        # Assign each odd vertex to a hub that covers it
                        # Try greedy: assign to first available hub
                        edges = []
                        valid = True
                        for o in odd_verts:
                            assigned = False
                            for h in [h1, h2, h3]:
                                if o in hub_candidates[h]:
                                    edges.append([h, o])
                                    assigned = True
                                    break
                            if not assigned:
                                valid = False
                                break
                        if valid and _check_validity(edges):
                            best = edges
                            best_len = total_edges
    
    if best_len < len(fallback):
        return best
    return fallback
# EVOLVE-BLOCK-END
