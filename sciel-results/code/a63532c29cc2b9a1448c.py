import itertools
from collections import deque

def _diameter_edges(edges, n):
    """Return (diameter, set of edges that are 'critical' for keeping diameter <=2).
    An edge is critical if removing it increases diameter above 2 or disconnects the graph.
    We compute this by checking each edge removal's effect on all-pairs distances."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check connectivity first
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    if not all(visited):
        return n + 1, set()  # disconnected
    
    # Compute all-pairs shortest paths (BFS from each vertex)
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    
    # Current diameter
    diam = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] > diam:
                diam = dist[i][j]
    
    if diam > 2:
        return diam, set()
    
    # Check each edge for criticality
    critical = set()
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    for e in edge_set:
        u, v = e
        # Remove edge and check if diameter still <= 2
        # We need to recompute distances without this edge
        # This is expensive but n=16 is small
        adj2 = [set() for _ in range(n)]
        for (a, b) in edge_set:
            if (a, b) != e and (b, a) != e:
                adj2[a].add(b)
                adj2[b].add(a)
        
        # Check connectivity
        visited2 = [False] * n
        queue2 = deque([0])
        visited2[0] = True
        while queue2:
            uu = queue2.popleft()
            for vv in adj2[uu]:
                if not visited2[vv]:
                    visited2[vv] = True
                    queue2.append(vv)
        if not all(visited2):
            critical.add(e)
            continue
        
        # Compute diameter without this edge
        diam2 = 0
        for s in range(n):
            dist_s = [-1] * n
            dist_s[s] = 0
            q = deque([s])
            while q:
                uu = q.popleft()
                for vv in adj2[uu]:
                    if dist_s[vv] == -1:
                        dist_s[vv] = dist_s[uu] + 1
                        q.append(vv)
            for d in dist_s:
                if d > diam2:
                    diam2 = d
                    if diam2 > 2:
                        break
            if diam2 > 2:
                break
        
        if diam2 > 2:
            critical.add(e)
    
    return diam, critical


def construct():
    n = 16
    
    # Start with K_8,8 as base
    half = n // 2
    base_edges = []
    for u in range(half):
        for v in range(half, n):
            base_edges.append([u, v])
    
    # Try to add edges within each partition to see if we can increase count
    # while maintaining that every edge is critical
    
    # Actually, let's try a different approach: start with complete graph and remove edges
    # until every remaining edge is critical, trying to maximize edge count
    
    # For n=16, complete graph has 120 edges. That's too many to check all subsets.
    
    # Let's try: start with K_8,8 and try adding intra-partition edges one by one,
    # checking if the result still has all edges critical.
    
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    # Try adding intra-left edges
    left_intra = [(u, v) for u in range(half) for v in range(u + 1, half)]
    right_intra = [(half + u, half + v) for u in range(half) for v in range(u + 1, half)]
    
    all_candidate_edges = left_intra + right_intra
    
    # Greedy: try to add edges that maintain the critical property
    current_edges = list(base_edges)
    
    for cand in all_candidate_edges:
        u, v = cand
        # Check if this edge already exists
        exists = False
        for eu, ev in current_edges:
            if (eu == u and ev == v) or (eu == v and ev == u):
                exists = True
                break
        if exists:
            continue
        
        # Try adding this edge
        test_edges = current_edges + [[u, v]]
        
        # Check if all edges are still critical and diameter is 2
        # This is expensive, so we'll do a quick check first
        # Quick check: adding an edge can only decrease distances, so diameter stays <= 2
        # But it might make some existing edge non-critical
        
        # Let's check if the new graph has all edges critical
        # We need to verify that removing any edge makes diameter > 2
        
        # This is O(m * n^2) which for ~40 edges and n=16 is manageable
        adj = [set() for _ in range(n)]
        for eu, ev in test_edges:
            adj[eu].add(ev)
            adj[ev].add(eu)
        
        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        while queue:
            uu = queue.popleft()
            for vv in adj[uu]:
                if not visited[vv]:
                    visited[vv] = True
                    queue.append(vv)
        if not all(visited):
            continue
        
        # Check diameter <= 2
        diam_ok = True
        for s in range(n):
            dist_s = [-1] * n
            dist_s[s] = 0
            q = deque([s])
            while q:
                uu = q.popleft()
                for vv in adj[uu]:
                    if dist_s[vv] == -1:
                        dist_s[vv] = dist_s[uu] + 1
                        q.append(vv)
            for d in dist_s:
                if d > 2:
                    diam_ok = False
                    break
            if not diam_ok:
                break
        
        if not diam_ok:
            continue
        
        # Now check if all edges are critical
        all_critical = True
        edge_set = set()
        for eu, ev in test_edges:
            edge_set.add((min(eu, ev), max(eu, ev)))
        
        for e in edge_set:
            eu, ev = e
            adj2 = [set() for _ in range(n)]
            for (a, b) in edge_set:
                if (a, b) != e:
                    adj2[a].add(b)
                    adj2[b].add(a)
            
            # Check connectivity
            visited2 = [False] * n
            queue2 = deque([0])
            visited2[0] = True
            while queue2:
                uu = queue2.popleft()
                for vv in adj2[uu]:
                    if not visited2[vv]:
                        visited2[vv] = True
                        queue2.append(vv)
            if not all(visited2):
                continue  # This edge is critical
            
            # Check diameter
            diam2 = 0
            for s in range(n):
                dist_s = [-1] * n
                dist_s[s] = 0
                q = deque([s])
                while q:
                    uu = q.popleft()
                    for vv in adj2[uu]:
                        if dist_s[vv] == -1:
                            dist_s[vv] = dist_s[uu] + 1
                            q.append(vv)
                for d in dist_s:
                    if d > diam2:
                        diam2 = d
                        if diam2 > 2:
                            break
                if diam2 > 2:
                    break
            
            if diam2 <= 2:
                # This edge is NOT critical
                all_critical = False
                break
        
        if all_critical:
            current_edges.append([u, v])
    
    return current_edges
