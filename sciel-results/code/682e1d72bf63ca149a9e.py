"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 16}

def _diameter_and_criticality(n, adj):
    """Check if graph has diameter exactly 2 and every edge is critical.
    adj: list of sets, adj[i] = set of neighbors of i.
    Returns (is_valid, diameter)."""
    # Check diameter <= 2 and >= 2
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # Check all reachable
        for i in range(n):
            if dist[i] == -1:
                return (False, -1)  # disconnected
            if dist[i] > 2:
                return (False, dist[i])
    # Diameter is exactly 2 (need at least one pair at distance 2)
    has_dist2 = False
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for i in range(n):
            if dist[i] == 2:
                has_dist2 = True
                break
        if has_dist2:
            break
    if not has_dist2:
        return (False, 1)  # diameter 1, not 2
    
    # Check criticality: for each edge, removing it must increase diameter or disconnect
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    
    for (u, v) in edge_list:
        # Remove edge (u,v)
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Check if disconnected or diameter > 2
        disconnected_or_gt2 = False
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            for i in range(n):
                if dist[i] == -1 or dist[i] > 2:
                    disconnected_or_gt2 = True
                    break
            if disconnected_or_gt2:
                break
        
        # Restore edge
        adj[u].add(v)
        adj[v].add(u)
        
        if not disconnected_or_gt2:
            return (False, 2)
    
    return (True, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    
    if n < 2:
        return []
        
    # Start with K_{8,8} which has 64 edges and is known valid
    half = n // 2
    adj = [set() for _ in range(n)]
    base_edges = []
    for u in range(half):
        for v in range(half, n):
            adj[u].add(v)
            adj[v].add(u)
            base_edges.append([int(u), int(v)])
    
    # Verify base is valid
    valid, diam = _diameter_and_criticality(n, adj)
    if not valid:
        # Shouldn't happen, but fallback
        return base_edges
    
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    # Try to add edges from within A or within B
    candidate_edges = []
    # Edges within A
    for u in range(half):
        for v in range(u+1, half):
            candidate_edges.append((u, v))
    # Edges within B
    for u in range(half, n):
        for v in range(u+1, n):
            candidate_edges.append((u, v))
    
    # Shuffle candidates
    random.seed(42)
    random.shuffle(candidate_edges)
    
    # Try adding one edge at a time
    for (u, v) in candidate_edges:
        if time.time() > deadline:
            break
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        
        valid, _ = _diameter_and_criticality(n, adj)
        
        if valid:
            best_edges.append([int(u), int(v)])
            best_count += 1
            # Keep the edge, try to add more
        else:
            # Remove edge
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Try to add more edges after the first pass
    # Rebuild candidate list with current adj
    for _ in range(3):  # A few more passes
        if time.time() > deadline:
            break
        added_any = False
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline:
                    break
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    valid, _ = _diameter_and_criticality(n, adj)
                    if valid:
                        best_edges.append([int(u), int(v)])
                        best_count += 1
                        added_any = True
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)
            if time.time() > deadline:
                break
        if not added_any:
            break
    
    # Build final edge list from adj to ensure consistency
    final_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                final_edges.append([int(u), int(v)])
    
    return final_edges
# EVOLVE-BLOCK-END
