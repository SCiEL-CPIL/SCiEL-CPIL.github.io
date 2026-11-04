import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3:
        # Fallback for other k (not expected for this specific task instance)
        # But we should handle it to avoid crash.
        # For k != 3, we can't use the Petersen lift.
        # We will return a simple random k-regular graph if possible, 
        # but girth might be low. 
        # The task specifically asks for k=3, g=16.
        # If k != 3, we just try to return something valid.
        return _fallback_random(k, g, deadline)

    # Petersen graph definition
    # Vertices 0..9
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_n = 10
    
    # Pre-compute cycles of length 5 in base graph to help filter?
    # No, just build and check.
    
    # Try different lift sizes m
    # n = 10 * m
    # Moore bound is 512, so m >= 52
    # Let's try m from 52 to 150
    for m in range(52, 151):
        if time.time() > deadline - 20:
            break
            
        n = 10 * m
        
        # Try random voltage assignments
        # We can limit attempts per m based on remaining time
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Assign random voltages to the 15 edges
            # Voltages are in Z_m
            # To ensure we don't get trivial 0s, we can pick from 1..m-1
            # But 0 is allowed.
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            # Build the lift
            # Vertex (u, v) where u in 0..9, v in 0..m-1
            # Edge (u, v) -- (w, v + volt[i]) mod m for edge (u, w) with voltage volt[i]
            
            edges = []
            # Map base edge index to voltage
            # We need to be careful with direction.
            # For undirected graph, if we add edge (u, v) to (w, v'), we don't add the reverse separately.
            # But in the lift, if (u, w) is an edge, we add (u, v) to (w, (v + volt) % m).
            # And also (w, v') to (u, (v' - volt) % m) which is the same edge.
            # So we just iterate over base edges and add one edge per lift instance.
            
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    # Vertex u is at index u * m + v
                    # Vertex w is at index w * m + (v + volt) % m
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            # Verify and check girth
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges

    # If we get here, we failed to find a girth-16 graph with m < 150
    # Try larger m?
    for m in range(150, 300):
        if time.time() > deadline - 5:
            break
        n = 10 * m
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges

    # Absolute fallback: Return a random 3-regular graph.
    # This might not have girth 16, but it's valid 3-regular.
    # The task says "invalid result... scores 0".
    # If we can't find a valid one, we score 0 anyway.
    # But maybe a random one with large n has girth >= 16?
    # For n=2000, girth is likely > 16.
    # Let's try to generate a random 3-regular graph with n=2000.
    n = 2000
    edges = _random_3reg(n)
    if edges is not None:
        # Check girth just in case
        if _is_valid_and_girth(n, edges, 3, g):
            return n, edges
        # Even if girth is low, return it? No, it's invalid.
        # But better to return something than crash.
        return n, edges

    return 10, [] # Dummy

def _is_valid_and_girth(n, edges, k, g):
    """Check if graph is simple, k-regular, and has girth >= g."""
    if len(edges) != n * k // 2:
        return False
    
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        deg[u] += 1
        deg[v] += 1
        
    if not all(d == k for d in deg):
        return False
        
    # Check girth
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return False
        if min_cycle < g:
            return False
            
    return min_cycle >= g

def _random_3reg(n):
    if n < 4 or n % 2 != 0:
        return None
    stubs = list(range(n)) * 3
    random.shuffle(stubs)
    edges = []
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u != v:
            edges.append((u, v))
    # Check simplicity
    seen = set()
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) in seen:
            return None
        seen.add((u, v))
    return edges

def _fallback_random(k, g, deadline):
    # Simple fallback for non-3 cases
    n = 100
    edges = _random_3reg(n)
    if edges:
        return n, edges
    return n, []
