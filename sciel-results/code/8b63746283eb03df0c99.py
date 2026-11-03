import time
import sys
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # For the specific case k=5, g=9, try a systematic search
    # For general cases, fall back to random generation with girth checking
    
    if k == 5 and g == 9:
        result = _construct_k5_g9(start_time, deadline)
        if result is not None:
            return result
    
    # General fallback: random regular graph with girth verification
    return _construct_general(k, g, start_time, deadline)

def _construct_k5_g9(start_time, deadline):
    """
    For k=5, g=9: Try to find a 5-regular graph with girth >= 9.
    Strategy: Generate random 5-regular graphs and check girth.
    We need n such that girth is at least 9.
    For random 5-regular graphs, girth is approximately log_4(n).
    To get girth 9, we need n to be large enough.
    n >= 4^8 = 65536 is a rough estimate, but let's try smaller values
    and check computationally.
    """
    
    # Try various n values, starting from a reasonable lower bound
    # Moore bound for k=5, g=9: 1 + 5 + 20 + 80 + 320 + 1280 = 1706 (girth 11)
    # For girth 9: 1 + 5 + 20 + 80 + 320 = 426 (lower bound)
    # Best known is 1116, so we're looking for something in that range.
    
    # Let's try n values and see if we can find one with girth >= 9
    # Start from a reasonable estimate
    candidate_ns = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1600, 1800, 2000]
    
    for n in candidate_ns:
        if (n * k) % 2 != 0:
            continue
        if time.time() > deadline - 30:
            break
            
        # Try a few random graphs for this n
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            edges = _generate_random_regular(k, n)
            if edges is None:
                continue
                
            # Check girth
            girth = _compute_girth(edges, n)
            if girth >= 9:
                return n, edges
    
    return None

def _construct_general(k, g, start_time, deadline):
    """
    General construction: random regular graph with girth >= g.
    """
    
    # Estimate n from Moore bound
    # For odd girth g = 2r+1: n >= 1 + k + k(k-1) + ... + k(k-1)^(r-1)
    # For even girth g = 2r: n >= 2(1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1
        term = k
        for i in range(r):
            moore += term
            term *= (k - 1)
    else:
        r = g // 2
        moore = 2
        term = 1
        for i in range(r - 1):
            moore += term
            term *= (k - 1)
    
    # Start from Moore bound, try increasing n
    n = max(moore, 2 * k)
    if n % 2 == 0:
        n += 1  # Ensure n*k is even? Actually n*k must be even, not n
    
    # Adjust n so n*k is even
    while (n * k) % 2 != 0:
        n += 1
    
    # Try increasing n until we find a valid graph
    max_n = 5000  # Upper bound to avoid infinite loop
    while n <= max_n and time.time() < deadline - 10:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            edges = _generate_random_regular(k, n)
            if edges is None:
                continue
                
            girth = _compute_girth(edges, n)
            if girth >= g:
                return n, edges
        
        n += 2  # Increment by 2 to maintain parity
    
    # If we didn't find one, return the best we have (even if girth < g)
    # This will score 0, but better than crashing
    edges = _generate_random_regular(k, n)
    if edges is not None:
        return n, edges
    
    # Last resort: return an empty-ish valid graph
    return 2, []

def _generate_random_regular(k, n):
    """
    Generate a random k-regular simple graph on n vertices.
    Returns list of edges or None if failed.
    """
    if n < k:
        return None
    if (n * k) % 2 != 0:
        return None
    
    # Use configuration model with rejection
    max_attempts = 100
    for _ in range(max_attempts):
        # Create half-edge list
        half_edges = []
        for i in range(n):
            half_edges.extend([i] * k)
        
        random.shuffle(half_edges)
        
        edges = []
        edge_set = set()
        valid = True
        
        for i in range(0, n * k, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append((u, v))
        
        if valid:
            return edges
    
    return None

def _compute_girth(edges, n):
    """
    Compute the girth of a graph using BFS from each vertex.
    Returns the length of the shortest cycle, or float('inf') if no cycle.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS from start
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
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    
    return min_cycle if min_cycle != float('inf') else n + 1
