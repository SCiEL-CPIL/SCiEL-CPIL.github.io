import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
    1. Try LUW construction for a guaranteed valid (but potentially large) graph.
    2. Use remaining time to search for smaller graphs using random generation.
    3. Return the smallest valid graph found.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_n = float('inf')
    best_edges = None

    # 1. LUW Construction (Fallback)
    # For k=3, g=16, LUW with d=8 gives n=13122, girth>=17.
    # This is valid but large.
    if k == 3:
        # Try LUW with d=8
        try:
            n_luw, edges_luw = _construct_luw(k, 8)
            if _has_girth_at_least_fast(n_luw, edges_luw, g):
                best_n = n_luw
                best_edges = edges_luw
        except Exception:
            pass

    # 2. Random Search for Smaller Graphs
    # Moore bound for k=3, g=16 is 762.
    # Best known is 936.
    # We search n from 762 upwards.
    
    if k == 3 and g == 16:
        search_start = 762
        search_end = 1000 # Try up to 1000
        
        # We will iterate n from search_start to search_end
        # But we should spend more time on promising n values.
        # Let's just iterate and check time.
        
        for n in range(search_start, search_end + 1):
            if time.time() > deadline - 10:
                break
            
            # Try to find a valid graph of size n
            # Number of attempts depends on remaining time
            # For smaller n, fewer attempts might suffice if it's unlikely.
            # For n around 936, it's known to exist.
            
            max_attempts = 50
            # Adjust attempts based on n?
            # Let's try a fixed number of attempts per n, but break early if found.
            
            found = False
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                edges = _generate_random_3reg(n)
                if edges is None:
                    continue
                
                # Check girth
                if _has_girth_at_least_fast(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        # If we find a graph close to Moore bound, we can stop?
                        # Or just keep searching for smaller n?
                        # Since we iterate n upwards, the first valid n is the smallest.
                        # But we started at 762. If we find one at 762, it's optimal.
                        # If not, we go to 763, etc.
                        # So we can return immediately if we find a valid graph for the current n?
                        # Yes, because we are increasing n.
                        return n, edges
            
            # If not found for this n, try next n

    # 3. Return Best
    if best_n != float('inf'):
        return best_n, best_edges
    
    # Last resort: Return a large LUW graph if nothing else worked
    # Or raise an error? No, must return valid graph.
    # The LUW construction should have worked.
    raise RuntimeError("No valid graph found")

def _construct_luw(k, d):
    """
    Construct LUW graph with degree k+1? No, degree q+1.
    For k=3, q=2? No, q must be such that q+1=k.
    If k=3, q=2.
    But LUW with q=2 has issues with girth.
    Let's use q=3, degree 4? No, task requires degree 3.
    Standard LUW for cubic graphs uses q=2?
    Actually, the Lubotzky-Phillips-Sarnak construction for cubic graphs uses q=2.
    But as established, q=2 LUW might have short girth.
    Let's check the girth of LUW(2, d).
    If it fails the girth check, we skip it.
    """
    q = k - 1 # For k=3, q=2.
    if q < 2:
        q = 2
        
    # If q=2, we use F_2.
    # Rules for LUW on F_2 are tricky.
    # Let's try to use a known valid construction if possible.
    # For now, we rely on the random search.
    # If LUW fails, best_n remains inf, and we hope random search works.
    
    # Simple LUW implementation for general q
    # But for k=3, q=2.
    
    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return x * y % q

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                # Standard LUW rules
                # a, b depend on c and q
                # For q=2, we need specific rules.
                # Let's use the rules from inspiration 1
                if c == 2:
                    a, b = 1, 1
                elif c == 3:
                    a, b = 2, 1
                elif c == 4:
                    a, b = 1, 2
                else:
                    r, c0 = (c - 5) % 4, c - (c - 5) % 4
                    a, b = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
                
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def _generate_random_3reg(n):
    """
    Generate a random 3-regular graph on n vertices using configuration model.
    Returns list of edges or None if invalid (multi-edges/loops).
    """
    if n < 4 or n % 2 != 0: # 3n must be even, so n must be even
        return None
    
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    adj = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    return edges

def _has_girth_at_least_fast(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks from every vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    half_g = g // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle through u-v
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
# EVOLVE-BLOCK-END
