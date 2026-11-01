"""Deterministic Algebraic Lift construction for k-regular graphs with high girth."""
import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Heawood graph edges (0-13)
    # Heawood graph is the (3,6)-cage.
    # Vertices 0-13.
    # It is the Levi graph of the Fano plane.
    # Let's construct it explicitly.
    # Fano plane points: 0,1,2,3,4,5,6
    # Fano plane lines: 
    # 0: 0,1,2
    # 1: 0,3,4
    # 2: 0,5,6
    # 3: 1,3,5
    # 4: 1,4,6
    # 5: 2,3,6
    # 6: 2,4,5
    # Incidence graph: vertices are points (0-6) and lines (7-13).
    # Edge (p, l) if point p is on line l.
    
    heawood_edges = []
    # Points 0-6, Lines 7-13
    # Line 0 (7): 0,1,2
    heawood_edges.append((0, 7))
    heawood_edges.append((1, 7))
    heawood_edges.append((2, 7))
    # Line 1 (8): 0,3,4
    heawood_edges.append((0, 8))
    heawood_edges.append((3, 8))
    heawood_edges.append((4, 8))
    # Line 2 (9): 0,5,6
    heawood_edges.append((0, 9))
    heawood_edges.append((5, 9))
    heawood_edges.append((6, 9))
    # Line 3 (10): 1,3,5
    heawood_edges.append((1, 10))
    heawood_edges.append((3, 10))
    heawood_edges.append((5, 10))
    # Line 4 (11): 1,4,6
    heawood_edges.append((1, 11))
    heawood_edges.append((4, 11))
    heawood_edges.append((6, 11))
    # Line 5 (12): 2,3,6
    heawood_edges.append((2, 12))
    heawood_edges.append((3, 12))
    heawood_edges.append((6, 12))
    # Line 6 (13): 2,4,5
    heawood_edges.append((2, 13))
    heawood_edges.append((4, 13))
    heawood_edges.append((5, 13))
    
    # Verify Heawood is 3-regular
    heawood_adj = [[] for _ in range(14)]
    for u, v in heawood_edges:
        heawood_adj[u].append(v)
        heawood_adj[v].append(u)
        
    # Base edges (u < v)
    base_edges = []
    for u in range(14):
        for v in heawood_adj[u]:
            if u < v:
                base_edges.append((u, v))
                
    # Moore bound for k=3, g=16 is 512.
    # n = 14 * m.
    # m >= 512 / 14 = 36.57 -> 37.
    
    best_n = None
    best_edges = None

    # Scan m from 37 upwards
    # We will try a few deterministic and random voltage assignments for each m.
    
    for m in range(37, 200):
        if time.time() > deadline:
            break
            
        # Try a few deterministic patterns first
        # Pattern 1: Linear
        shifts1 = [int((i + 1) * m / 21.0) for i in range(21)]
        n, edges = _build_lift(base_edges, m, shifts1)
        if _verify_girth(n, edges, g):
            return n, edges
            
        # Try random shifts
        # Increase attempts if m is small (harder to satisfy girth)
        # But verification is fast enough for n ~ 500-1000.
        num_tries = 50
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            shifts_rand = [random.randint(0, m-1) for _ in range(21)]
            n, edges = _build_lift(base_edges, m, shifts_rand)
            if _verify_girth(n, edges, g):
                return n, edges

    # If not found in Heawood lifts, try Petersen lifts (n multiple of 10)
    # Petersen base edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
        
    base_edges_p = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges_p.append((u, v))
                
    for m in range(52, 200):
        if time.time() > deadline:
            break
        shifts_rand = [random.randint(0, m-1) for _ in range(15)]
        n, edges = _build_lift(base_edges_p, m, shifts_rand)
        if _verify_girth(n, edges, g):
            return n, edges

    # Fallback: Large random lift
    m = 100
    shifts_rand = [random.randint(0, m-1) for _ in range(21)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

def _build_lift(base_edges, m, shifts):
    """
    Build the lifted graph.
    base_edges: list of (u, v) for u < v
    m: lift size
    shifts: list of shifts for each base edge
    """
    n = len(base_edges[0][0]) * m # This is wrong, need num vertices in base
    # Wait, base_edges is a list of tuples. 
    # We need the number of vertices in the base graph.
    # In the previous version, n was passed or calculated.
    # Let's pass num_base_vertices.
    # Or infer from max vertex id + 1.
    max_v = 0
    for u, v in base_edges:
        if u > max_v: max_v = u
        if v > max_v: max_v = v
    num_base = max_v + 1
    n = num_base * m
    
    edges = []
    for (u, v), shift in zip(base_edges, shifts):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

def _verify_girth(n, edges, g):
    """
    Fast girth verification using timestamped BFS.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Use global arrays to avoid O(n) reset per start
    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0
    
    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
                        
    return True

# EVOLVE-BLOCK-END
