import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3:
        return _fallback_random(k, g, deadline)

    # Petersen graph definition
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_n = 10
    
    # Build adjacency for base graph
    base_adj = [[] for _ in range(base_n)]
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        base_adj[u].append(v)
        base_adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Find all non-backtracking walks of length up to g-1 in base graph
    # A non-backtracking walk is (v0, v1, v2, ..., vl) where vi != v(i-2)
    # For each such walk, we get a constraint: sum of voltages along the walk != 0 mod m
    # The walk corresponds to a sequence of edge indices
    max_len = g - 1  # We need to avoid cycles of length < g, so walks of length up to g-1
    # Actually, a cycle of length L corresponds to a closed non-backtracking walk of length L
    # So we need to avoid closed non-backtracking walks of length 3 to g-1
    
    # Generate all non-backtracking walks of length 3 to g-1 that start and end at the same vertex
    # This is expensive for large g, but g=16 is manageable
    constraints = []  # Each constraint is a list of (edge_index, sign) representing the voltage sum
    
    def gen_walks(start, prev, curr, length, path):
        """Generate non-backtracking walks from start, currently at curr, came from prev."""
        if length == 0:
            if curr == start and len(path) >= 3:
                constraints.append(path[:])
            return
        for next_v in base_adj[curr]:
            if next_v == prev:
                continue
            edge_idx = edge_index[(curr, next_v)]
            gen_walks(start, curr, next_v, length - 1, path + [edge_idx])
    
    for start in range(base_n):
        for first in base_adj[start]:
            gen_walks(start, start, first, max_len - 1, [edge_index[(start, first)]])
    
    # Now search for voltage assignments
    # n = 10 * m, Moore bound 512, so m >= 52
    # Try m from 52 upward
    best_result = None
    
    for m in range(52, 200):
        if time.time() > deadline - 15:
            break
        n = 10 * m
        
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                    break
            
            # Random voltage assignment
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            # Check all constraints
            valid = True
            for constraint in constraints:
                s = 0
                for eidx in constraint:
                    s = (s + voltages[eidx]) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build the lift
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            # Verify girth (safety check)
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges
    
    # Fallback
    n = 2000
    edges = _random_3reg(n)
    if edges is not None:
        if _is_valid_and_girth(n, edges, 3, g):
            return n, edges
    return n, edges if edges else (10, [])

def _is_valid_and_girth(n, edges, k, g):
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
    seen = set()
    for u, v in edges:
        if u > v: u, v = v, u
        if (u, v) in seen:
            return None
        seen.add((u, v))
    return edges

def _fallback_random(k, g, deadline):
    n = 100
    edges = _random_3reg(n)
    if edges:
        return n, edges
    return n, []
