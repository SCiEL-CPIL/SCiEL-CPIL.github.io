import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For k=3, we use q=3.
    q = 3
    def build_luw(d):
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _petersen_edges():
    """
    Return the 15 edges of the Petersen graph (10 vertices, 3-regular, girth 5).
    """
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]

def _voltage_lift_petersen(k, g, deadline):
    if k != 3:
        return None
    
    edges = _petersen_edges()
    n_base = 10
    m_edges = len(edges)
    
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    # Find all cycles in Petersen graph up to length g-1
    # Petersen has girth 5. Cycles of length 5, 6, 7, ...
    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    # Sort cycles by length for faster rejection
    cycles.sort(key=lambda c: len(c))
    
    # We want n = 10 * m to be small.
    # Try m values starting from small primes.
    # To get girth >= 16, we need no cycle in the lift of length < 16.
    # A cycle in the lift projects to a cycle in the base with a certain "winding number".
    # If the base cycle has length L and the voltage sum is s, the lift cycle length is m * |s| (if s != 0) or m (if s=0, but that's a shorter cycle).
    # Actually, if the voltage sum is s, the cycle in the lift has length L * (m / gcd(L, m))? No.
    # In a cyclic voltage graph, a closed walk in the base with voltage sum s lifts to a cycle of length L * m / gcd(L, m) if s is not 0 mod m?
    # Wait, the lift of a cycle of length L with voltage sum s is a single cycle of length L * m / gcd(L, s) ? No.
    # The lift decomposes into gcd(L, s) cycles, each of length L * m / gcd(L, s) ?
    # Let's check. If s=0, it's m disjoint copies of the cycle, each of length L. So we need s != 0 for all cycles of length L < g.
    # If s != 0, the length of the lifted cycle is L * m / gcd(L, s) ?
    # Example: L=5, m=7, s=1. gcd(5,1)=1. Length = 5*7 = 35.
    # Example: L=5, m=10, s=1. gcd(5,1)=1. Length = 50.
    # Example: L=5, m=10, s=2. gcd(5,2)=1. Length = 50.
    # Example: L=5, m=10, s=5. gcd(5,5)=5. Length = 5*10/5 = 10.
    # So we need to ensure that for every base cycle of length L < g, the lifted cycle length is >= g.
    # This is complicated. The simple check "voltage sum != 0" only ensures that the lift is not a disjoint union of short cycles.
    # But it might still have a single long cycle that wraps around multiple times? No, a simple cycle in the base lifts to a simple cycle in the lift if and only if the voltage sum is not 0?
    # Actually, if the base cycle is simple and the voltage sum is s, the lift is a single cycle of length L * m / gcd(L, s) if we consider the walk that goes around the base cycle once?
    # No, the lift of a specific cycle (walk) is a walk. If the walk is simple (no repeated vertices) in the base, is the lift simple?
    # Yes, if the base cycle is simple, the lift is simple.
    # The length of the lifted simple cycle is L * m / gcd(L, s) ?
    # Let's re-verify.
    # Vertex in lift: (v, j). Edge (u,v) with voltage w. (u, j) -> (v, j+w).
    # Start at (v0, 0).
    # After L steps, we are at (v0, S) where S is the sum of voltages.
    # To close the cycle, we need to return to (v0, 0).
    # This happens after k steps where k * L is a multiple of m? No.
    # The position after t steps is (v_{t mod L}, S * (t // L) + partial_sum(t mod L)).
    # We want to return to (v0, 0).
    # This requires t to be a multiple of L. Let t = k * L.
    # Then position is (v0, k * S).
    # We need k * S = 0 mod m.
    # Smallest k > 0 is m / gcd(S, m).
    # So the length of the lifted cycle is L * (m / gcd(S, m)).
    # We need L * (m / gcd(S, m)) >= g.
    
    # So for each base cycle C of length L < g, we need to assign voltages such that:
    # L * (m / gcd(sum(C), m)) >= g
    # This implies m / gcd(sum(C), m) >= ceil(g / L)
    # Let K = ceil(g / L).
    # We need gcd(sum(C), m) <= m / K.
    # This is a constraint on the voltage sum.
    
    # Since we are doing random search, we can just check this condition directly.
    
    # Precompute L for each cycle
    cycle_lengths = [len(C) for C in cycles]
    
    # Try m values
    # We want n = 10 * m to be small.
    # Smallest m such that 10*m >= Moore bound?
    # Moore bound for k=3, g=16 is 2 * (1 + 2 * sum_{i=0}^{6} 2^i) = 2 * (1 + 2 * (2^7 - 1)) = 2 * (1 + 126) = 254.
    # So m >= 26.
    
    # Let's try m from 26 upwards.
    # To speed up, we can filter m values.
    # For L=5, K = ceil(16/5) = 4.
    # We need gcd(s, m) <= m/4.
    # This means s is not a multiple of m/4 (if m is divisible by 4).
    
    candidates = []
    for m in range(26, 100):
        if time.time() > deadline:
            break
        
        # Quick check: for L=5, is it possible?
        # We need to find voltages such that for all 5-cycles, gcd(s, m) <= m/4.
        # There are 12 5-cycles in Petersen.
        
        # Random search
        found = False
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(m_edges)]
            # Ensure no voltage is 0? 0 is allowed but might be bad.
            
            valid = True
            for C in cycles:
                L = len(C)
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
                K = (g + L - 1) // L
                if m // K < 1:
                    valid = False
                    break
                if s % (m // K) == 0: # This is a necessary condition for gcd(s,m) > m/K?
                    # Actually we need gcd(s, m) <= m // K.
                    # If m // K is not an integer, we need ceil.
                    pass
                
                # Correct check:
                g_s_m = __import__('math').gcd(s, m)
                lifted_len = L * (m // g_s_m)
                if lifted_len < g:
                    valid = False
                    break
            
            if valid:
                # Construct lift
                n = n_base * m
                lift_edges = []
                for idx, (u, v) in enumerate(edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        lift_edges.append((a, b))
                return n, lift_edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try Voltage Lift of Petersen
    result = _voltage_lift_petersen(k, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
