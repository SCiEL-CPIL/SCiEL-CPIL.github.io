import time
import random
import numpy as np
from collections import deque

def _moore_bound(k, g):
    if g % 2 == 0:
        r = g // 2
        if k == 2:
            return g
        return int(k * ((k - 1) ** (r - 1) - 1) / (k - 2))
    else:
        r = (g - 1) // 2
        if k == 2:
            return g
        return int(1 + k * ((k - 1) ** (r - 1) - 1) / (k - 2))

def _luw_construct(k, g):
    q = k
    d = max(3, (g - 2) // 2)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
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

def _find_short_cycles(base_adj, max_len):
    """
    Finds all simple cycles of length < max_len in the base graph.
    Returns a list of lists of edges (tuples (u, v)).
    """
    n = len(base_adj)
    cycles = []
    
    # Use DFS to find cycles
    # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the start,
    # and the second smallest is the first step.
    
    for start in range(n):
        # Neighbors of start
        for first in base_adj[start]:
            if first <= start:
                continue
            # DFS from 'first', keeping track of path
            # Path: [start, first, ...]
            # We want to close back to start.
            
            # Recursive DFS might be deep, but graph is small (Petersen).
            def dfs(current, path, visited):
                if len(path) >= max_len:
                    return
                for nb in base_adj[current]:
                    if nb == start and len(path) >= 3: # Cycle found
                        # Record cycle
                        # Edges: (path[0], path[1]), ..., (path[-1], start)
                        cycle_edges = []
                        for i in range(len(path)):
                            u = path[i]
                            v = path[(i+1) % len(path)]
                            if u > v: u, v = v, u
                            cycle_edges.append((u, v))
                        cycles.append(cycle_edges)
                    elif nb not in visited and nb > start:
                        # Only visit vertices > start to avoid duplicates
                        # Exception: we allow closing to start
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, path, visited)
                        path.pop()
                        visited.remove(nb)
                        
            visited = {start, first}
            dfs(first, [start, first], visited)
            
    return cycles

def _construct_lift(base_edges, voltages, m):
    """
    Construct the lift graph.
    base_edges: list of (u, v)
    voltages: dict (u, v) -> voltage (int)
    m: group size
    Returns: n, edges
    """
    # Map base vertices to 0..N-1
    # Base vertices are 0..9 for Petersen
    n_base = 10
    n = n_base * m
    edges = []
    
    # For each base edge (u, v), we have m edges in the lift
    # ((u, i), (v, i + v_uv))
    # And ((v, i), (u, i - v_uv)) which is the same set of edges
    # To avoid duplicates, we only add edges from u to v for u < v?
    # No, the lift is undirected.
    # Edge in base: u-v.
    # Lift edges:
    # For i in 0..m-1:
    #   (u*m + i, v*m + (i + v_uv) % m)
    
    for u, v in base_edges:
        # Ensure consistent orientation for voltage
        # Let's say voltage is defined for (u, v) with u < v
        # If u > v, we swap and negate voltage?
        # Let's store voltages for (min(u,v), max(u,v))
        a, b = (u, v) if u < v else (v, u)
        val = voltages.get((a, b), 0)
        # If the edge in base_edges was (v, u) and u > v, we used val for (u, v).
        # The lift edge from u to v adds val.
        # The lift edge from v to u adds -val.
        
        for i in range(m):
            # u is at index u*m + i
            # v is at index v*m + (i + val) % m
            u_lift = u * m + i
            v_lift = v * m + (i + val) % m
            edges.append((u_lift, v_lift))
            
    return n, edges

def _check_girth_bfs(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        return _luw_construct(k, g)
        
    # Petersen Graph
    # 0: 1, 4, 5
    # 1: 0, 2, 6
    # 2: 1, 3, 7
    # 3: 2, 4, 8
    # 4: 3, 0, 9
    # 5: 0, 7, 8
    # 6: 1, 9, 8
    # 7: 2, 5, 6
    # 8: 3, 5, 6
    # 9: 4, 6, 7
    
    petersen_adj = [
        [1, 4, 5],
        [0, 2, 6],
        [1, 3, 7],
        [2, 4, 8],
        [3, 0, 9],
        [0, 7, 8],
        [1, 9, 8],
        [2, 5, 6],
        [3, 5, 6],
        [4, 6, 7]
    ]
    
    petersen_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                petersen_edges.append((u, v))
                
    # Find short cycles in Petersen
    short_cycles = _find_short_cycles(petersen_adj, g)
    
    # Search for m and voltages
    # Moore bound for k=3, g=16 is 381.
    # n = 10*m. So m >= 39.
    # Try m from 40 upwards.
    
    best_result = None
    best_n = float('inf')
    
    for m in range(40, 200):
        if time.time() > deadline - 10:
            break
            
        # Try to find voltages
        found = False
        attempts = 0
        while not found and attempts < 50 and time.time() < deadline - 5:
            attempts += 1
            # Initialize random voltages
            voltages = {}
            for u, v in petersen_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                
            # Local search to fix violations
            # Violation: cycle sum == 0
            # We'll just check all short cycles.
            valid = True
            for _ in range(20): # Max iterations of local search
                violations = []
                for cycle in short_cycles:
                    s = 0
                    for u, v in cycle:
                        s += voltages[(u, v)]
                    if s % m == 0:
                        violations.append(cycle)
                        
                if not violations:
                    found = True
                    break
                    
                # Pick a random violation
                cycle = random.choice(violations)
                # Change a random edge in the cycle
                u, v = random.choice(cycle)
                voltages[(u, v)] = (voltages[(u, v)] + 1) % m
                
        if found:
            # Construct lift
            n, edges = _construct_lift(petersen_edges, voltages, m)
            
            # Verify
            if _check_girth_bfs(n, edges, g):
                return n, edges
            # If verification fails (due to composite cycles), try again with new voltages
            # But we break out of m loop? No, we just continue to next m or retry.
            # For simplicity, if BFS fails, we assume m is too small or bad luck.
            # We can retry a few times for this m.
            # Let's just break and try next m.
            
    # Fallback to LUW
    return _luw_construct(k, g)
