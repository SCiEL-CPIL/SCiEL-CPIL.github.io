import time
import random
import sys
from itertools import combinations, permutations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)
    
    # Strategy: Lift K6 (5-regular, 6 vertices) by prime p
    # Target n ~ 426. 426 / 6 = 71.
    # Try primes around 71.
    
    # K6 edges
    k6_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
    
    # Find all simple cycles in K6
    # K6 is small, we can enumerate them.
    # Cycles of length 3, 4, 5, 6.
    cycles = []
    # Length 3
    for i in range(6):
        for j in range(i+1, 6):
            for l in range(j+1, 6):
                cycles.append([(i, j), (j, l), (l, i)])
    # Length 4
    for verts in combinations(range(6), 4):
        v0, v1, v2, v3 = verts
        cycles.append([(v0, v1), (v1, v2), (v2, v3), (v3, v0)])
        cycles.append([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
        cycles.append([(v0, v2), (v2, v1), (v1, v3), (v3, v0)])
    # Length 5
    for verts in combinations(range(6), 5):
        v0 = verts[0]
        others = list(verts[1:])
        seen = set()
        for perm in permutations(others):
            cycle_edges = []
            prev = v0
            for u in perm:
                cycle_edges.append((min(prev, u), max(prev, u)))
                prev = u
            cycle_edges.append((min(prev, v0), max(prev, v0)))
            key = tuple(sorted(cycle_edges))
            if key not in seen:
                seen.add(key)
                cycles.append(cycle_edges)
    # Length 6
    for verts in combinations(range(6), 6):
        v0 = verts[0]
        others = list(verts[1:])
        seen = set()
        for perm in permutations(others):
            cycle_edges = []
            prev = v0
            for u in perm:
                cycle_edges.append((min(prev, u), max(prev, u)))
                prev = u
            cycle_edges.append((min(prev, v0), max(prev, v0)))
            key = tuple(sorted(cycle_edges))
            if key not in seen:
                seen.add(key)
                cycles.append(cycle_edges)
                
    # Remove duplicates in cycles list just in case
    unique_cycles = []
    seen_cycles = set()
    for c in cycles:
        key = tuple(sorted(c))
        if key not in seen_cycles:
            seen_cycles.add(key)
            unique_cycles.append(c)
            
    # Try primes
    primes = [67, 71, 73, 79, 83]
    
    for p in primes:
        if time.time() > deadline - 5:
            break
            
        n = 6 * p
        
        # Search for voltage assignment
        # Use backtracking to satisfy cycle sum != 0 mod p
        # We have 15 edges in K6.
        # Constraints: sum of voltages on any cycle != 0 mod p.
        
        # Let's use a simple backtracking search
        # Order edges
        edge_order = list(k6_edges)
        
        def backtrack(idx, voltages):
            if time.time() > deadline - 2:
                return None
            
            if idx == len(edge_order):
                # Check all cycles
                for c in unique_cycles:
                    s = 0
                    for u, v in c:
                        s = (s + voltages[(u, v)]) % p
                    if s == 0:
                        return None
                return dict(voltages)
                
            u, v = edge_order[idx]
            for val in range(p):
                voltages[(u, v)] = val
                voltages[(v, u)] = (p - val) % p
                
                # Pruning: check if any cycle is fully assigned and sums to 0
                # This is expensive, so maybe just check if we can prune early?
                # For now, just proceed.
                
                res = backtrack(idx + 1, voltages)
                if res is not None:
                    return res
                    
            return None
            
        # Randomize edge order and values?
        # Backtracking might be slow if p is large.
        # Let's try a few random starts or just one deterministic one.
        # Actually, the constraint is "sum != 0".
        # For p=71, the space is huge.
        
        # Alternative: Random assignment with local search?
        # Or just try a few random assignments first, then backtrack if time permits.
        
        found_volts = None
        for trial in range(50):
            if time.time() > deadline - 5:
                break
                
            # Random voltage assignment
            voltages = {}
            for u, v in k6_edges:
                voltages[(u, v)] = random.randint(0, p-1)
                voltages[(v, u)] = (p - voltages[(u, v)]) % p
                
            # Check all cycles
            valid = True
            for c in unique_cycles:
                s = 0
                for u, v in c:
                    s = (s + voltages[(u, v)]) % p
                if s == 0:
                    valid = False
                    break
                    
            if valid:
                found_volts = voltages
                break
                
        if found_volts is None and time.time() < deadline - 10:
            # Try backtracking
            voltages = {}
            found_volts = backtrack(0, voltages)
            
        if found_volts is not None:
            # Construct lift
            edges = []
            for u, v in k6_edges:
                w = found_volts[(u, v)]
                for i in range(p):
                    u_idx = u * p + i
                    v_idx = v * p + (i + w) % p
                    edges.append((u_idx, v_idx))
                    
            # Verify girth
            if _check_girth(edges, n, g):
                return n, edges
                    
    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _check_girth(edges, n, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g, deadline):
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
        
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
        
    return len(order), edges
