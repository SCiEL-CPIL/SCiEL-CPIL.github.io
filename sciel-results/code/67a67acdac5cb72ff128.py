import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Handle the specific case k=5, g=9 with the voltage lift strategy
    if k == 5 and g == 9:
        result = _voltage_lift_k55()
        if result is not None:
            return result
        # Fallback to LUW if voltage lift fails or times out
        return _luw_construct(k, g)
    
    # General fallback to LUW
    return _luw_construct(k, g)

def _voltage_lift_k55():
    """
    Constructs a 5-regular graph of girth 9 using a voltage lift of K_{5,5} over Z_m.
    Uses structured voltage assignments and verifies girth.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # Base graph K_{5,5}: U = {0,1,2,3,4}, V = {5,6,7,8,9}
    # Edge index: u*5 + (v-5) for u in 0..4, v in 5..9
    
    # Generate constraints: all non-backtracking closed walks of length 4, 6, 8
    # For a walk u1-v1-u2-v2-...-uk-vk-u1, the voltage sum must be nonzero mod m
    # Walk alternates U-V-U-V-...
    # For length 4: u1-v1-u2-v2-u1, need u1!=u2, v1!=v2
    # For length 6: u1-v1-u2-v2-u3-v3-u1, need u's distinct, v's distinct  
    # For length 8: u1-v1-u2-v2-u3-v3-u4-v4-u1, need u's distinct, v's distinct
    
    constraints = []
    
    # Length 4
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for v1 in range(5, 10):
                for v2 in range(5, 10):
                    if v1 == v2: continue
                    e1 = u1 * 5 + (v1 - 5)
                    e2 = u2 * 5 + (v1 - 5)
                    e3 = u2 * 5 + (v2 - 5)
                    e4 = u1 * 5 + (v2 - 5)
                    # Walk: u1->v1 (e1 forward), v1->u2 (e2 backward), u2->v2 (e3 forward), v2->u1 (e4 backward)
                    # Voltage: +e1 - e2 + e3 - e4
                    constraints.append({e1: 1, e2: -1, e3: 1, e4: -1})
    
    # Length 6
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u1 == u3 or u2 == u3: continue
                for v1 in range(5, 10):
                    for v2 in range(5, 10):
                        if v1 == v2: continue
                        for v3 in range(5, 10):
                            if v1 == v3 or v2 == v3: continue
                            e1 = u1 * 5 + (v1 - 5)
                            e2 = u2 * 5 + (v1 - 5)
                            e3 = u2 * 5 + (v2 - 5)
                            e4 = u3 * 5 + (v2 - 5)
                            e5 = u3 * 5 + (v3 - 5)
                            e6 = u1 * 5 + (v3 - 5)
                            constraints.append({e1: 1, e2: -1, e3: 1, e4: -1, e5: 1, e6: -1})
    
    # Length 8
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u1 == u3 or u2 == u3: continue
                for u4 in range(5):
                    if u1 == u4 or u2 == u4 or u3 == u4: continue
                    for v1 in range(5, 10):
                        for v2 in range(5, 10):
                            if v1 == v2: continue
                            for v3 in range(5, 10):
                                if v1 == v3 or v2 == v3: continue
                                for v4 in range(5, 10):
                                    if v1 == v4 or v2 == v4 or v3 == v4: continue
                                    e1 = u1 * 5 + (v1 - 5)
                                    e2 = u2 * 5 + (v1 - 5)
                                    e3 = u2 * 5 + (v2 - 5)
                                    e4 = u3 * 5 + (v2 - 5)
                                    e5 = u3 * 5 + (v3 - 5)
                                    e6 = u4 * 5 + (v3 - 5)
                                    e7 = u4 * 5 + (v4 - 5)
                                    e8 = u1 * 5 + (v4 - 5)
                                    constraints.append({e1: 1, e2: -1, e3: 1, e4: -1, e5: 1, e6: -1, e7: 1, e8: -1})
    
    # Deduplicate
    unique_constraints = set()
    for c in constraints:
        t = tuple(sorted(c.items()))
        unique_constraints.add(t)
    
    n_edges = 25
    n_constraints = len(unique_constraints)
    C = np.zeros((n_constraints, n_edges), dtype=np.int32)
    for i, t in enumerate(unique_constraints):
        for eidx, coeff in t:
            C[i, eidx] = coeff
    
    # Try prime moduli (better for avoiding zero divisors)
    primes = [43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    for m in primes:
        if time.time() > deadline - 15:
            break
            
        n_vertices = 10 * m
        
        # Try structured assignments first: use linear combinations
        # voltages[u*5+v] = a*u + b*v + c mod m for various a,b,c
        found = False
        for a in range(m):
            if time.time() > deadline - 5:
                break
            for b in range(m):
                if time.time() > deadline - 3:
                    break
                for c in range(m):
                    if time.time() > deadline - 2:
                        break
                    voltages = np.zeros(n_edges, dtype=np.int32)
                    for u in range(5):
                        for v in range(5):
                            voltages[u*5+v] = (a*u + b*v + c) % m
                    sums = C @ voltages
                    if not np.any((sums % m) == 0):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if found:
            # Build and verify the graph
            edges = []
            for u in range(5):
                for v in range(5, 10):
                    eidx = u * 5 + (v - 5)
                    vol = int(voltages[eidx])
                    for i in range(m):
                        vid1 = u * m + i
                        vid2 = v * m + ((i + vol) % m)
                        if vid1 < vid2:
                            edges.append((vid1, vid2))
            
            # Verify girth >= 9 using BFS
            if _verify_girth(n_vertices, edges, 9):
                return n_vertices, edges
        
        # If structured didn't work, try random
        if time.time() > deadline - 10:
            continue
            
        max_attempts = 2000
        for _ in range(max_attempts):
            if time.time() > deadline - 2:
                break
            voltages = np.random.randint(0, m, size=n_edges, dtype=np.int32)
            sums = C @ voltages
            if not np.any((sums % m) == 0):
                edges = []
                for u in range(5):
                    for v in range(5, 10):
                        eidx = u * 5 + (v - 5)
                        vol = int(voltages[eidx])
                        for i in range(m):
                            vid1 = u * m + i
                            vid2 = v * m + ((i + vol) % m)
                            if vid1 < vid2:
                                edges.append((vid1, vid2))
                if _verify_girth(n_vertices, edges, 9):
                    return n_vertices, edges
    
    return None

def _verify_girth(n, edges, min_girth):
    """Verify that the graph has no cycle shorter than min_girth using BFS."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 5:
            return False
    
    # BFS from each vertex to find shortest cycle
    # For efficiency, we do BFS up to depth (min_girth-1)//2
    max_depth = (min_girth - 1) // 2  # For girth 9, max_depth = 4
    
    for start in range(n):
        if time.time() > 100:  # Safety
            return False
        # BFS tracking parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        found_short = False
        
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node] and parent[neighbor] != node:
                    # Found a cycle
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        found_short = True
                        break
            if found_short:
                break
        
        if found_short:
            return False
    
    return True

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    # For k=5, g=9: use q=5, d=4 (girth 9 = 2*4+1)
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
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
