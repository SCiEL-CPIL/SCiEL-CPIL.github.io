"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound_lower(k, g):
    """Return a safe lower bound on n(k,g)."""
    if g % 2 == 0:
        return 2 * sum((k-1)**i for i in range(g//2))
    else:
        return 1 + k * sum((k-1)**i for i in range((g-1)//2))

def _verify_girth(n, edges, k, g):
    """Verify the graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # For girth verification, use non-backtracking walks
    # We can optimize by early termination
    for v in range(n):
        visited = {}
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    return True

def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
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

def _construct_cage_voltage(k, g, deadline):
    """
    Attempt to construct the (4,10)-cage using voltage graph method.
    Base graph: Bipartite double cover of K5 (10 vertices, 4-regular).
    Voltage group: Z_32.
    Lift has 320 vertices.
    """
    if k != 4 or g != 10:
        return None

    n_base = 10
    n_lift = 320
    group_size = 32
    
    # Base graph edges: (i, 5+j) for i != j, i in 0..4, j in 0..4
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    # Each vertex in base has degree 4.
    # We assign a voltage v_e in Z_32 to each base edge e.
    # The lift vertex (u, x) is connected to (v, x + v_e) if (u, v) is an edge with voltage v_e.
    
    # Try deterministic algebraic constructions first
    # These are more likely to yield high girth
    
    # Strategy 1: Use a known good voltage assignment pattern
    # The (4,10)-cage can be constructed via specific voltage assignments
    # Let's try several structured patterns
    
    def try_voltage_assignment(voltages):
        """Construct lift and verify girth."""
        edges = []
        for u, v in base_edges:
            w = voltages[(u, v)]
            for x in range(group_size):
                u_id = u * group_size + x
                v_id = v * group_size + (x + w) % group_size
                edges.append((u_id, v_id))
        
        if _verify_girth(n_lift, edges, k, g):
            return n_lift, edges
        return None
    
    # Pattern 1: Linear voltage assignment based on edge index
    for seed_offset in range(50):
        if time.time() > deadline - 30:
            break
        
        voltages = {}
        for idx, (u, v) in enumerate(base_edges):
            # Use a structured pattern
            voltages[(u, v)] = (idx * 7 + seed_offset * 3) % group_size
        
        result = try_voltage_assignment(voltages)
        if result is not None:
            return result
    
    # Pattern 2: Based on vertex positions
    for seed_offset in range(50):
        if time.time() > deadline - 30:
            break
        
        voltages = {}
        for u, v in base_edges:
            # u is 0-4, v is 5-9 (so v-5 is 0-4)
            i, j = u, v - 5
            voltages[(u, v)] = (i * 5 + j * 3 + seed_offset) % group_size
        
        result = try_voltage_assignment(voltages)
        if result is not None:
            return result
    
    # Pattern 3: Use powers of 2
    for seed_offset in range(30):
        if time.time() > deadline - 30:
            break
        
        voltages = {}
        for idx, (u, v) in enumerate(base_edges):
            voltages[(u, v)] = (2**idx + seed_offset) % group_size
        
        result = try_voltage_assignment(voltages)
        if result is not None:
            return result
    
    # Now try random search with better heuristics
    rng = random.Random(42)
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
        
        # Assign random voltages with some structure
        voltages = {}
        for u, v in base_edges:
            i, j = u, v - 5
            # Mix of structured and random
            base_val = (i * 5 + j * 3) % group_size
            voltages[(u, v)] = (base_val + rng.randint(0, group_size - 1)) % group_size
        
        result = try_voltage_assignment(voltages)
        if result is not None:
            return result
    
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try the specific cage construction for (4,10)
    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result
            
    # Fallback to LUW construction
    n, edges = _construct_luw(k, g)
    
    return n, edges

# EVOLVE-BLOCK-END
