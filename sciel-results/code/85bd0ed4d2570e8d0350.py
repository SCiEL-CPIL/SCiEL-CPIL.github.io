"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _check_girth_fast(adj, n, g):
    """Check girth >= g using non-backtracking BFS from every vertex."""
    for start in range(n):
        depth = {}
        queue = deque()
        for nb in adj[start]:
            depth[(start, nb)] = 1
            queue.append((start, nb))
        while queue:
            u, v = queue.popleft()
            d = depth[(u, v)]
            if d + 1 >= g:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in depth:
                    if d + 1 + depth[key] < g:
                        return False
                else:
                    depth[key] = d + 1
                    if d + 1 < g:
                        queue.append(key)
    return True


def _edges_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _petersen_edges():
    """Return edges of the Petersen graph (10 vertices, 3-regular, girth 5)."""
    # Outer 5-cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    edges.append((5, 7))
    edges.append((7, 9))
    edges.append((9, 6))
    edges.append((6, 8))
    edges.append((8, 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return edges


def _build_lift(base_edges, base_n, k, group_order, voltage_fn):
    """
    Build a k-regular graph by lifting a base graph via voltage assignment.
    
    base_edges: list of edges in base graph (each edge connects 2 vertices)
    base_n: number of vertices in base graph
    k: desired degree (must equal degree in base graph)
    group_order: order of the abelian group (lift size multiplier)
    voltage_fn: function(edge_index, direction) -> element in Z_{group_order}
                For undirected edge, voltage in reverse direction is negative.
    
    Returns (n, edges) of the lifted graph.
    """
    n = base_n * group_order
    
    # Build adjacency of base graph
    base_adj = [[] for _ in range(base_n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    edges = []
    edge_set = set()
    
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                # Edge u-v in base, with voltage
                # We need to assign voltages consistently
                # Use voltage_fn to get the voltage for this edge
                volt = voltage_fn(u, v)
                for t in range(group_order):
                    # Vertex (u, t) connects to (v, t + volt)
                    a = u * group_order + t
                    b = v * group_order + (t + volt) % group_order
                    if a < b:
                        e = (a, b)
                    else:
                        e = (b, a)
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
    
    return n, edges


def _construct_via_lift(k, g, deadline):
    """
    Try to construct a k-regular graph of girth >= g using voltage lifts.
    Returns (n, edges) or None.
    """
    # For k=4, use a 4-regular base graph
    # Start with small base graphs and try different group orders
    
    # Base graph options for k=4:
    # 1. K5 (5 vertices, 4-regular, girth 3) - too small girth but good for lifting
    # 2. 4-regular graph on 6 vertices: complement of 3 disjoint edges = K_{3,3}
    # 3. 4-regular graph on 8 vertices
    
    # For k=3: Petersen graph (10 vertices, girth 5)
    
    if k == 3:
        base_edges = _petersen_edges()
        base_n = 10
    elif k == 4:
        # K5: 5 vertices, 4-regular
        base_edges = []
        for i in range(5):
            for j in range(i + 1, 5):
                base_edges.append((i, j))
        base_n = 5
    elif k == 5:
        # K6: 6 vertices, 5-regular
        base_edges = []
        for i in range(6):
            for j in range(i + 1, 6):
                base_edges.append((i, j))
        base_n = 6
    else:
        return None
    
    # Try different group orders and voltage assignments
    # The lifted graph has n = base_n * group_order vertices
    # Girth of lift >= girth of base if all non-backtracking closed walks
    # in base of length < g have non-zero total voltage
    
    # For K5 (k=4): base girth is 3, so we need voltages such that
    # no 3-cycle has zero total voltage, no 4-cycle, etc. up to length g-1.
    
    # Strategy: try random voltage assignments and check girth
    best_result = None
    best_n = float('inf')
    
    # Try group orders that give reasonable n
    # For k=4, g=10: Moore bound is 322, known cage is 320
    # K5 * group_order: 5 * m vertices
    # We need 5*m >= 322, so m >= 65. Try m = 65, 66, ...
    
    import math
    moore = k * (k - 1) * (k ** (g // 2 - 1) + 1) + 2 * (k - 1) if g % 2 == 0 else \
             2 * k * (k ** ((g - 3) // 2) + k ** ((g - 1) // 2)) - 2
    
    min_group_order = max(1, math.ceil(moore / base_n))
    
    # Try a range of group orders
    for group_order in range(min_group_order, min_group_order + 50):
        if time.time() > deadline:
            break
        
        n = base_n * group_order
        if n > 500:  # Don't go too large
            break
        
        # Try random voltage assignments
        for trial in range(20):
            if time.time() > deadline:
                break
            
            rng = random.Random(trial * 1000 + group_order)
            
            # Assign random voltages to each edge
            edge_voltages = {}
            for u, v in base_edges:
                volt = rng.randint(0, group_order - 1)
                edge_voltages[(u, v)] = volt
                edge_voltages[(v, u)] = (-volt) % group_order
            
            def voltage_fn(u, v):
                return edge_voltages.get((u, v), 0)
            
            n_lift, edges = _build_lift(base_edges, base_n, k, group_order, voltage_fn)
            
            # Check if it's simple (no multi-edges) and k-regular
            adj = _edges_to_adj(n_lift, edges)
            if any(len(adj[i]) != k for i in range(n_lift)):
                continue
            
            # Check girth
            if _check_girth_fast(adj, n_lift, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break  # Found valid for this group_order, try smaller
    
    if best_result is not None:
        return best_result
    
    return None


def _construct_luw(k, g):
    """L-U-W (Lubotzky-Phillips-Sarnak style) construction for k-regular graphs."""
    # This is a fallback algebraic construction
    try:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
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
        
        n = len(order)
        # Verify
        adj = _edges_to_adj(n, edges)
        if all(len(adj[i]) == k for i in range(n)) and _check_girth_fast(adj, n, g):
            return n, edges
    except:
        pass
    return None


def construct(k, g):
    """
    Build a k-regular graph with girth at least g on as few vertices as possible.
    Uses voltage lift construction over finite abelian groups.
    """
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Try voltage lift construction
    result = _construct_via_lift(k, g, deadline)
    if result is not None:
        return result
    
    # Try L-U-W construction
    result = _construct_luw(k, g)
    if result is not None:
        return result
    
    # Last resort: return a simple regular graph (may not satisfy girth)
    n_fallback = 2 * k
    edges_fallback = []
    # Cycle + chords
    for i in range(n_fallback):
        edges_fallback.append((i, (i + 1) % n_fallback))
    for i in range(k - 1):
        edges_fallback.append((i, (i + n_fallback // 2) % n_fallback))
    # Deduplicate
    edges_fallback = list(set((min(u,v), max(u,v)) for u,v in edges_fallback))
    return n_fallback, edges_fallback
# EVOLVE-BLOCK-END
