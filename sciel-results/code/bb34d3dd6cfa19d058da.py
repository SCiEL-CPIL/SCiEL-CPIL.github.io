import time
import random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    for start in range(n):
        # state: (current_vertex, previous_vertex)
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(k, g):
    """
    Original LUW construction (seed logic).
    Returns (n, edges).
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

def _thue_morse_shifts(m, count):
    """
    Generate voltage shifts based on Thue-Morse sequence.
    """
    shifts = []
    for i in range(count):
        # Thue-Morse bit
        tm = 0
        x = i
        while x:
            tm ^= x & 1
            x >>= 1
        # Map to range [0, m)
        # Split into two halves, linearly spread within each
        half = m // 2
        if tm:
            # Second half
            # Use a different linear step to decorrelate
            val = (half + (i * 31) % half) % m
        else:
            # First half
            val = (i * 31) % half
        shifts.append(val)
    return shifts

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Fallback result (always valid, but large)
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    if k != 3:
        # For other k, stick to LUW as voltage graphs are complex to generalize quickly
        return best_n, best_edges

    # Target: girth g (typically 16)
    # Moore bound for k=3, even g: 2*(2^(g/2)-1)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Base graphs to try (all 3-regular)
    # 1. Petersen Graph (10 vertices, 15 edges, girth 5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # 2. K3,3 (6 vertices, 9 edges, girth 4)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

    # 3. Dodecahedron (20 vertices, 30 edges, girth 5)
    # Vertices 0-9: outer pentagon? No, standard dodecahedron construction.
    # Let's use a known edge list for Dodecahedron.
    # 20 vertices.
    # This might be too complex to hardcode correctly without errors.
    # Let's stick to Petersen and K3,3 but expand search.
    # Or add Cube (8 verts).
    
    # 4. Cube Graph Q3 (8 vertices, 12 edges, girth 4)
    cube_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    base_graphs = [
        (10, petersen_edges),
        (6, k33_edges),
        (8, cube_edges),
    ]

    # Strategy: Try voltage lifts over Z_m.
    # We want n = n_base * m to be small.
    # m must be at least ceil(moore / n_base).
    
    for n_base, base_edges in base_graphs:
        e_count = len(base_edges)
        # Minimum m to satisfy Moore bound
        m_min = max(1, (moore + n_base - 1) // n_base)
        
        # If n_base * m_min is already larger than best_n, skip this base graph
        if n_base * m_min >= best_n:
            continue

        # Try m from m_min to m_min + 100 (increased search depth)
        # We prioritize smaller m.
        
        max_m_search = 100
        for m in range(m_min, m_min + max_m_search):
            if time.time() > deadline - 5:
                break
            
            n = n_base * m
            if n >= best_n:
                continue

            # Try deterministic voltage patterns
            # Pattern 1: Linear
            shifts1 = [(i * m) // e_count for i in range(e_count)]
            
            # Pattern 2: Quadratic
            shifts2 = [(i * i) % m for i in range(e_count)]
            
            # Pattern 3: Thue-Morse
            shifts3 = _thue_morse_shifts(m, e_count)
            
            # Pattern 4: Random (seeded)
            random.seed(42 + m + n_base)
            shifts4 = [random.randint(0, m-1) for _ in range(e_count)]
            
            # Pattern 5: Random (different seed)
            random.seed(123 + m + n_base)
            shifts5 = [random.randint(0, m-1) for _ in range(e_count)]

            for shifts in [shifts1, shifts2, shifts3, shifts4, shifts5]:
                if time.time() > deadline - 2:
                    break
                    
                # Build the lift
                adj = [set() for _ in range(n)]
                edges = []
                valid = True
                for idx, (u, v) in enumerate(base_edges):
                    vol = shifts[idx] % m
                    for i in range(m):
                        a = u * m + i
                        b = v * m + (i + vol) % m
                        if b not in adj[a]:
                            adj[a].add(b)
                            adj[b].add(a)
                            edges.append((a, b))
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check degree
                deg_ok = True
                for i in range(n):
                    if len(adj[i]) != 3:
                        deg_ok = False
                        break
                if not deg_ok:
                    continue
                
                # Check Girth
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n, g):
                    # Found a valid graph!
                    return n, edges

    # If no lift was found within time, return the LUW result
    return best_n, best_edges
