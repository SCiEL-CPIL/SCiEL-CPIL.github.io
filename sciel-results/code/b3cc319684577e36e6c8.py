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

    # 3. Cube Graph Q3 (8 vertices, 12 edges, girth 4)
    cube_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # 4. Prism Graph (6 vertices, 9 edges, girth 3)
    prism_edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5)
    ]

    base_graphs = [
        (10, petersen_edges),
        (6, k33_edges),
        (8, cube_edges),
        (6, prism_edges),
    ]

    # Strategy: Try voltage lifts over Z_m.
    # We want n = n_base * m to be small.
    # m must be at least ceil(moore / n_base).
    
    for n_base, base_edges in base_graphs:
        e_count = len(base_edges)
        # Minimum m to satisfy Moore bound
        m_min = max(1, (moore + n_base - 1) // n_base)
        
        # We'll try a range of m values starting from m_min.
        # For each m, we try several deterministic voltage assignments.
        
        # To save time, we limit the number of m values we try per base graph
        # and the number of voltage patterns per m.
        
        # If n_base * m_min is already larger than best_n, skip this base graph
        if n_base * m_min >= best_n:
            continue

        # Try m from m_min to m_min + 20 (or until time runs out)
        # We prioritize smaller m.
        
        max_m_search = 50
        for m in range(m_min, m_min + max_m_search):
            if time.time() > deadline - 5:
                break
            
            n = n_base * m
            if n >= best_n:
                continue

            # Try deterministic voltage patterns
            # Pattern 1: Quadratic Residues (spread)
            # We need e_count distinct (or at least well-spread) shifts in Z_m.
            
            # Helper to generate shifts
            def get_shifts_pattern1(m, count):
                # Use quadratic residues if possible, else linear
                if m <= 1:
                    return [0] * count
                # Simple heuristic: spread out
                # For cubic graphs, we can try (i * m // count) or similar
                # Let's try a few specific structures known to work for cages
                # e.g. for Petersen, specific voltage assignments are known.
                
                # Generic: 
                # 1. Linear: i * (m // count) 
                # 2. Golden Ratio
                # 3. Random (seeded)
                
                shifts = []
                for i in range(count):
                    # Use a mix to avoid small cycles
                    # Shift = (i * 31 + 7) % m ?
                    # Let's try quadratic residues properly
                    pass
                
                # Better: Try a few specific deterministic sets
                sets = []
                
                # Set 1: Linear spread
                s1 = []
                for i in range(count):
                    s1.append((i * m) // count)
                sets.append(s1)
                
                # Set 2: Quadratic Residues (if m is prime-ish or just mod m)
                s2 = []
                for i in range(count):
                    # i^2 mod m
                    s2.append((i * i) % m)
                sets.append(s2)
                
                # Set 3: Random (fixed seed for reproducibility in this run)
                random.seed(42 + m + n_base)
                s3 = [random.randint(0, m-1) for _ in range(count)]
                sets.append(s3)
                
                # Set 4: Another random
                random.seed(123 + m + n_base)
                s4 = [random.randint(0, m-1) for _ in range(count)]
                sets.append(s4)
                
                return sets

            shifts_sets = get_shifts_pattern1(m, e_count)
            
            for shifts in shifts_sets:
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
                            # Multi-edge or loop?
                            # If b == a, it's a loop. If already in adj, it's a multi-edge.
                            # Voltage graphs over simple base graphs with distinct voltages 
                            # in a fiber usually don't create loops/multiedges unless 
                            # vol=0 and u=v (loop in base) or specific symmetries.
                            # Since base is simple, u!=v. So a != b.
                            # If b in adj[a], it means we added this edge twice?
                            # No, (u,v) and (v,u) are distinct edges in base? 
                            # My base_edges list contains each undirected edge once.
                            # So we add a-b once.
                            # However, if the graph is not simple in the lift, it's invalid.
                            # Actually, if b is already in adj[a], it means another edge (u', v') 
                            # with voltage vol' created the same pair.
                            # This happens if (u,v) and (u',v') are "parallel" in the lift.
                            # For a simple base graph, this shouldn't happen for distinct edges 
                            # unless the voltages align perfectly in a way that maps distinct 
                            # base edges to the same fiber edge.
                            # Wait, if base has (u,v) and (u,v) again? No, simple.
                            # So this check is just for safety.
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
