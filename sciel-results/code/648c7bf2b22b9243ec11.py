import time
import random
from collections import defaultdict
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary: Petersen graph voltage lift for k=3.
    Fallback: LUW with small d for compact graph.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    best_n = float('inf')
    best_edges = None
    
    # Primary: Petersen lift
    try:
        n, edges = _construct_petersen_lift(k, g, deadline)
        if edges is not None:
            if _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
    except Exception:
        pass
    
    # Fallback: LUW with small d
    if best_edges is None:
        try:
            for d_try in [7, 8, 9, 10, 11]:
                if time.time() > deadline - 10:
                    break
                for rv in range(3):
                    if time.time() > deadline - 10:
                        break
                    n, edges = _construct_luw(k, g, d=d_try, rule_variant=rv)
                    if _has_girth_at_least(n, edges, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
        except Exception:
            pass
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Final fallback: voltage graph construction
    return _construct_voltage(k, g, deadline)

def _construct_petersen_lift(k, g, deadline):
    """
    Construct a 3-regular graph with girth >= g by lifting the Petersen graph
    using voltage assignments over Z_m.
    
    The Petersen graph has 10 vertices and 15 edges, is 3-regular.
    A voltage lift over Z_m gives a 3-regular graph on 10*m vertices.
    
    We search for voltage assignments such that no non-backtracking closed
    walk in the base graph has total voltage 0 mod m.
    """
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    # Petersen graph edges (10 vertices, 15 edges, 3-regular)
    # Standard Petersen graph: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5, spokes i-(i+5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    n_petersen = 10
    n_edges_petersen = 15
    
    # Build adjacency for Petersen
    adj = [[] for _ in range(n_petersen)]
    for i, (u, v) in enumerate(petersen_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Find all non-backtracking closed walks of length < g in the Petersen graph
    # For each, compute the coefficient vector (how many times each edge is traversed)
    # We need: for all such walks, sum(coeff[i] * voltage[i]) != 0 (mod m)
    
    max_walk_len = g - 1
    coeff_vectors = set()
    
    for start_v in range(n_petersen):
        # State: (current_vertex, previous_edge, length, edge_counts)
        # Non-backtracking: cannot immediately go back along the same edge
        stack = [(start_v, -1, 0, [0] * n_edges_petersen)]
        
        while stack:
            v, prev_e, l, counts = stack.pop()
            
            if l > 0 and v == start_v:
                coeff_vectors.add(tuple(counts))
            
            if l >= max_walk_len:
                continue
            
            for next_v, e_idx in adj[v]:
                if e_idx == prev_e:
                    continue  # non-backtracking
                
                new_counts = counts.copy()
                new_counts[e_idx] += 1
                stack.append((next_v, e_idx, l + 1, new_counts))
    
    # Remove the zero vector (empty walk)
    zero_vec = tuple([0] * n_edges_petersen)
    if zero_vec in coeff_vectors:
        coeff_vectors.remove(zero_vec)
    
    coeff_list = list(coeff_vectors)
    
    if not coeff_list:
        # No constraints, just return the Petersen graph itself (but it has small girth)
        # Actually if no non-backtracking closed walks < g, the Petersen itself works
        # But Petersen has girth 5, so this won't happen for g > 5
        pass
    
    best_n = float('inf')
    best_edges = None
    
    # Try various moduli m
    # The lift has 10*m vertices. We want small m.
    # Try m from small primes upward
    candidate_moduli = []
    for m in range(2, 500):
        # Check if m is suitable (ideally prime for better mixing)
        candidate_moduli.append(m)
    
    random.seed(42)
    
    for m in candidate_moduli:
        if time.time() > deadline - 5:
            break
        
        if 10 * m >= best_n:
            break
        
        # Try random voltage assignments
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(n_edges_petersen)]
            
            valid = True
            for c in coeff_list:
                s = sum(c[i] * voltages[i] for i in range(n_edges_petersen))
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Build the lifted graph
                n = 10 * m
                edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    vol = voltages[i]
                    for x in range(m):
                        u_node = x * 10 + u
                        v_node = ((x + vol) % m) * 10 + v
                        if u_node < v_node:
                            edges.append((u_node, v_node))
                        else:
                            edges.append((v_node, u_node))
                
                edges = list(set(edges))
                
                # Verify girth
                if _has_girth_at_least(n, edges, g):
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    return None, None

def _construct_luw(k, g, d=None, rule_variant=0):
    """
    LUW construction with specified depth d and rule variant.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule table based on variant
    if rule_variant == 0:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif rule_variant == 1:
        rule = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 1), (c0 - 2, 1), (c0, 1), (1, c0 + 2)][r]
    elif rule_variant == 2:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 3
            c0 = c - (c - 5) % 3
            rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
    else:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 5
            c0 = c - (c - 5) % 5
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (1, c0 + 2)][r]
    
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _construct_voltage(k, g, deadline):
    """
    Voltage graph construction over K4 for k=3.
    """
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    edge_map = {
        (0, 1): 0, (1, 0): 0,
        (0, 2): 1, (2, 0): 1,
        (0, 3): 2, (3, 0): 2,
        (1, 2): 3, (2, 1): 3,
        (1, 3): 4, (3, 1): 4,
        (2, 3): 5, (3, 2): 5
    }
    
    target_girth = g
    max_walk_len = target_girth - 1
    
    # Precompute coefficient vectors for all reduced closed walks of length < g
    coeff_vectors = set()
    
    for start_v in range(4):
        stack = [(start_v, -1, 0, [0]*6)]
        
        while stack:
            v, prev_e, l, counts = stack.pop()
            
            if l > 0 and v == start_v:
                coeff_vectors.add(tuple(counts))
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(4):
                if next_v == v:
                    continue
                e_idx = edge_map[(v, next_v)]
                if e_idx == prev_e:
                    continue
                
                new_counts = counts.copy()
                new_counts[e_idx] += 1
                stack.append((next_v, e_idx, l + 1, new_counts))
    
    if (0,0,0,0,0,0) in coeff_vectors:
        coeff_vectors.remove((0,0,0,0,0,0))
        
    coeff_list = list(coeff_vectors)
    
    best_n = float('inf')
    best_edges = None
    
    primes = [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    
    for m in primes:
        if time.time() > deadline:
            break
            
        max_attempts = 30
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(1, m-1) for _ in range(6)]
            
            valid = True
            for c in coeff_list:
                s = sum(c[i] * voltages[i] for i in range(6))
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                n = 4 * m
                edges = []
                for u in range(4):
                    for w in range(4):
                        if u == w:
                            continue
                        e_idx = edge_map[(u, w)]
                        vol = voltages[e_idx]
                        for v in range(m):
                            u_node = v * 4 + u
                            w_node = ((v + vol) % m) * 4 + w
                            if u_node < w_node:
                                edges.append((u_node, w_node))
                            else:
                                edges.append((w_node, u_node))
                
                edges = list(set(edges))
                
                girth = _compute_girth(n, edges)
                
                if girth >= g:
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    return _construct_luw(k, g, d=7)

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each starting vertex, do BFS with parent tracking
    # This is O(n * (n + m)) which for n up to ~1000 is manageable
    
    for start in range(n):
        # BFS from start
        parent = [-1] * n
        level = [-1] * n
        level[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if level[u] > (g - 1) // 2:
                continue
            
            for v in adj[u]:
                if v == parent[u]:
                    continue  # non-backtracking
                if level[v] == -1:
                    level[v] = level[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = level[u] + level[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _compute_girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    girth = float('inf')
    for i in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [i]
        dist[i] = 0
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth
