"""
Construct a 3-regular graph of girth >= 16 with as few vertices as possible.
Uses voltage graph (fiber product) approach with a small base graph and Z_m group.
"""
import time
import random
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def _moore_bound_3(g):
    """Lower bound on number of vertices for 3-regular graph of girth g."""
    if g % 2 == 0:
        r = g // 2
        # n >= 3 * (2 * 3^(r-1) - 3^(r-2)) for r >= 2
        if r >= 2:
            return 3 * (2 * (3 ** (r - 1)) - (3 ** (r - 2)))
        else:
            return 3
    else:
        r = (g - 1) // 2
        # n >= 3 * (3^r - 3^(r-1) + 1) for r >= 1
        if r >= 1:
            return 3 * ((3 ** r) - (3 ** (r - 1)) + 1)
        else:
            return 3


def _build_base_graphs():
    """Return list of (n_base, edges) for 3-regular base graphs."""
    graphs = []
    # K4: 4 vertices, 6 edges
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    graphs.append((4, k4))
    # 6-vertex: two triangles with matching
    g6 = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (0, 3), (1, 4), (2, 5)]
    graphs.append((6, g6))
    # Cube: 8 vertices, 12 edges
    cube = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
            (4, 5), (4, 6), (5, 7), (6, 7)]
    graphs.append((8, cube))
    # 10-vertex: Petersen graph
    petersen = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]
    graphs.append((10, petersen))
    # 12-vertex: icosahedron
    ico = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
           (1, 2), (1, 6), (1, 7),
           (2, 3), (2, 8),
           (3, 4), (3, 9),
           (4, 5), (4, 10),
           (5, 6), (5, 11),
           (6, 7), (6, 11),
           (7, 8), (7, 11),
           (8, 9), (8, 10),
           (9, 10), (9, 11)]
    graphs.append((12, ico))
    return graphs


def _build_cover_edges(base_edges, labels, m):
    """Build the full edge list for the voltage graph cover."""
    n_total = len(labels)  # placeholder, will be set by caller
    edges = []
    for (u, v), lbl in zip(base_edges, labels):
        for g_val in range(m):
            u_id = u * m + g_val
            v_id = v * m + ((g_val + lbl) % m)
            edges.append((u_id, v_id))
    return edges


def _quick_girth_check(n_total, adj, g, deadline, sample_rate=10):
    """Quick girth check by sampling vertices."""
    for start in range(0, n_total, max(1, n_total // sample_rate)):
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            visited.add((start, nb))
        while queue:
            if time.time() > deadline:
                return False
            prev, node, dist = queue.popleft()
            if dist >= g:
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (node, nb) in visited:
                    if dist + 1 < g:
                        return False
                    continue
                visited.add((node, nb))
                queue.append((node, nb, dist + 1))
    return True


def _full_girth_check(n, edges, g, deadline):
    """Full girth check: no cycle shorter than g."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        if time.time() > deadline:
            return False
        visited = set()
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            visited.add((start, nb))
        while queue:
            if time.time() > deadline:
                return False
            prev, node, dist = queue.popleft()
            if dist >= g:
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (node, nb) in visited:
                    if dist + 1 < g:
                        return False
                    continue
                visited.add((node, nb))
                queue.append((node, nb, dist + 1))
    return True


def _voltage_search(k, g, deadline):
    """Search for a valid voltage graph construction."""
    base_graphs = _build_base_graphs()
    best_n = None
    best_edges = None
    rng = random.Random(12345)
    
    moore_lb = _moore_bound_3(g)
    
    for n_base, base_edges in base_graphs:
        n_edges_base = len(base_edges)
        # Cover has n_base * m vertices
        # Need n_base * m >= moore_lb
        min_m = max(2, (moore_lb + n_base - 1) // n_base)
        # Try a range of m values
        max_m = min(200, 3 * min_m)  # Cap to avoid too large searches
        
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            n_total = n_base * m
            
            # If we already have a better result, skip
            if best_n is not None and n_total >= best_n:
                break
            
            # Try multiple random initializations
            for init in range(5):
                if time.time() > deadline:
                    break
                
                # Random initial labels
                labels = [rng.randint(0, m - 1) for _ in range(n_edges_base)]
                
                # Simulated annealing
                T = 50.0
                cooling = 0.995
                max_iter = 2000
                
                for it in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    # Pick random edge to relabel
                    e_idx = rng.randint(0, n_edges_base - 1)
                    old_label = labels[e_idx]
                    new_label = rng.randint(0, m - 1)
                    if new_label == old_label:
                        continue
                    
                    labels[e_idx] = new_label
                    
                    # Build cover and check girth
                    edges = _build_cover_edges(base_edges, labels, m)
                    adj = defaultdict(list)
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    ok = _quick_girth_check(n_total, adj, g, deadline, sample_rate=20)
                    
                    if ok:
                        # Verify with full check
                        if _full_girth_check(n_total, edges, g, deadline):
                            if best_n is None or n_total < best_n:
                                best_n = n_total
                                best_edges = edges
                            break  # Found valid for this m, move on
                    
                    # Decide whether to keep or revert
                    # Simple approach: always revert unless ok
                    labels[e_idx] = old_label
                    
                    T *= cooling
                
                if best_n is not None and best_n <= n_total:
                    break
        if best_n is not None:
            break
    
    return best_n, best_edges


def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y_coords = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    y_coords[c] = add(x[c], mul(y_coords[a], x[b]))
                else:
                    y_coords[c] = sub(x[c], mul(x[a], y_coords[b]))
            other = (1 - side, tuple(y_coords))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges


def construct(k, g):
    deadline = time.time() + 110.0
    
    # Try voltage graph search first
    best_n, best_edges = _voltage_search(k, g, deadline)
    if best_n is not None:
        return best_n, best_edges
    
    # Fall back to LUW construction
    return _luw_construct(k, g, deadline)

# EVOLVE-BLOCK-END
