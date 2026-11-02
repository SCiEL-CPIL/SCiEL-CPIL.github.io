"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque, defaultdict

# EVOLVE-BLOCK-START

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _girth_check(n, edges, g, deadline):
    """Check if graph has girth at least g using non-backtracking BFS from each vertex."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        if time.time() > deadline:
            return False
        # BFS with state (node, prev_node) to avoid backtracking
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
                # Found a cycle of length >= g, but we need to check if any shorter exists
                # Actually, if we reach dist >= g, it means no cycle < g was found from this start
                break
            for nb in adj[node]:
                if nb == prev:
                    continue
                if (node, nb) in visited:
                    # Found a cycle of length dist + 1
                    if dist + 1 < g:
                        return False
                    continue
                visited.add((node, nb))
                queue.append((node, nb, dist + 1))
    
    return True


def _voltage_graph_search(k, g, deadline):
    """
    Voltage graph (lift) construction with simulated annealing.
    Use a small base graph (e.g., a tree or cycle) with edges labeled by elements of Z_m.
    The cover has |V(B)| * m vertices.
    """
    # For 3-regular graphs, the base graph should be 3-regular too (or we use a different approach)
    # A common approach: use a 3-regular base graph B with m edges, and label each edge with an element of Z_n
    # The cover has |V(B)| * n vertices and is 3-regular
    
    # Start with a small base graph. For 3-regular, the smallest is K4 (4 vertices, 6 edges)
    # or a cycle with chords. Let's try different base graphs.
    
    # Base graph options: K4 (4 vertices, 6 edges), or larger 3-regular graphs
    # For K4: vertices 0-3, edges: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    
    base_graphs = []
    # K4
    base_graphs.append((4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]))
    # 6-vertex 3-regular: two triangles connected by matching
    base_graphs.append((6, [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(0,3),(1,4),(2,5)]))
    # 8-vertex 3-regular: cube graph
    base_graphs.append((8, [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]))
    
    best_n = None
    best_edges = None
    
    for n_base, base_edges in base_graphs:
        n_edges_base = len(base_edges)
        
        # The cover will have n_base * m vertices where m is the group order
        # We want to find the smallest m such that the cover has girth >= g
        
        # Moore bound for 3-regular graph with girth g:
        # For even g = 2r: n >= 3 * (2 * 3^(r-1) - 3^(r-2)) 
        # For odd g = 2r+1: n >= 3 * (3^r - 3^(r-1) + 1)
        # Simplified: n >= 3 * (3^(g//2 - 1) + ...)
        
        # For girth 16: Moore bound is around 512
        # So we need n_base * m >= 512, so m >= 512 / n_base
        
        # Try different values of m (group order Z_m)
        # Start from a reasonable lower bound
        
        # For girth g, the girth of the cover is related to the shortest non-backtracking 
        # closed walk in the base graph whose voltage product is identity.
        
        # Simulated annealing to optimize edge labels
        rng = random.Random(42)
        
        max_m = 50  # Try group orders up to 50
        min_m = max(2, (512 + n_base - 1) // n_base)  # Ensure n_base * m >= 512
        
        for m in range(min_m, max_m + 1):
            if time.time() > deadline:
                break
            
            n_total = n_base * m
            
            # Initialize edge labels randomly
            labels = [rng.randint(0, m - 1) for _ in range(n_edges_base)]
            
            # Simulated annealing
            T = 10.0
            cooling = 0.99
            max_iter = 1000
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to relabel
                e_idx = rng.randint(0, n_edges_base - 1)
                old_label = labels[e_idx]
                new_label = rng.randint(0, m - 1)
                if new_label == old_label:
                    continue
                
                labels[e_idx] = new_label
                
                # Check girth of the cover
                # Build the cover graph
                # Vertex (v, g) where v in V(B), g in Z_m
                # Edge between (u, g) and (v, g + label(e)) for each edge (u,v) in B with label e
                
                # For girth check, we need to verify no cycle shorter than g
                # This is expensive, so we do a quick check
                
                # Quick girth check using BFS in the cover
                girth_ok = True
                adj_cover = defaultdict(list)
                for (u, v), lbl in zip(base_edges, labels):
                    for g_val in range(m):
                        u_id = u * m + g_val
                        v_id = v * m + ((g_val + lbl) % m)
                        adj_cover[u_id].append(v_id)
                        adj_cover[v_id].append(u_id)
                
                # Check girth from a few sample vertices
                for start in range(0, n_total, max(1, n_total // 10)):
                    visited = set()
                    queue = deque()
                    for nb in adj_cover[start]:
                        queue.append((start, nb, 1))
                        visited.add((start, nb))
                    
                    while queue:
                        if time.time() > deadline:
                            girth_ok = False
                            break
                        prev, node, dist = queue.popleft()
                        if dist >= g:
                            break
                        for nb in adj_cover[node]:
                            if nb == prev:
                                continue
                            if (node, nb) in visited:
                                if dist + 1 < g:
                                    girth_ok = False
                                    break
                                continue
                            visited.add((node, nb))
                            queue.append((node, nb, dist + 1))
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # Found a valid configuration
                    # Build the full edge list
                    edges = []
                    for (u, v), lbl in zip(base_edges, labels):
                        for g_val in range(m):
                            u_id = u * m + g_val
                            v_id = v * m + ((g_val + lbl) % m)
                            edges.append((u_id, v_id))
                    
                    if best_n is None or n_total < best_n:
                        best_n = n_total
                        best_edges = edges
                    break  # Found valid config for this m, try smaller m? No, m is increasing
                
                # If not valid, decide whether to keep the change
                # (Simplified: always revert for now, can improve with proper SA)
                labels[e_idx] = old_label
            
            if best_n is not None and best_n <= n_total:
                break  # Already found something at most this size
    
    return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110.0
    
    # Try voltage graph search first
    result = _voltage_graph_search(k, g, deadline)
    if result[0] is not None:
        n, edges = result
        # Verify the result
        if _girth_check(n, edges, g, deadline):
            return n, edges
    
    # Fall back to LUW construction
    return _luw_construct(k, g, deadline)

# EVOLVE-BLOCK-END
