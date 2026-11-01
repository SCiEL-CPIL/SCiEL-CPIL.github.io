import time
import random
import sys
from itertools import product

def construct(k, g):
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # For k=5, g=9: try voltage graph over Icosahedron
    if k == 5 and g == 9:
        result = _voltage_graph_icosahedron(deadline)
        if result is not None:
            return result
        # Fallback to LUW
        return _luw_construct(k, g)
    
    # Fallback to LUW for other parameters
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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

def _voltage_graph_icosahedron(deadline):
    """Construct a 5-regular graph of girth >= 9 using voltage graph over Icosahedron."""
    
    # Icosahedron: 12 vertices, 30 edges, 5-regular, girth 5.
    # Vertices 0..11
    # Edges defined by standard icosahedron coordinates or adjacency list.
    # Adjacency list for Icosahedron:
    # 0: 1, 2, 3, 4, 5
    # 1: 0, 2, 6, 7, 8
    # 2: 0, 1, 3, 9, 10
    # 3: 0, 2, 4, 6, 11
    # 4: 0, 3, 5, 7, 11
    # 5: 0, 4, 1, 8, 9
    # 6: 1, 3, 7, 10, 11
    # 7: 1, 4, 6, 8, 10
    # 8: 1, 5, 7, 9, 10
    # 9: 2, 5, 8, 10, 11
    # 10: 2, 6, 7, 8, 9
    # 11: 3, 4, 6, 9, 10
    
    adj_base = [
        [1, 2, 3, 4, 5],
        [0, 2, 6, 7, 8],
        [0, 1, 3, 9, 10],
        [0, 2, 4, 6, 11],
        [0, 3, 5, 7, 11],
        [0, 4, 1, 8, 9],
        [1, 3, 7, 10, 11],
        [1, 4, 6, 8, 10],
        [1, 5, 7, 9, 10],
        [2, 5, 8, 10, 11],
        [2, 6, 7, 8, 9],
        [3, 4, 6, 9, 10]
    ]
    
    # List of edges (u, v) with u < v
    base_edges = []
    edge_index = {}
    for u in range(12):
        for v in adj_base[u]:
            if u < v:
                idx = len(base_edges)
                base_edges.append((u, v))
                edge_index[(u, v)] = idx
    
    num_edges = len(base_edges) # 30
    
    # Enumerate non-backtracking closed walks of length 5, 6, 7, 8
    # A walk is a sequence v0, v1, ..., vL=v0
    # Non-backtracking: v_{i+1} != v_{i-1}
    
    def get_walk_coefficients(max_len):
        """
        Returns a list of coefficient vectors for all non-backtracking closed walks
        of length L in {5, 6, 7, 8}.
        Each vector has length num_edges.
        coeff[e] = +1 if traversed forward (u->v where u<v), -1 if backward, 0 otherwise.
        """
        coeffs_list = []
        seen = set()
        
        # To avoid duplicates due to rotation, we fix v0 = 0.
        # To avoid reversal duplicates, we can enforce v1 < v_{L-1} or similar, 
        # but since voltages are signed, reversal gives -sum. 
        # If sum != 0, then -sum != 0. So checking one direction is enough?
        # No, if we miss a walk, we might miss a constraint.
        # But if we include all walks starting at 0, we cover all cycles.
        # Actually, a cycle can be started at any vertex.
        # Due to voltage translation invariance? No, voltages are on edges.
        # The constraint is linear.
        # Let's just enumerate all non-backtracking closed walks of length L starting at 0.
        
        for L in range(5, max_len + 1):
            # v0 = 0
            # v1 in adj_base[0]
            for v1 in adj_base[0]:
                # Recursively build walk
                def build(path, prev, cur, remaining):
                    if remaining == 0:
                        # Must return to 0
                        if 0 in adj_base[cur] and 0 != prev:
                            walk = path + [0]
                            # Compute coeffs
                            coeffs = [0] * num_edges
                            for i in range(len(walk) - 1):
                                u, v = walk[i], walk[i+1]
                                if u < v:
                                    idx = edge_index[(u, v)]
                                    coeffs[idx] += 1
                                else:
                                    idx = edge_index[(v, u)]
                                    coeffs[idx] -= 1
                            
                            key = tuple(coeffs)
                            if key not in seen:
                                seen.add(key)
                                coeffs_list.append(coeffs)
                        return
                    
                    for nxt in adj_base[cur]:
                        if nxt == prev:
                            continue
                        if remaining == 1:
                            # Last step must go to 0
                            if nxt == 0 and 0 != prev:
                                walk = path + [0]
                                coeffs = [0] * num_edges
                                for i in range(len(walk) - 1):
                                    u, v = walk[i], walk[i+1]
                                    if u < v:
                                        idx = edge_index[(u, v)]
                                        coeffs[idx] += 1
                                    else:
                                        idx = edge_index[(v, u)]
                                        coeffs[idx] -= 1
                                key = tuple(coeffs)
                                if key not in seen:
                                    seen.add(key)
                                    coeffs_list.append(coeffs)
                        else:
                            build(path + [nxt], cur, nxt, remaining - 1)
                
                build([0, v1], 0, v1, L - 2)
        
        return coeffs_list

    print(f"Enumerating walks...")
    constraints = get_walk_coefficients(8)
    print(f"Number of unique constraints: {len(constraints)}")
    
    # Search for voltage assignments
    random.seed(42)
    best_n = float('inf')
    best_edges = None
    
    # Moore bound n >= 362. n = 12 * m. m >= 31.
    for m in range(31, 100):
        if time.time() > deadline:
            break
        
        # Try random voltage assignments
        for trial in range(200):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(0, m - 1) for _ in range(num_edges)]
            
            valid = True
            for coeffs in constraints:
                total = 0
                for c, v in zip(coeffs, voltages):
                    if c != 0:
                        total += c * v
                if total % m == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build the graph
            n = 12 * m
            edges = []
            adj = [[] for _ in range(n)]
            
            for (u, v), vol in zip(base_edges, voltages):
                for x in range(m):
                    u_lift = u * m + x
                    v_lift = v * m + ((x + vol) % m)
                    if u_lift < v_lift:
                        edges.append((u_lift, v_lift))
                    adj[u_lift].append(v_lift)
                    adj[v_lift].append(u_lift)
            
            # Verify regularity
            is_regular = all(len(adj[i]) == 5 for i in range(n))
            if not is_regular:
                continue
            
            # Verify girth using non-backtracking BFS
            if _check_girth(adj, n, 9):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    if n < 500:
                        return n, edges
                    break  # Found valid for this m, try smaller m
    
    if best_edges is not None:
        return best_n, best_edges
    
    return None

def _check_girth(adj, n, target_girth):
    """Check if graph has girth >= target_girth using non-backtracking BFS."""
    max_depth = (target_girth - 1) // 2
    
    for start in range(n):
        # BFS with parent tracking
        visited = [[-1] * n for _ in range(max_depth + 1)]
        visited[0][start] = -2
        
        queue = [(start, 0, -1)]
        
        while queue:
            v, depth, parent = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for u in adj[v]:
                if u == parent:
                    continue
                if visited[depth + 1][u] != -1:
                    return False
                visited[depth + 1][u] = v
                queue.append((u, depth + 1, v))
    
    return True
