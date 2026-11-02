"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    # Use a voltage graph (lift) over a small base graph with abelian voltage
    # assignments from a cyclic group Z_m.
    #
    # Base graph: K_6 (complete graph on 6 vertices) is 5-regular.
    # A voltage graph over K_6 with voltages in Z_m gives a 5-regular graph
    # on 6*m vertices.
    
    if k != 5 or g != 9:
        # Fallback to L-U-W construction for other parameters
        return _luw_construct(k, g)
    
    deadline = time.time() + 115  # 115 seconds max (safety margin)
    
    # Calculate Moore Bound to determine minimum m
    # For odd girth g=2R+1, N >= 1 + k * sum_{i=0}^{R-1} (k-1)^i
    R = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(R))
    
    base_n = 6
    min_m = (moore_bound + base_n - 1) // base_n  # Ceiling division
    
    # Prepare base graph structure
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_base_edges = len(base_edges)  # 15 edges
    
    # Generate non-backtracking closed walks of length 3..8 in K_6
    def gen_nonbacktracking_closed_walks(length):
        """Generate non-backtracking closed walks of given length in K_6, starting at 0."""
        if length < 3:
            return
        
        def walk_gen(pos, cur, prv, path):
            if pos == length - 1:
                # Next must be 0
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            
            for nv in range(base_n):
                if nv == cur:
                    continue
                if nv == prv:
                    continue
                path.append(nv)
                yield from walk_gen(pos + 1, nv, cur, path)
                path.pop()
        
        for v1 in range(1, base_n):
            yield from walk_gen(1, v1, 0, [0, v1])
    
    # Collect constraints
    constraints = []
    for L in range(3, 9):
        walks = list(gen_nonbacktracking_closed_walks(L))
        for walk in walks:
            coeff_dict = {}
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                if u == v:
                    continue
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                coeff_dict[eidx] = coeff_dict.get(eidx, 0) + sign
            terms = tuple(sorted((eidx, c) for eidx, c in coeff_dict.items() if c != 0))
            if terms:
                constraints.append(terms)
    
    constraints = list(set(constraints))
    n_constraints = len(constraints)
    
    # Build constraint matrix
    constraint_matrix = np.zeros((n_constraints, n_base_edges), dtype=np.int32)
    for i, terms in enumerate(constraints):
        for eidx, coeff in terms:
            constraint_matrix[i, eidx] = coeff
    
    best_result = None
    best_n = float('inf')
    
    # Search for m starting from the Moore bound limit
    # We want the smallest n, so smallest m.
    for m in range(min_m, 2000):
        if time.time() > deadline - 20:
            break
        
        n_vertices = base_n * m
        if n_vertices >= best_n:
            continue
        
        # Vectorized random search
        # Increase batch size to improve probability of finding a valid assignment
        # especially for smaller m where the valid region is smaller.
        batch_size = 100000
        found = False
        
        while not found and time.time() < deadline - 5:
            voltages = np.random.randint(0, m, size=(batch_size, n_base_edges), dtype=np.int32)
            sums = voltages @ constraint_matrix.T
            valid = ~np.any((sums % m) == 0, axis=1)
            
            if np.any(valid):
                valid_idx = np.where(valid)[0][0]
                voltages = voltages[valid_idx]
                found = True
                break
        
        if found:
            # Build the lifted graph
            graph_edges = []
            for u in range(base_n):
                for v in range(base_n):
                    if u == v:
                        continue
                    eu, ev = min(u, v), max(u, v)
                    eidx = edge_index[(eu, ev)]
                    vol = int(voltages[eidx])
                    for i in range(m):
                        vid1 = u * m + i
                        if u < v:
                            j = (i + vol) % m
                        else:
                            j = (i - vol) % m
                        vid2 = v * m + j
                        if vid1 < vid2:
                            graph_edges.append((vid1, vid2))
            
            best_result = (n_vertices, graph_edges)
            best_n = n_vertices
            break  # Found valid for this m, which is the smallest possible n for this m.
            # Since we iterate m increasing, the first found is the global minimum n for this base.
    
    if best_result is not None:
        return best_result
    
    # Fallback: use L-U-W construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
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

# EVOLVE-BLOCK-END
