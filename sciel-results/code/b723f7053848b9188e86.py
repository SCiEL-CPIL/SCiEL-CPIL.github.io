import time
import random
import numpy as np
from itertools import permutations

def _luw_construct(k, g):
    """
    Fallback to Lazebnik-Ustimenko-Woldar construction if voltage lift fails or isn't applicable.
    This is kept from the seed to ensure we always return *something* valid if the new method fails.
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

def _voltage_graph_construct(k, g):
    if k != 5:
        return None

    start_time = time.time()
    deadline = start_time + 110.0

    base_n = 6
    base_edges = []
    edge_index = {}
    idx = 0
    for u in range(base_n):
        for v in range(u + 1, base_n):
            base_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1
            
    # Precompute simple cycles of length 3, 4, 5, 6 in K6
    simple_cycles = []
    seen_cycles = set()
    
    # Generate all simple cycles
    for L in range(3, 7):
        for verts in permutations(range(base_n), L):
            # Canonical form: rotate min to front, check reversal
            min_idx = verts.index(min(verts))
            rotated = verts[min_idx:] + verts[:min_idx]
            rev = (rotated[0],) + rotated[:0:-1]
            if rotated < rev:
                edges_seq = []
                for i in range(L):
                    u = rotated[i]
                    v = rotated[(i+1)%L]
                    e = edge_index[(u, v)]
                    edges_seq.append(e)
                edge_set = frozenset(edges_seq)
                if edge_set not in seen_cycles:
                    seen_cycles.add(edge_set)
                    simple_cycles.append(edges_seq)
                    
    M = np.zeros((len(simple_cycles), len(base_edges)), dtype=np.int64)
    for i, seq in enumerate(simple_cycles):
        for e in seq:
            M[i, e] = 1

    for m in range(72, 300):
        if time.time() > deadline:
            break
        n_lift = 6 * m
        
        # Try random voltage assignments
        for attempt in range(10000):
            if time.time() > deadline:
                break
                
            v = np.random.randint(0, m, size=len(base_edges))
            sums = M @ v
            
            if not np.any(sums % m == 0):
                edges = []
                for u, v_idx in base_edges:
                    e_idx = edge_index[(u, v_idx)]
                    w = v[e_idx]
                    for g in range(m):
                        u_idx = u * m + g
                        v_idx_lift = v_idx * m + (g + w) % m
                        edges.append((u_idx, v_idx_lift))
                        
                return (n_lift, edges)
            
    return None

def construct(k, g):
    if k == 5 and g == 9:
        res = _voltage_graph_construct(k, g)
        if res is not None:
            return res
    return _luw_construct(k, g)
