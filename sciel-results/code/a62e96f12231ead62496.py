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
    
    for v in range(n):
        visited = {}
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
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
    """
    if k != 4 or g != 10:
        return None

    n_base = 10
    group_size = 32
    
    # Base graph edges: (i, 5+j) for i != j
    base_edges = []
    edge_indices = {}
    for i in range(5):
        for j in range(5):
            if i != j:
                edge_id = len(base_edges)
                base_edges.append((i, 5 + j))
                edge_indices[(i, 5 + j)] = edge_id
                edge_indices[(5 + j, i)] = edge_id
    
    # Pre-compute cycles in the base graph
    # 4-cycles
    cycles_4 = []
    for i in range(5):
        for k in range(5):
            if i == k: continue
            rem = [x for x in range(5) if x != i and x != k]
            for a in range(3):
                for b in range(3):
                    if a == b: continue
                    j = rem[a]
                    l = rem[b]
                    e1 = edge_indices.get((i, 5+j))
                    e2 = edge_indices.get((k, 5+j))
                    e3 = edge_indices.get((k, 5+l))
                    e4 = edge_indices.get((i, 5+l))
                    if e1 is None or e2 is None or e3 is None or e4 is None: continue
                    cycles_4.append((e1, e2, e3, e4, [1, -1, 1, -1]))
    
    # 6-cycles
    cycles_6 = []
    # u_i -> v_j -> u_k -> v_l -> u_m -> v_n -> u_i
    # i, k, m are u-nodes. j, l, n are v-nodes.
    # Edges: (i, 5+j), (k, 5+j), (k, 5+l), (m, 5+l), (m, 5+n), (i, 5+n)
    # Constraints: i!=j, k!=j, k!=l, m!=l, m!=n, i!=n.
    # Also, to be a simple cycle, the vertices must be distinct in the base graph?
    # No, the base graph is bipartite, so u and v are distinct.
    # We just need to ensure the cycle is valid.
    for i in range(5):
        for k in range(5):
            for m in range(5):
                if i == k or k == m or m == i: continue
                # j must be != i, k
                for j in range(5):
                    if j == i or j == k: continue
                    # l must be != k, m
                    for l in range(5):
                        if l == k or l == m: continue
                        # n must be != m, i
                        for n in range(5):
                            if n == m or n == i: continue
                            # Check if edges exist
                            if (i, 5+j) not in edge_indices: continue
                            if (k, 5+j) not in edge_indices: continue
                            if (k, 5+l) not in edge_indices: continue
                            if (m, 5+l) not in edge_indices: continue
                            if (m, 5+n) not in edge_indices: continue
                            if (i, 5+n) not in edge_indices: continue
                            
                            e1 = edge_indices[(i, 5+j)]
                            e2 = edge_indices[(k, 5+j)]
                            e3 = edge_indices[(k, 5+l)]
                            e4 = edge_indices[(m, 5+l)]
                            e5 = edge_indices[(m, 5+n)]
                            e6 = edge_indices[(i, 5+n)]
                            cycles_6.append((e1, e2, e3, e4, e5, e6, [1, -1, 1, -1, 1, -1]))
    
    # 8-cycles are too many to precompute efficiently in Python within time limits?
    # We might skip 8-cycles in the penalty and rely on verification.
    
    rng = random.Random(42)
    best_voltages = None
    best_penalty = float('inf')
    
    voltages = [rng.randint(0, group_size - 1) for _ in range(len(base_edges))]
    
    def calc_penalty(volts):
        penalty = 0
        for cycle in cycles_4:
            s = sum(sign * volts[e] for e, sign in zip(cycle[:4], cycle[4:]))
            if s % group_size == 0:
                penalty += 1
        for cycle in cycles_6:
            s = sum(sign * volts[e] for e, sign in zip(cycle[:6], cycle[6:]))
            if s % group_size == 0:
                penalty += 1
        return penalty

    current_penalty = calc_penalty(voltages)
    
    start_time = time.time()
    while time.time() < deadline - 10:
        e_idx = rng.randint(0, len(base_edges) - 1)
        old_val = voltages[e_idx]
        best_new_val = old_val
        best_local_penalty = current_penalty
        
        for new_val in range(group_size):
            if new_val == old_val: continue
            voltages[e_idx] = new_val
            p = calc_penalty(voltages)
            if p < best_local_penalty:
                best_local_penalty = p
                best_new_val = new_val
        
        voltages[e_idx] = best_new_val
        current_penalty = best_local_penalty
        
        if current_penalty == 0:
            best_voltages = voltages[:]
            break
        
        if rng.random() < 0.01:
            voltages = [rng.randint(0, group_size - 1) for _ in range(len(base_edges))]
            current_penalty = calc_penalty(voltages)

    if best_voltages is None:
        return None

    edges = []
    for u, v in base_edges:
        w = best_voltages[edge_indices[(u, v)]]
        for x in range(group_size):
            u_id = u * group_size + x
            v_id = v * group_size + (x + w) % group_size
            edges.append((u_id, v_id))
            
    if _verify_girth(320, edges, 4, 10):
        return 320, edges
        
    return None

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    deadline = time.time() + 140  # Safety margin
    
    if k == 4 and g == 10:
        result = _construct_cage_voltage(k, g, deadline)
        if result is not None:
            return result
            
    n, edges = _construct_luw(k, g)
    return n, edges

# EVOLVE-BLOCK-END
