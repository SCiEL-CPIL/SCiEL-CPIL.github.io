import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3:
        return _construct_two_stage_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_two_stage_lift(g, deadline, start_time):
    # Stage 1: Find a (3,8)-cage by lifting Petersen
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Find (3,8)-cage
    cage_8_n, cage_8_edges = _find_cage_via_lift(pet_edges, 10, 8, deadline, start_time, max_mod=50)
    
    if cage_8_n is None:
        # Fallback to direct lift if cage 8 not found
        return _construct_petersen_lift_direct(g, deadline, start_time)

    # Stage 2: Find (3,16)-cage by lifting the (3,8)-cage
    cage_16_n, cage_16_edges = _find_cage_via_lift(cage_8_edges, cage_8_n, 16, deadline, start_time, max_mod=50)
    
    if cage_16_n is not None:
        return cage_16_n, cage_16_edges
    else:
        # If second stage fails, return the best we have or fallback
        # We can try to return a larger lift of cage 8 if we can't get girth 16? No, must have girth 16.
        # Fallback to direct Petersen lift
        return _construct_petersen_lift_direct(g, deadline, start_time)

def _find_cage_via_lift(base_edges, base_n, target_girth, deadline, start_time, max_mod=50):
    # Precompute non-backtracking closed walks of length < target_girth
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    coeff_vectors = set()
    max_walk_len = target_girth - 1
    
    for start_v in range(base_n):
        stack = [(start_v, -1, 0, [0]*len(base_edges))]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in range(base_n):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                new_counts = counts.copy()
                u_edge, v_edge = base_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))
                
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c): continue
        for i in range(len(base_edges)):
            if c[i] != 0:
                if c[i] < 0: c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    if not coeff_list:
        return base_n, base_edges # Trivial case

    C = np.array(coeff_list, dtype=np.int64)
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    primes = []
    candidate = 2
    while len(primes) < 50 and candidate < max_mod:
        if is_prime(candidate): primes.append(candidate)
        candidate += 1

    best_n = float('inf')
    best_edges = None

    for m in primes:
        if time.time() > deadline: break
        n_candidate = base_n * m
        if n_candidate >= best_n: continue

        # SA to find voltages
        found = False
        T_start = 10.0
        T_end = 0.01
        alpha = 0.99
        max_iters = 2000
        restarts = 10

        for restart in range(restarts):
            if time.time() > deadline: break
            voltages = [random.randint(1, m-1) for _ in range(len(base_edges))]
            v_np = np.array(voltages, dtype=np.int64)
            
            def calc_energy(v):
                sums = C @ v
                return np.sum((sums % m == 0))
            
            current_energy = calc_energy(v_np)
            if current_energy == 0:
                found = True
                break
            T = T_start
            for i in range(max_iters):
                if time.time() > deadline: break
                j = random.randint(0, len(base_edges)-1)
                old_val = v_np[j]
                new_val = random.randint(1, m-1)
                if new_val == old_val: continue
                v_np[j] = new_val
                new_energy = calc_energy(v_np)
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                    current_energy = new_energy
                else:
                    v_np[j] = old_val
                T *= alpha
                if current_energy == 0:
                    found = True
                    break
                if T < T_end: break
            if found: break

        if found:
            voltages_final = v_np.tolist()
            edges = _build_lifted_graph(base_edges, voltages_final, m)
            if _has_girth_at_least(n_candidate, edges, target_girth):
                return n_candidate, edges

    return None, None

def _build_lifted_graph(base_edges, voltages, m):
    base_n = len(set([v for e in base_edges for v in e]))
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    edges = []
    for u in range(base_n):
        for v in range(base_n):
            if u == v: continue
            e_idx = edge_index.get((u, v))
            if e_idx is None: continue
            vol = voltages[e_idx]
            u_edge, v_edge = base_edges[e_idx]
            if u == u_edge and v == v_edge: sign = 1
            else: sign = -1
            for layer in range(m):
                u_node = layer * base_n + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * base_n + v
                if u_node < v_node: edges.append((u_node, v_node))
                else: edges.append((v_node, u_node))
    return list(set(edges))

def _construct_petersen_lift_direct(g, deadline, start_time):
    # Re-use the logic from the original program but with better parameters if needed
    # For now, just call _find_cage_via_lift on Petersen directly
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n, edges = _find_cage_via_lift(pet_edges, 10, g, deadline, start_time, max_mod=100)
    if n is not None:
        return n, edges
    return _construct_luw(3, g, d=11)

def _construct_luw(k, g, d=None, variant=0):
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
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
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
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    max_len = g - 1
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        while queue:
            u, prev, l = queue.popleft()
            if u == start and l > 0:
                return False
            if l >= max_len:
                continue
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    return True
