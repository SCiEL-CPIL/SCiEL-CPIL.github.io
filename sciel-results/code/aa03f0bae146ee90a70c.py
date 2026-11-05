import random
import time
import math

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # Base Graph 1: Petersen
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base_p = 10
    adj_p = [[] for _ in range(n_base_p)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_p[u].append((v, i))
        adj_p[v].append((u, i))
        
    walks_p = _get_nb_walks(petersen_edges, adj_p, n_base_p, g)
    walk_coeffs_p = _get_walk_coeffs(walks_p, len(petersen_edges))
    
    # Try Petersen lifts
    # n = 10 * m
    # Moore bound 510 -> m >= 52
    for m in range(52, 100):
        if time.time() > deadline - 10.0:
            break
        candidate_n = 10 * m
        if candidate_n >= best_n:
            continue
            
        found = False
        # Try random voltages
        tries = 2000
        for _ in range(tries):
            if time.time() > deadline - 5.0:
                break
            volts = [random.randint(1, m-1) for _ in range(len(petersen_edges))]
            # Check connectivity: gcd(volts, m) == 1
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
                
            valid = True
            for coeffs in walk_coeffs_p:
                s = 0
                for i, c in enumerate(coeffs):
                    if c != 0:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            best_n = candidate_n
            best_edges = _construct_lift(petersen_edges, volts, m)
            # If we found a small one, we might still check Heawood for smaller n?
            # Heawood n=14m.
            # If 10m < 14m', we might have found a better one.
            # But we are iterating m for Petersen.
            # We should also check Heawood in parallel or sequentially.
            # Let's break if we found a very good one, e.g., n < 550.
            if best_n < 550:
                break

    # Base Graph 2: Heawood
    heawood_edges = []
    heawood_edges.extend([(0,7), (0,8), (0,9)])
    heawood_edges.extend([(1,7), (1,10), (1,11)])
    heawood_edges.extend([(2,7), (2,12), (2,13)])
    heawood_edges.extend([(3,8), (3,10), (3,12)])
    heawood_edges.extend([(4,8), (4,11), (4,13)])
    heawood_edges.extend([(5,9), (5,10), (5,13)])
    heawood_edges.extend([(6,9), (6,11), (6,12)])
    
    n_base_h = 14
    adj_h = [[] for _ in range(n_base_h)]
    for i, (u, v) in enumerate(heawood_edges):
        adj_h[u].append((v, i))
        adj_h[v].append((u, i))

    walks_h = _get_nb_walks(heawood_edges, adj_h, n_base_h, g)
    walk_coeffs_h = _get_walk_coeffs(walks_h, len(heawood_edges))

    # Try Heawood lifts
    # n = 14 * m
    # Moore bound 510 -> m >= 37
    for m in range(37, 80):
        if time.time() > deadline - 5.0:
            break
        candidate_n = 14 * m
        if candidate_n >= best_n:
            continue
            
        found = False
        tries = 2000
        for _ in range(tries):
            if time.time() > deadline - 2.0:
                break
            volts = [random.randint(1, m-1) for _ in range(len(heawood_edges))]
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
                
            valid = True
            for coeffs in walk_coeffs_h:
                s = 0
                for i, c in enumerate(coeffs):
                    if c != 0:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            if candidate_n < best_n:
                best_n = candidate_n
                best_edges = _construct_lift(heawood_edges, volts, m)
            # If we found the minimum possible m for Heawood, we can break
            if m == 37:
                break

    if best_edges is None:
        return _luw_fallback(k, g)

    return best_n, best_edges

def _get_nb_walks(edges, adj, n, g):
    walks = []
    for start_v in range(n):
        queue = []
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
            
        while queue:
            curr_v, prev_e, length, path = queue.pop(0)
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
                
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    return list(unique_walks)

def _get_walk_coeffs(walks, num_edges):
    coeffs_list = []
    for w in walks:
        coeffs = [0] * num_edges
        for e_id, sign in w:
            coeffs[e_id] += sign
        coeffs_list.append(coeffs)
    return coeffs_list

def _construct_lift(edges, volts, m):
    lift_edges = []
    for i, (u, v) in enumerate(edges):
        vol = volts[i]
        for t in range(m):
            u_lift = u * m + t
            v_lift = v * m + (t + vol) % m
            lift_edges.append((u_lift, v_lift))
    return lift_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
