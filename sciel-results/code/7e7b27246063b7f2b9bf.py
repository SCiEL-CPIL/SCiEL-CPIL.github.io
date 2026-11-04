import random
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # LUW fallback construction
    def _luw_graph(k, g):
        q = k
        d = max(3, (g - 2) // 2)
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

    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        for start in range(n):
            visited = set()
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited.add((nb, start))
            while queue:
                curr, prev, dist = queue.popleft()
                if dist >= girth_req:
                    continue
                if curr == start and dist >= 3:
                    return False
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (nb, curr) not in visited:
                        visited.add((nb, curr))
                        queue.append((nb, curr, dist + 1))
        return True

    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    def _get_closed_nb_walks(base_edges, base_adj, n_base, max_len):
        walks = set()
        for start_v in range(n_base):
            queue = []
            for next_v, e_id in base_adj[start_v]:
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                queue.append((next_v, e_id, 1, [(e_id, sign)]))
            while queue:
                curr_v, prev_e, length, path = queue.pop(0)
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    queue.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(base_edges, base_adj, n_base, g)
    n_edges_base = len(base_edges)
    walk_coeffs = []
    for w in closed_walks:
        coeffs = [0] * n_edges_base
        for e_id, sign in w:
            coeffs[e_id] += sign
        if any(c != 0 for c in coeffs):
            walk_coeffs.append(coeffs)
    
    # Remove duplicate constraints (linearly dependent or identical)
    # This reduces the number of checks
    unique_walks = []
    seen = set()
    for c in walk_coeffs:
        # Normalize: make first non-zero element positive
        # Actually, just use tuple as key
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_walks.append(c)
    walk_coeffs = unique_walks

    best_n = float('inf')
    best_edges = None
    
    random.seed(42)
    
    # Moore bound for k=3, g=16 is 512. n = 10*m. m >= 52.
    # Try m from 52 upwards.
    
    for m in range(52, 200):
        if time.time() > deadline - 5.0:
            break
        
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue
            
        # Simulated Annealing for voltage assignment
        def cost(volts):
            c = 0
            for coeffs in walk_coeffs:
                s = 0
                for i, coeff in enumerate(coeffs):
                    if coeff != 0:
                        s += coeff * volts[i]
                if s % m == 0:
                    c += 1
            return c

        def is_valid(volts):
            if cost(volts) > 0:
                return False
            # Check connectivity
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            return g_val == 1

        # Initialize
        volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
        # Ensure gcd is 1 initially? Not strictly necessary for SA, but helps.
        # Force one edge to be 1?
        # Let's just run SA.
        
        current_cost = cost(volts)
        best_volts = volts[:]
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.995
        steps = 50000
        
        for step in range(steps):
            if time.time() > deadline - 2.0:
                break
                
            # Pick random edge
            i = random.randint(0, n_edges_base - 1)
            old_val = volts[i]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            volts[i] = new_val
            
            new_cost = cost(volts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_volts = volts[:]
                    if best_cost == 0:
                        # Check gcd
                        g_val = m
                        for v in best_volts:
                            g_val = math.gcd(g_val, v)
                        if g_val == 1:
                            # Found solution
                            lift_edges = []
                            for idx, (u, v) in enumerate(base_edges):
                                vol = best_volts[idx]
                                for t in range(m):
                                    u_lift = u * m + t
                                    v_lift = v * m + (t + vol) % m
                                    lift_edges.append((u_lift, v_lift))
                            # Verify
                            if _verify_girth(candidate_n, lift_edges, g):
                                best_n = candidate_n
                                best_edges = lift_edges
                                return best_n, best_edges
                            else:
                                # Should not happen if constraints are correct
                                break
            else:
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    volts[i] = old_val
            
            T *= cooling
            if T < 0.01:
                break
        
        # If best_cost == 0, check gcd and build
        if best_cost == 0:
            g_val = m
            for v in best_volts:
                g_val = math.gcd(g_val, v)
            if g_val == 1:
                lift_edges = []
                for idx, (u, v) in enumerate(base_edges):
                    vol = best_volts[idx]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))
                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges
    
    return _luw_graph(k, g)
