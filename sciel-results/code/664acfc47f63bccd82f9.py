import random
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

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

    # Verify girth of a graph using non-backtracking BFS from every vertex
    def _verify_girth(n, edges, girth_req):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Non-backtracking walk check from every vertex
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

    # Petersen graph base
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

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
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

    # Precompute sparse coefficient vectors: list of (edge_idx, coeff) for nonzero entries
    n_edges_base = len(base_edges)
    walk_sparse = []
    for w in closed_walks:
        coeff_dict = {}
        for e_id, sign in w:
            coeff_dict[e_id] = coeff_dict.get(e_id, 0) + sign
        # Remove zeros
        sparse = [(eid, c) for eid, c in coeff_dict.items() if c != 0]
        if sparse:
            walk_sparse.append(sparse)

    best_n = float('inf')
    best_edges = None

    # Search for voltage assignments
    # Moore bound for k=3, g=16 is 512, so n = 10*m >= 512 => m >= 52
    # We want to find the smallest m such that a valid assignment exists.
    # We iterate m from 52 upwards.
    
    random.seed(42)
    
    # We will try m from 52 to 100.
    # For each m, we try a number of random assignments.
    # We allocate more time to smaller m's if we want to be thorough, 
    # but random search is usually effective.
    
    for m in range(52, 101):
        if time.time() > deadline - 5.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue
        
        # Number of tries: 
        # Smaller m might have fewer solutions, so we might need more tries?
        # Or just use a fixed high number.
        max_tries = 20000
        
        found = False
        for _ in range(max_tries):
            if time.time() > deadline - 2.0:
                break
            
            # Generate random voltages
            # Ensure connectivity: gcd(volts, m) = 1
            # We pick random voltages from 0 to m-1.
            # To ensure gcd=1, we can just pick random and check.
            # Or pick one voltage as 1, and others random.
            # If we set one voltage to 1, gcd is always 1.
            # This guarantees connectivity.
            volts = [random.randint(0, m - 1) for _ in range(n_edges_base)]
            # Force one voltage to be 1 to ensure connectivity
            # Actually, if all are 0, gcd is m.
            # If we set volts[0] = 1, then gcd(1, ..., m) = 1.
            volts[0] = 1
            
            # Check all walk conditions using sparse representation
            valid = True
            for sparse in walk_sparse:
                s = 0
                for eid, c in sparse:
                    s += c * volts[eid]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the lift
                lift_edges = []
                for i, (u, v) in enumerate(base_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))
                
                # Verify girth before accepting
                # This is a safety check. The walk condition should be sufficient.
                # But it's good to be sure.
                if _verify_girth(candidate_n, lift_edges, g):
                    best_n = candidate_n
                    best_edges = lift_edges
                    found = True
                    break
        
        if found:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_graph(k, g)
