import random
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Fallback: LUW Construction
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

    # Heawood Graph Definition (14 vertices, 21 edges, girth 7)
    # Bipartite (0-6, 7-13)
    # Cycle 0-1-2-3-4-5-0 (length 6? No, Heawood has girth 7)
    # Let's use the standard Fano plane incidence graph representation.
    # Vertices: 0..12 (lines), 13 (point? No, 7 points, 7 lines)
    # Standard Heawood:
    # Points: 0, 1, 2, 3, 4, 5, 6
    # Lines: 7, 8, 9, 10, 11, 12, 13
    # Incidences:
    # 7: 0-1-2
    # 8: 0-3-4
    # 9: 0-5-6
    # 10: 1-3-5
    # 11: 1-4-6
    # 12: 2-3-6
    # 13: 2-4-5
    
    heawood_edges = [
        (0, 7), (1, 7), (2, 7),
        (0, 8), (3, 8), (4, 8),
        (0, 9), (5, 9), (6, 9),
        (1, 10), (3, 10), (5, 10),
        (1, 11), (4, 11), (6, 11),
        (2, 12), (3, 12), (6, 12),
        (2, 13), (4, 13), (5, 13)
    ]
    n_heawood = 14
    
    # Petersen Graph Definition
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_petersen = 10

    def _get_walk_coeffs(base_edges, n_base, girth_req):
        """
        Generate all closed non-backtracking walks of length L where min_len <= L < girth_req.
        Returns a list of coefficient vectors (sparse) for each unique walk.
        """
        adj = [[] for _ in range(n_base)]
        for i, (u, v) in enumerate(base_edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        
        walks = set()
        for start_v in range(n_base):
            # BFS state: (current_vertex, prev_edge_id, length, path_tuple)
            # We store path as tuple of (edge_id, sign)
            queue = deque()
            for next_v, e_id in adj[start_v]:
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                queue.append((next_v, e_id, 1, ((e_id, sign),)))
            
            while queue:
                curr_v, prev_e, length, path = queue.popleft()
                if length >= girth_req:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(path)
                for next_v, e_id in adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + ((e_id, sign),)
                    queue.append((next_v, e_id, length + 1, new_path))
        
        # Convert to sparse coefficient vectors
        n_edges = len(base_edges)
        coeffs_list = []
        for w in walks:
            coeffs = [0] * n_edges
            for e_id, sign in w:
                coeffs[e_id] += sign
            # Filter zero coefficients
            nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
            if nonzero:
                coeffs_list.append(nonzero)
        
        return coeffs_list

    def _sa_search(base_edges, walk_coeffs, m, time_limit):
        """
        Simulated Annealing to find voltage assignment v in Z_m^E
        such that no walk sum is 0 mod m.
        Returns (volts, valid)
        """
        n_edges = len(base_edges)
        if n_edges == 0:
            return [], True
        
        # Initial random state
        volts = [random.randint(1, m - 1) for _ in range(n_edges)]
        
        def calc_energy(v):
            energy = 0
            for coeffs in walk_coeffs:
                s = 0
                for idx, c in coeffs:
                    s += c * v[idx]
                if s % m == 0:
                    energy += 1
            return energy
        
        current_energy = calc_energy(volts)
        if current_energy == 0:
            return volts, True
            
        T = 1.0
        cooling_rate = 0.999
        start_t = time.time()
        best_volts = volts[:]
        best_energy = current_energy
        
        while T > 0.01 and time.time() - start_t < time_limit:
            # Pick random edge to perturb
            idx = random.randint(0, n_edges - 1)
            old_val = volts[idx]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            volts[idx] = new_val
            
            new_energy = calc_energy(volts)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_volts = volts[:]
                    if best_energy == 0:
                        return best_volts, True
            else:
                volts[idx] = old_val
                
            T *= cooling_rate
            
        return best_volts, best_energy == 0

    # Strategy 1: Heawood Lift
    # Moore bound 512. n = 14m. m >= ceil(512/14) = 37.
    # Try m from 37 to 100.
    heawood_coeffs = _get_walk_coeffs(heawood_edges, n_heawood, g)
    
    for m in range(37, 150):
        if time.time() > deadline - 20.0:
            break
        candidate_n = n_heawood * m
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
        # Allocate time based on remaining
        alloc_time = min(5.0, time_left / 5.0)
        
        # Try a few restarts
        found = False
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            volts, valid = _sa_search(heawood_edges, heawood_coeffs, m, alloc_time)
            if valid:
                # Construct lift
                lift_edges = []
                for i, (u, v) in enumerate(heawood_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))
                
                if _verify_girth(candidate_n, lift_edges, g):
                    return candidate_n, lift_edges
                else:
                    # Verification failed, likely due to bug or edge case, skip
                    pass
    
    # Strategy 2: Petersen Lift with SA
    petersen_coeffs = _get_walk_coeffs(petersen_edges, n_petersen, g)
    
    # Moore bound 512. n = 10m. m >= 52.
    for m in range(52, 200):
        if time.time() > deadline - 10.0:
            break
        candidate_n = n_petersen * m
        time_left = deadline - time.time() - 5.0
        if time_left < 1.0:
            break
        alloc_time = min(3.0, time_left / 5.0)
        
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            volts, valid = _sa_search(petersen_edges, petersen_coeffs, m, alloc_time)
            if valid:
                lift_edges = []
                for i, (u, v) in enumerate(petersen_edges):
                    vol = volts[i]
                    for t in range(m):
                        u_lift = u * m + t
                        v_lift = v * m + (t + vol) % m
                        lift_edges.append((u_lift, v_lift))
                
                if _verify_girth(candidate_n, lift_edges, g):
                    return candidate_n, lift_edges

    # Fallback
    return _luw_graph(k, g)
