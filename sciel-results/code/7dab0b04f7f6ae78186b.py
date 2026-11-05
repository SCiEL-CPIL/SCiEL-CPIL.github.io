import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    if k == 3:
        # Use the voltage lift approach on the Petersen graph with Simulated Annealing
        result = _construct_petersen_lift_sa(g, deadline, start_time)
        if result is not None:
            return result
        # Fallback to LUW construction
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift_sa(g, deadline, start_time):
    # Petersen graph edges
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    n_edges_base = 15

    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in Petersen
    def _get_closed_nb_walks(max_len):
        walks = set()
        for start_v in range(n_base):
            stack = []
            for next_v, e_id in base_adj[start_v]:
                u, v = pet_edges[e_id]
                sign = 1 if u == start_v else -1
                stack.append((next_v, e_id, 1, [(e_id, sign)]))
            while stack:
                curr_v, prev_e, length, path = stack.pop()
                if length >= max_len:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = pet_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    new_path = path + [(e_id, sign)]
                    stack.append((next_v, e_id, length + 1, new_path))
        return list(walks)

    closed_walks = _get_closed_nb_walks(g)

    # Precompute sparse coefficient vectors
    walk_coeffs = []
    for w in closed_walks:
        coeff = [0] * n_edges_base
        for e_id, sign in w:
            coeff[e_id] += sign
        sparse = [(eid, c) for eid, c in enumerate(coeff) if c != 0]
        if sparse:
            walk_coeffs.append(sparse)

    seen = set()
    unique_walk_coeffs = []
    for sparse in walk_coeffs:
        key = tuple(sorted(sparse))
        if key not in seen:
            seen.add(key)
            unique_walk_coeffs.append(sparse)
    walk_coeffs = unique_walk_coeffs

    best_n = float('inf')
    best_edges = None
    
    random.seed(42)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Generate candidate m values, prioritizing primes
    m_candidates = []
    for m in range(52, 300):
        if is_prime(m):
            m_candidates.append(m)
    for m in range(52, 300):
        if m not in m_candidates:
            m_candidates.append(m)
    
    m_candidates.sort()
    
    for m in m_candidates:
        if time.time() > deadline - 25.0:
            break
        candidate_n = n_base * m
        if candidate_n >= best_n:
            continue

        # Simulated Annealing
        # Objective: number of violated constraints
        # Constraint i: sum(c_j * v_j) % m != 0
        
        def count_violations(volts):
            viol = 0
            for sparse in walk_coeffs:
                s = 0
                for eid, c in sparse:
                    s += c * volts[eid]
                if s % m == 0:
                    viol += 1
            return viol

        def sa_search(m, max_iters=5000, restarts=20):
            best_volts = None
            best_viol = float('inf')
            
            for r in range(restarts):
                if time.time() > deadline - 10.0:
                    return None
                
                volts = [random.randint(1, m - 1) for _ in range(n_edges_base)]
                # Ensure gcd is 1? SA might find non-connected. We can check later.
                # Actually, if gcd > 1, the graph is disconnected. 
                # The problem asks for "a graph". Disconnected is allowed?
                # "every vertex must have degree exactly k".
                # "no cycle may be shorter than g".
                # If disconnected, each component must satisfy girth.
                # A lift with gcd d > 1 is d disjoint copies of a lift with m/d.
                # If m/d is valid, then m is valid (just repeated).
                # But we want minimum n. So we should prefer gcd=1 (connected) or just smallest n.
                # If we find a solution with m, n=10m.
                
                current_viol = count_violations(volts)
                if current_viol == 0:
                    return volts
                
                T = 10.0
                cooling = 0.99
                step = 0
                
                while step < max_iters and T > 0.01:
                    if time.time() > deadline - 10.0:
                        break
                    
                    # Pick a random edge to change
                    idx = random.randint(0, n_edges_base - 1)
                    old_val = volts[idx]
                    delta = random.randint(1, m - 1)
                    new_val = (old_val + delta) % m
                    if new_val == 0:
                        new_val = 1 # Avoid 0? 0 is allowed in Z_m, but if all 0, not regular?
                                     # Voltage 0 means edge connects (u,t) to (v,t).
                                     # This is fine.
                        # Actually, if we allow 0, we must ensure the graph is 3-regular.
                        # Lifts are always 3-regular.
                        # So 0 is allowed.
                    
                    volts[idx] = new_val
                    new_viol = count_violations(volts)
                    
                    delta_viol = new_viol - current_viol
                    
                    if delta_viol < 0 or (delta_viol == 0 and random.random() < 0.5) or random.random() < math.exp(-delta_viol / T):
                        current_viol = new_viol
                        if current_viol < best_viol:
                            best_viol = current_viol
                            best_volts = volts[:]
                        if current_viol == 0:
                            return best_volts
                    else:
                        volts[idx] = old_val
                    
                    T *= cooling
                    step += 1
            
            return best_volts if best_viol == 0 else None

        # Run SA
        found_volts = sa_search(m)
        
        if found_volts:
            # Check gcd for connectivity (optional but good for "cage" context)
            # If gcd != 1, it's a multiple of a smaller graph.
            # We iterate m from small to large, so if a smaller m/d worked, we would have found it?
            # Not necessarily, because the voltage assignment for m/d might not exist, but one for m does.
            # However, if a solution exists for m, and gcd(v)=d>1, then v/d is a solution for m/d?
            # Sum c_i (v_i/d) = (1/d) Sum c_i v_i.
            # If Sum c_i v_i = 0 mod m, then Sum c_i v_i = k*m.
            # Sum c_i (v_i/d) = k*m/d = k*(m/d).
            # So Sum c_i (v_i/d) = 0 mod (m/d).
            # Yes! If v is a solution for m with gcd d, then v/d is a solution for m/d.
            # So we only need to search for primitive solutions (gcd=1).
            
            g_val = m
            for v in found_volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                # This is a non-primitive solution.
                # We should have found the primitive one at m/g_val earlier.
                # If we didn't, maybe it didn't exist then?
                # But we just proved it does.
                # So this is redundant. We can skip or accept.
                # Accepting it gives n=10m. The primitive one gives n=10(m/g_val) which is smaller.
                # Since we iterate m increasing, we would have hit m/g_val earlier.
                # If we failed to find it then, maybe SA was unlucky.
                # We can try to use this.
                pass

            # Build the lifted graph
            lift_edges = []
            for i, (u, v) in enumerate(pet_edges):
                vol = found_volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))

            if _verify_girth(candidate_n, lift_edges, g):
                best_n = candidate_n
                best_edges = lift_edges
                break

    if best_edges is not None:
        return best_n, best_edges

    return _construct_luw(k=3, g=g, d=11)

def _verify_girth(n, edges, girth_req):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != 3:
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

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, (g - 2) // 2)
    
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
