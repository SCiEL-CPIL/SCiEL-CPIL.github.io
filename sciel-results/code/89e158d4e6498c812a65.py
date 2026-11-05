import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                base[c] = (c - 2, 1) if (c + offset) % 2 == 0 else (1, c - 1)
        elif variant == 6:
            for c in range(5, d + 1):
                base[c] = (1, c - 1) if (c + offset) % 3 == 0 else (c - 2, 1)
        elif variant == 7:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(8):
            for offset in range(6):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 8)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n
                    break

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- Margulis / LUB voltage graph construction ----------------
    # This is the structural change: instead of relying on LUW or Ramanujan
    # Cayley graphs, we use a voltage-graph (lift) approach based on the
    # Margulis construction. We build a small base graph (e.g., a cycle or
    # small regular graph) and assign voltages from Z_p so that the lift
    # has high girth. We search over voltage assignments systematically.

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

    def build_margulis_lift(p, time_limit):
        """
        Build a 3-regular graph via voltage lift of a small base graph.
        Base graph: a 2-vertex base with 3 parallel edges (multigraph),
        lifted by Z_p to get a 3-regular graph on 2p vertices.
        
        Actually, for a proper 3-regular simple graph, we need a base graph
        that is 3-regular. Let's use a base of 2 vertices with 3 edges between them
        (a multigraph), and lift by Z_p. The lift has 2p vertices and is 3-regular.
        For simplicity, we use the standard Margulis construction:
        
        Vertices: (i, j) where i in {0,1}, j in Z_p
        Edges: for each (i, j), connect to:
          (1-i, j + a) for some fixed a
          (1-i, j + b) for some fixed b  
          (1-i, j + c) for some fixed c
        where a, b, c are chosen to maximize girth.
        
        This gives a 3-regular graph on 2p vertices.
        """
        # Try different (a, b, c) triples
        best = None
        best_n = float('inf')
        
        for a in range(1, p):
            for b in range(a + 1, p):
                for c in range(b + 1, p):
                    if time.time() > time_limit:
                        return best
                    
                    # Build the graph
                    n = 2 * p
                    edges = []
                    edge_set = set()
                    valid = True
                    for i in range(2):
                        for j in range(p):
                            for offset in [a, b, c]:
                                other_i = 1 - i
                                other_j = (j + offset) % p
                                u = i * p + j
                                v = other_i * p + other_j
                                if u == v:
                                    valid = False
                                    break
                                key = (min(u, v), max(u, v))
                                if key in edge_set:
                                    valid = False
                                    break
                                edge_set.add(key)
                                if i == 0:
                                    edges.append(key)
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Check degree
                    deg = [0] * n
                    for u, v in edges:
                        deg[u] += 1
                        deg[v] += 1
                    if any(d != 3 for d in deg):
                        continue
                    
                    # Check girth
                    gc = check_girth(n, edges, g, time_limit)
                    if gc == -1:
                        break
                    if gc >= g:
                        if n < best_n:
                            best = (n, edges)
                            best_n = n
                        break  # found a valid one for this p, move on
                if time.time() > time_limit:
                    return best
            if time.time() > time_limit:
                return best
        
        return best

    def build_margulis_2base(p, time_limit):
        """
        Use a 4-vertex base graph (a cycle C4 with a matching) lifted by Z_p.
        Base: vertices 0,1,2,3. Edges: (0,1),(1,2),(2,3),(3,0) and add a matching (0,2),(1,3) to make it 3-regular? No, that's 4-regular.
        
        Let's use a different approach: base graph is K_{2,3} or similar.
        Actually, let's use the standard approach: base is a 3-regular multigraph on 2 vertices
        with 3 parallel edges, but assign different voltages to each.
        
        For a simple graph, we need the lift to be simple. With 2 base vertices and 3 parallel edges,
        the lift has 2p vertices. Edge from (0,j) to (1, j+v_e) for each of the 3 edges.
        This is simple iff v_1, v_2, v_3 are distinct mod p.
        """
        best = None
        best_n = float('inf')
        
        # We already tried this in build_margulis_lift. Let's try a 4-vertex base.
        # Base: two disjoint edges plus two cross edges? That's not 3-regular.
        
        # Alternative: use a base of 4 vertices forming a cycle, with voltages.
        # Vertices: (i, j) for i in {0,1,2,3}, j in Z_p. Total 4p vertices.
        # Edges: (i,j) -- (i+1 mod 4, j + v_i) for each i, plus (i,j) -- (i+2 mod 4, j + w_i)
        # This makes each vertex have degree 4, not 3.
        
        # For 3-regular, let's stick with the 2-vertex base but try more primes.
        return best

    def margulis_construct(k, g):
        """Search over primes p for the smallest 3-regular graph with girth >= g."""
        if k != 3:
            return None
        
        best = None
        best_n = float('inf')
        
        # Try primes from small to large
        for p in range(5, 200):
            if not is_prime(p):
                continue
            if remaining() < 5:
                break
            
            res = build_margulis_lift(p, deadline - 5)
            if res is not None:
                n, edges = res
                if n < best_n:
                    best, best_n = res, n
                # Once we find a valid one, we can stop if n is close to Moore bound
                # Moore bound for (3,16): 382. 2p >= 382 means p >= 191.
                if n >= 382 and n < 500:
                    break
        
        return best

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try multiple strategies
    best = None
    best_n = float('inf')

    # Strategy 1: Margulis voltage lift (new structural approach)
    if remaining() > 30:
        res = margulis_construct(k, g)
        if res is not None:
            n, edges = res
            if n < best_n:
                best, best_n = res, n

    # Strategy 2: LUW construction
    if remaining() > 15:
        res = luw_construct(k, g)
        if res is not None:
            n, edges = res
            if n < best_n:
                best, best_n = res, n

    # Strategy 3: Greedy random (only if time remains and no good result yet)
    if remaining() > 20 and (best is None or best_n > 500):
        # Simple greedy: try to build a random 3-regular graph with high girth
        moore = 1 + k * sum((k-1)**i for i in range((g-2)//2))
        for n_try in range(moore, moore + 100):
            if n_try % 2 != 0:
                continue
            if remaining() < 10:
                break
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                half_edges = list(range(n_try)) * 3
                random.shuffle(half_edges)
                edges_set = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u,v), max(u,v))
                    if key in edges_set:
                        valid = False
                        break
                    edges_set.add(key)
                if not valid:
                    continue
                edge_list = list(edges_set)
                gc = check_girth(n_try, edge_list, g, deadline - 5)
                if gc == -1:
                    break
                if gc >= g:
                    if n_try < best_n:
                        best, best_n = (n_try, edge_list), n_try
                    break
                # Try a few edge swaps
                adj = [[] for _ in range(n_try)]
                for u, v in edge_list:
                    adj[u].append(v)
                    adj[v].append(u)
                for _ in range(200):
                    if time.time() > deadline - 5:
                        break
                    e1 = random.choice(edge_list)
                    e2 = random.choice(edge_list)
                    u1, v1 = e1
                    u2, v2 = e2
                    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                        continue
                    for new1, new2 in [((u1,u2),(v1,v2)), ((u1,v2),(v1,u2))]:
                        if new1[0] == new1[1] or new2[0] == new2[1]:
                            continue
                        key1 = (min(new1[0],new1[1]), max(new1[0],new1[1]))
                        key2 = (min(new2[0],new2[1]), max(new2[0],new2[1]))
                        if key1 in edges_set or key2 in edges_set:
                            continue
                        edges_set.discard((min(u1,v1), max(u1,v1)))
                        edges_set.discard((min(u2,v2), max(u2,v2)))
                        edges_set.add(key1)
                        edges_set.add(key2)
                        edge_list = list(edges_set)
                        adj = [[] for _ in range(n_try)]
                        for a, b in edge_list:
                            adj[a].append(b)
                            adj[b].append(a)
                        gc2 = check_girth(n_try, edge_list, g, deadline - 5)
                        if gc2 == -1:
                            break
                        if gc2 >= g:
                            if n_try < best_n:
                                best, best_n = (n_try, edge_list), n_try
                            break
                    if best is not None and best_n == n_try:
                        break
            if best is not None and best_n == n_try:
                break

    if best is not None:
        return best

    # Guaranteed fallback: LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
