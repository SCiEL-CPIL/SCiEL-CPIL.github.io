import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150 s kill

    # ---------- LUW construction (guaranteed fallback) ----------
    def luw_construct(k, g, d_override=None, rule_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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
        if rule_override is not None:
            rule = rule_override
        else:
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

    def gen_rules(d, variant=0):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            if variant == 0:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 1:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 1), (c0, 1), (c0 + 1, 1), (1, c0)][r]
            elif variant == 2:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(2, c0 - 1), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            elif variant == 3:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 1) % 4]
            elif variant == 4:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 2) % 4]
            elif variant == 5:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][(r + 3) % 4]
            elif variant == 6:
                rule[c] = (1, c - 2) if c % 2 == 1 else (c - 1, 1)
            elif variant == 7:
                rule[c] = (1, c - 2) if c % 3 == 0 else ((c - 1, 1) if c % 3 == 1 else (c, 1))
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    # ---------- girth verification (non-backtracking BFS, task contract) ----------
    def verify_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or v < 0 or u >= n or v >= n:
                return False
            key = (u, v) if u < v else (v, u)
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        half = (target_girth - 1) // 2
        for s in range(n):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue
                    st = (w, v)
                    if st in dist:
                        if dcur + dist[st] + 1 < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        queue.append(st)
        return True

    if k != 3 or g != 16:
        return luw_construct(k, g)

    # ---------- Petersen-graph voltage lift with finite-field (algebraic) voltages ----------
    def petersen_voltage_lift(k, g, deadline):
        # Petersen graph: 3-regular, girth 5, 10 vertices, 15 edges.
        pe = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
              (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
              (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        n_base = 10
        nb_adj = [[] for _ in range(n_base)]
        for ei, (u, v) in enumerate(pe):
            nb_adj[u].append((v, ei))
            nb_adj[v].append((u, ei))

        # All closed non-backtracking walks of length 3..g-1 (as edge-id sequences).
        walks = []
        for sv in range(n_base):
            stack = []
            for nv, ei in nb_adj[sv]:
                stack.append((nv, ei, 1, [ei]))
            while stack:
                cv, pei, L, path = stack.pop()
                if L >= g:
                    continue
                if cv == sv and L >= 3:
                    walks.append(tuple(path))
                for nv, ei in nb_adj[cv]:
                    if ei == pei:
                        continue
                    stack.append((nv, ei, L + 1, path + [ei]))
        # Deduplicate.
        walks = list(set(walks))
        W = len(walks)

        def is_prime(x):
            if x < 2:
                return False
            i = 2
            while i * i <= x:
                if x % i == 0:
                    return False
                i += 1
            return True

        # Candidate moduli: primes with 10*m >= 512 (Moore bound).
        primes = []
        x = 53
        while len(primes) < 80:
            if is_prime(x):
                primes.append(x)
            x += 1

        rng = random.Random(12345)

        def build_lift(m, volts):
            return [(u * m + t, v * m + (t + vol) % m)
                    for (u, v), vol in zip(pe, volts)
                    for t in range(m)]

        def precheck(m, volts):
            for w in walks:
                s = 0
                for ei in w:
                    s += volts[ei]
                if s % m == 0:
                    return False
            return True

        def field_voltages(m):
            # primitive element of GF(m)
            for a in range(2, m):
                seen = {1}
                cur = 1
                ok = True
                for _ in range(m - 1):
                    cur = (cur * a) % m
                    if cur in seen:
                        ok = False
                        break
                    seen.add(cur)
                if ok:
                    return a
            return None

        best = None
        for m in primes:
            if time.time() > deadline - 8.0:
                break
            if 10 * m >= (best[0] if best else 10 ** 9):
                continue
            a = field_voltages(m)
            if a is None:
                continue

            # Build a diverse, deterministic set of voltage assignments.
            candidates = []
            # (1) all edges = same power of a primitive element
            for p in range(1, m):
                candidates.append([pow(a, p, m)] * 15)
            # (2) two alternating groups (inner / outer / spokes)
            for p in range(1, m):
                for q2 in range(1, m):
                    ap, aq = pow(a, p, m), pow(a, q2, m)
                    candidates.append([ap] * 5 + [aq] * 5 + [ap] * 5)
            # (3) random assignments
            for _ in range(250):
                candidates.append([rng.randrange(1, m) for _ in range(15)])

            seen_v = set()
            for volts in candidates:
                if time.time() > deadline - 6.0:
                    break
                key = tuple(volts)
                if key in seen_v:
                    continue
                seen_v.add(key)
                if not precheck(m, volts):
                    continue
                edges = build_lift(m, volts)
                if verify_girth(10 * m, edges, g):
                    return 10 * m, edges

        return None

    # ---------- Configuration-model random 3-regular graph with girth repair ----------
    def config_model_repair(k, g, deadline, seed=42):
        rng = random.Random(seed)
        # Moore bound for k=3, g=16: 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*(2^8-1) = 512
        # Try a few sizes; aim to find a valid graph.
        # Start with a size that's feasible but not too large.
        # The known best is 936. Let's try sizes around 512-936.
        # Actually, random 3-regular graphs have girth ~ O(log n). For n=512,
        # expected girth is around log_2(512) = 9, so we'd need to repair many cycles.
        # Let's start with a larger n where girth is more likely to be >= 16.
        # For n ~ 1000, typical girth is around 10-12. Still need repair.
        # For n ~ 5000+, girth tends to be higher. But we want small n.
        
        # Strategy: try multiple n values and multiple random seeds.
        # For each, generate a random 3-regular graph and try to repair.
        
        def generate_random_3reg(n, rng):
            """Generate a random 3-regular simple graph on n vertices using configuration model."""
            if n < 4 or n % 2 != 0:
                return None
            # Each vertex has 3 stubs
            stubs = [i for i in range(n) for _ in range(3)]
            rng.shuffle(stubs)
            edges = []
            for i in range(0, 3 * n, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    return None  # loop
                if (min(u, v), max(u, v)) in set((min(e[0], e[1]), max(e[0], e[1])) for e in edges):
                    return None  # parallel edge
                edges.append((u, v))
            # Verify it's 3-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                return None
            return edges

        def find_shortest_cycles(adj, n, target_girth):
            """Find all cycles of length < target_girth. Returns list of cycles (as vertex lists)."""
            cycles = []
            half = (target_girth - 1) // 2
            for s in range(n):
                # BFS from s, tracking parent
                dist = {s: 0}
                parent = {s: None}
                queue = deque([s])
                while queue:
                    v = queue.popleft()
                    if dist[v] >= half:
                        continue
                    for w in adj[v]:
                        if w not in dist:
                            dist[w] = dist[v] + 1
                            parent[w] = v
                            queue.append(w)
                        elif w != parent.get(v) and dist[w] < half:
                            # Found a cycle
                            # Reconstruct cycle from v to w via parents
                            # Actually, this finds cycles but may find duplicates
                            # For repair, we just need to know if girth < target
                            pass
            # Simpler: just check girth
            return cycles

        def check_girth_fast(adj, n, target_girth):
            """Quick check: return True if girth >= target_girth."""
            half = (target_girth - 1) // 2
            for s in range(n):
                dist = {}
                queue = deque()
                for nb in adj[s]:
                    st = (nb, s)
                    dist[st] = 1
                    queue.append(st)
                while queue:
                    v, p = queue.popleft()
                    dcur = dist[(v, p)]
                    if dcur >= half:
                        continue
                    for w in adj[v]:
                        if w == p:
                            continue
                        st = (w, v)
                        if st in dist:
                            if dcur + dist[st] + 1 < target_girth:
                                return False
                        else:
                            dist[st] = dcur + 1
                            queue.append(st)
            return True

        def swap_edges(adj, n, edges, rng):
            """Perform a 2-switch to maintain 3-regularity. Returns new edge list or None."""
            # Pick two edges (a,b) and (c,d) with no common vertices
            # Remove them, add (a,c) and (b,d) or (a,d) and (b,c)
            if len(edges) < 2:
                return None
            for _ in range(100):  # try up to 100 random pairs
                i, j = rng.sample(range(len(edges)), 2)
                a, b = edges[i]
                c, d = edges[j]
                if len({a, b, c, d}) < 4:
                    continue
                # Try both possible rewirings
                for new_edges in [((a, c), (b, d)), ((a, d), (b, c))]:
                    (e1, e2) = new_edges
                    if e1[0] == e1[1] or e2[0] == e2[1]:
                        continue
                    # Check no parallel edges
                    key1 = (min(e1[0], e1[1]), max(e1[0], e1[1]))
                    key2 = (min(e2[0], e2[1]), max(e2[0], e2[1]))
                    # Check these don't already exist
                    existing = set()
                    for u, v in edges:
                        existing.add((min(u, v), max(u, v)))
                    existing.discard((min(a, b), max(a, b)))
                    existing.discard((min(c, d), max(c, d)))
                    if key1 in existing or key2 in existing:
                        continue
                    # Apply the swap
                    new_edge_list = list(edges)
                    del new_edge_list[i]
                    del new_edge_list[j]
                    new_edge_list.append(e1)
                    new_edge_list.append(e2)
                    return new_edge_list
            return None

        # Try different sizes
        # The Moore bound is 512. Known best is 936.
        # Let's try a range of sizes.
        sizes_to_try = []
        # Start with sizes that are multiples of 2 (for 3-regular, n must be even)
        # Try sizes from around 512 to 1000
        for n in range(512, 1000, 8):
            sizes_to_try.append(n)
        # Also try some larger sizes
        for n in range(1000, 1500, 16):
            sizes_to_try.append(n)

        best_n = float('inf')
        best_edges = None

        for n in sizes_to_try:
            if time.time() > deadline - 10.0:
                break
            if n >= best_n:
                continue
            
            # Try multiple random seeds for this size
            for seed in range(20):
                if time.time() > deadline - 8.0:
                    break
                rng = random.Random(seed * 1000 + n)
                edges = generate_random_3reg(n, rng)
                if edges is None:
                    continue
                
                # Build adjacency
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check girth
                if check_girth_fast(adj, n, g):
                    best_n = n
                    best_edges = edges
                    break
                
                # Try to repair: perform random 2-switches and check girth
                current_edges = edges
                current_adj = adj
                improved = False
                for step in range(200):
                    if time.time() > deadline - 5.0:
                        break
                    new_edges = swap_edges(current_adj, n, current_edges, rng)
                    if new_edges is None:
                        continue
                    # Build new adjacency
                    new_adj = [[] for _ in range(n)]
                    for u, v in new_edges:
                        new_adj[u].append(v)
                        new_adj[v].append(u)
                    
                    # Check girth
                    if check_girth_fast(new_adj, n, g):
                        best_n = n
                        best_edges = new_edges
                        improved = True
                        break
                    
                    # Accept the swap if it doesn't make girth worse (hard to check quickly)
                    # For simplicity, always accept
                    current_edges = new_edges
                    current_adj = new_adj
        
        if best_edges is not None:
            return best_n, best_edges
        return None

    # ---------- main search ----------
    best_n = float('inf')
    best_edges = None

    # 1) algebraic Petersen voltage lift (aim well below 936)
    res = petersen_voltage_lift(k, g, deadline)
    if res is not None:
        best_n, best_edges = res

    # 2) Configuration model with repair
    res2 = config_model_repair(k, g, deadline)
    if res2 is not None:
        n2, e2 = res2
        if n2 < best_n:
            best_n, best_edges = n2, e2

    # 3) LUW family (guaranteed valid, fallback)
    for d in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 2.0:
            break
        max_variants = 12 if d == 7 else (8 if d == 8 else 5)
        for variant in range(max_variants):
            if time.time() > deadline - 2.0:
                break
            rule = gen_rules(d, variant)
            n, edges = luw_construct(k, g, d_override=d, rule_override=rule)
            if n >= best_n:
                continue
            if verify_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is not None:
        return best_n, best_edges

    return luw_construct(k, g)
