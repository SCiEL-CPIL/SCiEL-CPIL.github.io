import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Fast, always-valid baseline: LUW construction.
    best_n, best_edges = construct_luw(k, g)

    # 2. Benson-style coset-graph construction.
    try:
        n_b, edges_b = construct_benson(k, g, deadline, best_n)
        if n_b is not None and n_b < best_n:
            best_n = n_b
            best_edges = edges_b
    except Exception:
        pass

    # 3. Double-cover construction: lift a girth-g-1 base by Z_2 voltage.
    try:
        n_d, edges_d = construct_double_cover(k, g, deadline, best_n)
        if n_d is not None and n_d < best_n:
            best_n = n_d
            best_edges = edges_d
    except Exception:
        pass

    # 4. Voltage-lift search over larger cyclic groups.
    try:
        n_lift, edges_lift = construct_voltage_lift(k, g, deadline, best_n)
        if n_lift is not None and n_lift < best_n:
            best_n = n_lift
            best_edges = edges_lift
    except Exception:
        pass

    return best_n, best_edges


def _is_prime(q):
    if q < 2:
        return False
    if q < 4:
        return True
    i = 2
    while i * i <= q:
        if q % i == 0:
            return False
        i += 1
    return True


def construct_benson(k, g, deadline, current_best):
    if k < 3 or k % 2 != 0:
        return None, None

    half = k // 2
    # Try prime q where half < q <= k+3
    for q in range(max(half + 1, 3), min(k + 4, 100) + 1):
        if time.time() > deadline:
            break
        if not _is_prime(q):
            continue
        if half >= q:
            continue

        n = q * q * q
        if n >= current_best:
            continue
        if n * k % 2 != 0:
            continue

        A = list(range(1, half + 1))
        edges_set = set()
        for c in range(q):
            for a in range(q):
                for b in range(q):
                    u = (a * q + b) * q + c
                    for t in A:
                        a2 = (a + t) % q
                        b2 = (b + t * c) % q
                        v = (a2 * q + b2) * q + c
                        edges_set.add((min(u, v), max(u, v)))

        edges = list(edges_set)
        if len(edges) != n * k // 2:
            continue

        if n <= 5000:
            if verify_girth(n, edges, k, g):
                return n, edges
        else:
            return n, edges

    return None, None


def construct_luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]

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


def construct_double_cover(k, g, deadline, current_best):
    """Construct a double cover (Z_2 voltage lift) of a k-regular base graph.
    
    The base graph has girth >= g-1. We assign voltages in {0,1} to each edge.
    The double cover has 2*n_base vertices. A cycle of length L in the base lifts
    to a cycle of length L in the cover iff its total voltage is 0 mod 2.
    So we need: every cycle of length < g in the base must have odd total voltage.
    """
    # Generate base graphs with girth >= g-1
    def gen_base_graphs():
        out = []
        # Try circulants and random regular graphs
        for n in range(2 * k, 2 * k + 15):
            if n * k % 2 != 0:
                continue
            # Circulant
            if n % 2 == 0 and k == n // 2:
                edges = set()
                for i in range(n):
                    for s in range(1, n // 2 + 1):
                        a, b = i, (i + s) % n
                        edges.add((min(a, b), max(a, b)))
                el = list(edges)
                if len(el) == n * k // 2:
                    out.append((n, el))
            elif n % 2 != 0 and k % 2 != 0:
                half = k // 2
                edges = set()
                for i in range(n):
                    for s in range(1, half + 1):
                        a, b = i, (i + s) % n
                        edges.add((min(a, b), max(a, b)))
                el = list(edges)
                if len(el) == n * k // 2:
                    out.append((n, el))
            # Random
            for _ in range(5):
                if time.time() > deadline:
                    return out
                rg = gen_random_k_reg(n)
                if rg is not None:
                    out.append((n, rg))
        seen = set()
        res = []
        for n, e in out:
            key = frozenset(e)
            if key not in seen:
                seen.add(key)
                res.append((n, e))
        return res

    def gen_random_k_reg(n, tries=200):
        if n * k % 2 != 0 or n < k + 1:
            return None
        for _ in range(tries):
            adj = [set() for _ in range(n)]
            deg = [0] * n
            ok = True
            for _ in range(n * k // 2):
                cands = [v for v in range(n) if deg[v] < k]
                if not cands:
                    ok = False
                    break
                u = random.choice(cands)
                cands2 = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
                if not cands2:
                    ok = False
                    break
                v = random.choice(cands2)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            if ok and all(len(adj[v]) == k for v in range(n)):
                return [(min(a, b), max(a, b)) for a in range(n) for b in adj[a]]
        return None

    def base_girth_ge(base_n, base_edges, girth_req):
        """Check if base graph has girth >= girth_req using non-backtracking BFS."""
        adj = [[] for _ in range(base_n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for start in range(base_n):
            if len(adj[start]) != k:
                return False
            visited = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            while queue:
                curr, prev, dist = queue.popleft()
                if curr == start:
                    if dist < girth_req:
                        return False
                    break
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    key = (nb, curr)
                    if key not in visited:
                        visited[key] = dist + 1
                        queue.append((nb, curr, dist + 1))
        return True

    def find_valid_voltage(base_n, base_edges, m):
        """Find voltage assignment in Z_m such that all cycles of length < g have non-zero sum.
        For m=2: all cycles of length < g must have odd length (voltage sum 1).
        We use a smart approach: first check if all short cycles have odd length.
        If so, voltage 1 on all edges works (every cycle has sum = length mod 2 = 1 for odd, 0 for even).
        Actually we need: no cycle of length L < g has voltage sum 0.
        For m=2, sum = parity of number of edges with voltage 1 on the cycle.
        Simplest: if all cycles of length < g are odd, then voltage=1 on all edges works.
        """
        if m == 2:
            # Check: are all cycles of length 3..g-1 odd?
            # If the graph is bipartite, all cycles are even -> voltage 1 on all gives even sum = 0 -> bad.
            # If not bipartite, there's an odd cycle. But we need ALL short cycles to be odd.
            # Strategy: assign voltage 1 to all edges, then check if any short cycle has sum 0.
            # Sum of voltage-1 edges on a cycle = length of cycle (if all voltages are 1).
            # So sum = L mod 2. This is 0 iff L is even.
            # So voltage-1-on-all works iff no even cycle of length < g exists.
            # Equivalently: all cycles of length < g are odd.
            # A graph where all cycles are odd is one where every cycle is odd.
            # This is true iff the graph is a tree plus... no, a graph where every cycle is odd
            # must have all cycles of the same parity? No. A graph can have both odd and even cycles.
            # We need: no cycle of length 3,4,...,g-1 is even.
            # i.e., all cycles of length < g are odd.
            # Let's just enumerate non-backtracking walks and check.
            
            # Build adjacency
            adj = [[] for _ in range(base_n)]
            for u, v in base_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # For each pair of consecutive edges (u,v,w), track non-backtracking walks
            # and check if any closed walk of length L < g has even total voltage.
            # With voltage 1 on all edges, total voltage = L.
            # So we just need: no cycle of length L < g where L is even.
            # Check if there's an even cycle of length < g.
            # An even cycle of length L exists iff there's a non-backtracking closed walk of length L.
            
            # Use BFS to find shortest even cycle
            # Actually, let's just check: does the graph have girth >= g among even cycles?
            # Simplest: run non-backtracking BFS and check if any start vertex is reached
            # at an even distance < g.
            
            for start in range(base_n):
                # BFS tracking (curr, prev) -> min distance
                visited = {}
                queue = deque()
                for nb in adj[start]:
                    queue.append((nb, start, 1))
                    visited[(nb, start)] = 1
                while queue:
                    curr, prev, dist = queue.popleft()
                    if curr == start:
                        # Found a cycle of length dist
                        if dist < g:
                            if dist % 2 == 0:
                                return None  # even short cycle exists, voltage-1-all fails
                        break
                    for nb in adj[curr]:
                        if nb == prev:
                            continue
                        key = (nb, curr)
                        if key not in visited:
                            visited[key] = dist + 1
                            queue.append((nb, curr, dist + 1))
            
            # All good: assign voltage 1 to every edge
            volt = {(u, v): 1 for (u, v) in base_edges}
            return volt
        
        # For m > 2: try random assignments
        for _ in range(500):
            if time.time() > deadline:
                return None
            volt = {(u, v): random.randint(0, m - 1) for (u, v) in base_edges}
            if check_voltage_valid(base_n, base_edges, volt, m, g):
                return volt
        return None

    def check_voltage_valid(base_n, base_edges, volt, m, girth_req):
        """Check that no non-backtracking closed walk of length < girth_req has voltage sum 0 mod m."""
        adj = [[] for _ in range(base_n)]
        for (u, v) in base_edges:
            vu = volt.get((u, v), 0)
            adj[u].append((v, vu))
            adj[v].append((u, (-vu) % m))
        
        # state: (start, curr, prev) -> set of voltage sums
        cur = {}
        for u in range(base_n):
            for (v, w) in adj[u]:
                key = (u, v, u)
                if key not in cur:
                    cur[key] = set()
                cur[key].add(w)
        
        for L in range(2, girth_req - 1):  # total length L+1
            if time.time() > deadline:
                return None
            nxt = {}
            for (s, c, p), sums in cur.items():
                for (v, w) in adj[c]:
                    if v == p:
                        continue
                    for sm in sums:
                        ns = (sm + w) % m
                        key = (s, v, c)
                        if key not in nxt:
                            nxt[key] = set()
                        nxt[key].add(ns)
            cur = nxt
            for (s, c, p), sums in cur.items():
                if c == s and 0 in sums:
                    return False
        return True

    # Main: try base graphs, find valid voltage, build cover
    base_graphs = gen_base_graphs()
    # Sort by size
    base_graphs.sort(key=lambda x: x[0])
    
    for base_n, base_edges in base_graphs:
        if time.time() > deadline:
            break
        cover_n = 2 * base_n
        if cover_n >= current_best:
            break
        if cover_n * k % 2 != 0:
            continue
        
        # Check base has girth >= g-1
        if not base_girth_ge(base_n, base_edges, g - 1):
            continue
        
        # Try m=2 first (double cover)
        volt = find_valid_voltage(base_n, base_edges, 2)
        if volt is not None:
            # Build the double cover
            cover_edges = []
            for (u, v) in base_edges:
                vu = volt[(u, v)]
                cover_edges.append((u * 2 + 0, v * 2 + ((0 + vu) % 2)))
                cover_edges.append((u * 2 + 1, v * 2 + ((1 + vu) % 2)))
            # Verify
            if verify_girth(cover_n, cover_edges, k, g):
                return cover_n, cover_edges
        
        # Try m=3,4,5 if still within budget
        if time.time() < deadline - 20:
            for m in (3, 4, 5):
                if time.time() > deadline:
                    break
                cover_n_m = m * base_n
                if cover_n_m >= current_best:
                    break
                volt = find_valid_voltage(base_n, base_edges, m)
                if volt is not None:
                    cover_edges = []
                    for (u, v) in base_edges:
                        vu = volt[(u, v)]
                        for t in range(m):
                            cover_edges.append((u * m + t, v * m + ((t + vu) % m)))
                    if verify_girth(cover_n_m, cover_edges, k, g):
                        return cover_n_m, cover_edges

    return None, None


def construct_voltage_lift(k, g, deadline, current_best):
    """General voltage lift search over cyclic groups Z_m."""
    def gen_base_graphs():
        out = []
        for n in range(2 * k, 2 * k + 10):
            if n * k % 2 != 0:
                continue
            if n % 2 == 0 and k == n // 2:
                edges = set()
                for i in range(n):
                    for s in range(1, n // 2 + 1):
                        a, b = i, (i + s) % n
                        edges.add((min(a, b), max(a, b)))
                el = list(edges)
                if len(el) == n * k // 2:
                    out.append((n, el))
            for _ in range(3):
                rg = gen_random_k_reg(n)
                if rg is not None:
                    out.append((n, rg))
        seen = set()
        res = []
        for n, e in out:
            key = frozenset(e)
            if key not in seen:
                seen.add(key)
                res.append((n, e))
        return res

    def gen_random_k_reg(n, tries=100):
        if n * k % 2 != 0 or n < k + 1:
            return None
        for _ in range(tries):
            adj = [set() for _ in range(n)]
            deg = [0] * n
            ok = True
            for _ in range(n * k // 2):
                cands = [v for v in range(n) if deg[v] < k]
                if not cands:
                    ok = False
                    break
                u = random.choice(cands)
                cands2 = [v for v in range(n) if v != u and deg[v] < k and v not in adj[u]]
                if not cands2:
                    ok = False
                    break
                v = random.choice(cands2)
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            if ok and all(len(adj[v]) == k for v in range(n)):
                return [(min(a, b), max(a, b)) for a in range(n) for b in adj[a]]
        return None

    def base_girth_ge(base_n, base_edges, girth_req):
        adj = [[] for _ in range(base_n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)
        for start in range(base_n):
            if len(adj[start]) != k:
                return False
            visited = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1))
                visited[(nb, start)] = 1
            while queue:
                curr, prev, dist = queue.popleft()
                if curr == start:
                    if dist < girth_req:
                        return False
                    break
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    key = (nb, curr)
                    if key not in visited:
                        visited[key] = dist + 1
                        queue.append((nb, curr, dist + 1))
        return True

    def check_voltage_valid(base_n, base_edges, volt, m, girth_req):
        adj = [[] for _ in range(base_n)]
        for (u, v) in base_edges:
            vu = volt.get((u, v), 0)
            adj[u].append((v, vu))
            adj[v].append((u, (-vu) % m))
        cur = {}
        for u in range(base_n):
            for (v, w) in adj[u]:
                key = (u, v, u)
                if key not in cur:
                    cur[key] = set()
                cur[key].add(w)
        for L in range(2, girth_req - 1):
            if time.time() > deadline:
                return None
            nxt = {}
            for (s, c, p), sums in cur.items():
                for (v, w) in adj[c]:
                    if v == p:
                        continue
                    for sm in sums:
                        ns = (sm + w) % m
                        key = (s, v, c)
                        if key not in nxt:
                            nxt[key] = set()
                        nxt[key].add(ns)
            cur = nxt
            for (s, c, p), sums in cur.items():
                if c == s and 0 in sums:
                    return False
        return True

    base_graphs = gen_base_graphs()
    base_graphs.sort(key=lambda x: x[0])

    for base_n, base_edges in base_graphs:
        if time.time() > deadline:
            break
        m_max = max(2, current_best // base_n - 1)
        m_max = min(m_max, 8)
        for m in range(2, m_max + 1):
            if time.time() > deadline:
                break
            total_n = m * base_n
            if total_n >= current_best:
                break
            if total_n * k % 2 != 0:
                continue
            # Need base girth >= g-1 for the lift to potentially have girth >= g
            # (a cycle of length L in base lifts to cycle of length L in cover only if voltage sum is 0)
            # Actually we need base to have no cycles shorter than g that lift to cycles in cover.
            # The safest: base girth >= g.
            if not base_girth_ge(base_n, base_edges, g):
                continue
            # Try random voltage assignments
            E = len(base_edges)
            for _ in range(200):
                if time.time() > deadline:
                    return None, None
                volt = {(u, v): random.randint(0, m - 1) for (u, v) in base_edges}
                r = check_voltage_valid(base_n, base_edges, volt, m, g)
                if r is None:
                    return None, None
                if r:
                    cover_edges = []
                    for (u, v) in base_edges:
                        vu = volt[(u, v)]
                        for t in range(m):
                            cover_edges.append((u * m + t, v * m + ((t + vu) % m)))
                    if verify_girth(total_n, cover_edges, k, g):
                        return total_n, cover_edges
    return None, None


def verify_girth(n, edges, k, g):
    """Full non-backtracking BFS from every vertex; returns True iff girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for v in range(n):
        if len(adj[v]) != k:
            return False
    for start in range(n):
        visited = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        while queue:
            curr, prev, dist = queue.popleft()
            if curr == start:
                if dist < g:
                    return False
                break
            for nb in adj[curr]:
                if nb == prev:
                    continue
                key = (nb, curr)
                if key not in visited:
                    visited[key] = dist + 1
                    queue.append((nb, curr, dist + 1))
    return True
