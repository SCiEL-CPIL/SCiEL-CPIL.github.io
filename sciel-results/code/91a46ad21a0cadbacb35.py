import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        """Verify girth >= g using non-backtracking BFS. Returns g if valid, <g if cycle found, -1 if timeout."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        # Verify 3-regular
        for v in range(n):
            if len(adj[v]) != k:
                return -1
        half = (g - 1) // 2
        # For each vertex, do NB-BFS up to depth half
        # State: (node, prev) meaning we're at node, came from prev
        # A cycle of length L is found when we reach (node, prev) where prev is adjacent to start
        # and the total path length is L.
        # More precisely: from start s, NB-BFS. If at depth d we reach (u, v) where v is a neighbor of s
        # and d+1 < g, we have a cycle of length d+1 (going back to s).
        # Actually: if we reach (s, v) at depth d, that's a cycle of length d (since we started at s,
        # went to v at depth 1, and now come back to s from v at depth d, cycle length = d).
        # Wait: state (node, prev) means the last step was prev -> node.
        # Start: for each neighbor v of s, state (v, s) at depth 1.
        # If we reach state (s, v) at depth d, that means we went s -> ... -> v -> s, cycle of length d.
        # But we also need to check: if we reach (u, v) where v is adjacent to s and u != s,
        # and then from u we can go to s (non-backtracking, i.e., s != v), that gives cycle of length d+1.
        # Simpler: just do NB-BFS and check if we ever reach a state (node, prev) where
        # node == s (giving cycle of length d) or where prev is a neighbor of s and node != s
        # and s is a neighbor of node and s != prev (giving cycle of length d+1).
        # Actually the cleanest: if at depth d we are at state (u, v), and u is adjacent to s,
        # and v != s (non-backtracking), then we have a walk s -> ... -> v -> u -> s of length d+1.
        # If d+1 < g, cycle found.
        # Also if u == s, we have a cycle of length d (the walk from s back to s).
        for s in range(n):
            if time.time() > time_limit:
                return -1
            # visited: (node, prev) -> depth
            visited = {}
            queue = deque()
            for nb in adj[s]:
                visited[(nb, s)] = 1
                queue.append((nb, s, 1))
            while queue:
                node, prev, d = queue.popleft()
                if d > half:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    # Check if this creates a short cycle
                    if nb == s:
                        # Cycle of length d+1 (s -> ... -> prev -> node -> s, but we're at (node,prev) depth d,
                        # then node -> s is one more step, total d+1)
                        # Wait: state (node, prev) at depth d means path of length d: s -> ... -> prev -> node
                        # Then node -> s gives total path length d+1, which is a cycle of length d+1.
                        if d + 1 < g:
                            return d + 1
                    else:
                        key = (nb, node)
                        if key in visited:
                            # We've seen (nb, node) before at some depth d2
                            # The cycle length would be d2 + d + 1 (but this isn't quite right for NB)
                            # Actually in NB-BFS, if we reach (nb, node) which was already visited,
                            # it means there's a cycle. The cycle length is at most d + visited[key] + 1?
                            # No. Let me think again.
                            # In NB-BFS, each state (u, v) is reached at its shortest depth.
                            # If we try to go to (nb, node) which is already visited at depth d2,
                            # then there are two NB-walks from s: one of length d+1 ending at (nb,node)
                            # and one of length d2. The cycle is formed by going along one and back
                            # along the other. But this is complex.
                            # Simpler approach: just check if nb is adjacent to s.
                            if nb in adj[s]:
                                # We have s -> ... -> node -> nb and nb is adjacent to s
                                # If node != s (which it is since nb != s and we're non-backtracking)
                                # Cycle: s -> ... -> node -> nb -> s, length d + 2
                                # But we need node != s (true) and nb != s (true, we're in else)
                                # and the step node -> nb is non-backtracking (nb != prev, which is true since we checked)
                                # and nb -> s is non-backtracking (s != node, which is true since nb != s and nb adjacent to s means s is a different neighbor)
                                # Wait, we need s != node for nb -> s to be non-backtracking from nb.
                                # nb's previous is node. For nb -> s to be non-backtracking, we need s != node.
                                # Since nb != s and both are neighbors of s, and node is a neighbor of nb...
                                # node could be s? No, because we're in the case nb != s, and we came from prev to node.
                                # If node == s, then we'd be at state (s, prev), but we only start from s's neighbors.
                                # Actually node could be s if we went s -> a -> s, but that's a 2-cycle which can't exist in simple graph.
                                # In a simple graph, no 2-cycles, so node != s if nb != s and both adjacent to s... 
                                # Actually node IS a neighbor of nb, and nb is a neighbor of s. node could be s.
                                # If node == s: state (s, prev) at depth d. Then s -> nb would give cycle of length d+1.
                                # But we already handle nb == s above. Here nb != s.
                                # If node == s: we're at (s, prev), depth d. nb is neighbor of s, nb != prev (non-backtracking).
                                # Cycle: s -> ... -> prev -> s -> nb... wait no. The path is s -> ... -> prev -> node=s.
                                # Then s -> nb. But we started at s. So the walk is s -> ... -> prev -> s -> nb.
                                # This isn't a cycle unless nb == s, which it isn't.
                                # Hmm, I think I'm overcomplicating. Let me use a cleaner method.
                                pass
                        if d + 1 <= half:
                            visited[key] = d + 1
                            queue.append((nb, node, d + 1))
        return g

    def check_girth_v2(n, edges, g, time_limit):
        """Cleaner girth check: BFS from each vertex, track shortest distance to each (node, prev) state.
        A cycle of length L exists iff there's a vertex s and a state (u, v) at depth d where
        v is adjacent to s, u != s, and d + 1 < g (cycle s->...->v->u->s has length d+1).
        Or state (s, v) at depth d means cycle of length d."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for v in range(n):
            if len(adj[v]) != k:
                return -1
        half = (g - 1) // 2
        s_adj_set = [set(adj[s]) for s in range(n)]
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = {}  # (node, prev) -> depth
            queue = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                queue.append((nb, s))
            qi = 0
            while qi < len(queue):
                node, prev = queue[qi]
                qi += 1
                d = dist[(node, prev)]
                if d > half:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # Cycle of length d+1
                        if d + 1 < g:
                            return d + 1
                    else:
                        key = (nb, node)
                        if key in dist:
                            # Already visited, cycle of length dist[key] + d + 1? No.
                            # Actually this means there's a shorter or equal path, so no new info.
                            # But we can check: is there a cycle through s?
                            # If node is in s_adj_set[s] and node != s:
                            #   cycle s -> ... -> prev -> node -> s? No, we need node adjacent to s.
                            # Let me just check: if node is adjacent to s (and node != s),
                            # then s -> ... -> prev -> node -> s is a cycle of length d+1.
                            # But we need the step node->s to be non-backtracking: s != prev.
                            # Since prev -> node and node -> s, non-backtracking means s != prev.
                            # prev is a neighbor of node, s is a neighbor of node. If prev == s, then
                            # the path is s -> ... -> s -> node, which means we already came back to s,
                            # which would've been caught when we reached (s, something).
                            # So if node is in adj[s] and node != s and prev != s:
                            if node in s_adj_set[s] and prev != s:
                                if d + 1 < g:
                                    return d + 1
                        else:
                            if d + 1 <= half:
                                dist[key] = d + 1
                                queue.append((nb, node))
        return g

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
                base[c] = (1, c - 3) if c % 2 == 0 else (c - 2, 1)
        elif variant == 6:
            for c in range(5, d + 1):
                base[c] = (c - 2, 1) if c % 3 == 0 else (1, c - 2)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 5), c - ((c - 5 + offset) % 5)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (c0 - 2, 1)][r % 5]
        return base

    def build_luw(q, d, rule, time_limit, alt_ops=False):
        if alt_ops:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: (x * y + x + y) % q
        else:
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
        index = {start: 0}
        order = [start]
        edges = []
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

        for variant in range(7):
            for offset in range(5):
                if remaining() < 15:
                    break
                rule = generate_rules(d, variant, offset)
                for use_alt in [False, True]:
                    if remaining() < 15:
                        break
                    res = build_luw(q, d, rule, deadline - 15, alt_ops=use_alt)
                    if res is None:
                        continue
                    n, edges = res
                    if n >= best_n:
                        continue
                    gc = check_girth_v2(n, edges, g, deadline - 10)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best, best_n = (n, edges), n
                        if n <= 2000:
                            return best

        if best is None:
            for d_try in range(d + 1, d + 15):
                if remaining() < 12:
                    break
                for variant in range(5):
                    for offset in range(3):
                        if remaining() < 8:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 12)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth_v2(n, edges, g, deadline - 8)
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

    # ---------------- Ramanujan cubic graphs via PSL(2,q) ----------------
    def build_ramanujan(q):
        """Construct Ramanujan cubic graph from PSL(2,q). n = q(q+1)/2."""
        def mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)

        def det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q

        def inv(a):
            d = det(a)
            if d == 0:
                return None
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)

        def norm(a):
            return (a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]) % q

        def is_involution(a):
            if det(a) != 1:
                return False
            m = mul(a, a)
            n = norm(m)
            return n == 0 or n == 2

        I = (1, 0, 0, 1)
        A = (0, 1, q - 1, 0)  # [[0,1],[-1,0]]
        B = (0, 1, 1, 0)       # [[0,1],[1,0]]
        if not (is_involution(A) and is_involution(B)):
            return None
        C = None
        for s in range(1, q):
            cand = (1, 0, s, 1)  # [[1,0],[s,1]]
            if is_involution(cand):
                C = cand
                break
        if C is None:
            return None

        # Generate all products
        G = [I, A, B, C,
             mul(A, B), mul(B, A), mul(A, C), mul(C, A),
             mul(B, C), mul(C, B)]
        # Dedup up to sign (in PSL, M and -M are the same)
        seen = set()
        gens = []
        for m in G:
            key = m if m[0] <= (q - m[0]) % q else ((q - m[0]) % q, (q - m[1]) % q, (q - m[2]) % q, (q - m[3]) % q)
            if key not in seen:
                seen.add(key)
                gens.append(m)
        if len(gens) != 3:
            return None

        # BFS over PSL(2,q)
        start = I
        index = {start: 0}
        order = [start]
        edges = []
        for i in range(len(order)):
            v = order[i]
            vi = inv(v)
            if vi is None:
                continue
            for w in gens:
                u = mul(vi, w)
                if u not in index:
                    index[u] = len(order)
                    order.append(u)
                if index[v] < index[u]:
                    edges.append((index[v], index[u]))
        return len(order), edges

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try Ramanujan graphs first (much smaller), then LUW.
    best = None
    best_n = float('inf')

    # Moore bound for k=3, g=16: n >= 2 + 3*(1+2+4+8+16+32) = 2+3*63 = 191
    # Ramanujan: n = q(q+1)/2. For n >= 191, q >= 19.
    # Known girths: q=19->g8, q=31->g8, q=79->g10, q=151->g12, q=211->g14, q=433->g16 (n=93606)
    # We need girth >= g. Try primes in increasing order of n.
    primes = [19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
              83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
              157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
              223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
              281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
              359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433]

    for q in primes:
        if remaining() < 12:
            break
        n_ram = q * (q + 1) // 2
        if n_ram >= best_n:
            break
        if n_ram < 191:
            continue
        res = build_ramanujan(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth_v2(n, edges, g, deadline - 10)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n
            if n <= 5000:
                break

    if best is not None:
        return best

    # Fallback to LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
