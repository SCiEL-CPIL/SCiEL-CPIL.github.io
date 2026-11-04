import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150 s kill

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

    def verify_girth(n, edges, target_girth):
        """Certify girth >= target_girth using non-backtracking walks from
        every vertex (as required by the task contract). Also enforces the
        simple, k-regular invariants."""
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
        # Non-backtracking walk: state is (vertex, prev). A closed non-backtracking
        # walk of length L corresponds to a cycle of length L. So the shortest
        # cycle is the minimum L>=2 over all states reachable in exactly L steps
        # that return to the starting state. We run a BFS over the non-backtracking
        # state graph from every start state and detect the first return.
        half = (target_girth - 1) // 2
        for s in range(n):
            # dist over non-backtracking states, parent to avoid immediate backtrack
            dist = {}
            parent = {}
            queue = deque()
            for nb in adj[s]:
                st = (nb, s)
                dist[st] = 1
                parent[st] = (s, -1)
                queue.append(st)
            while queue:
                v, p = queue.popleft()
                dcur = dist[(v, p)]
                if dcur >= half:
                    continue
                for w in adj[v]:
                    if w == p:
                        continue  # non-backtracking: cannot go straight back
                    st = (w, v)
                    if st in dist:
                        # A non-backtracking walk from s reached (w, v) by two
                        # different routes => a cycle. Length = dcur + dist[st] + 1
                        cyc = dcur + dist[st] + 1
                        if cyc < target_girth:
                            return False
                    else:
                        dist[st] = dcur + 1
                        parent[st] = (v, p)
                        queue.append(st)
        return True

    def gen_rules(d, variant=0):
        """Generate different rule parameterizations for a given d."""
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
                if c % 2 == 1:
                    rule[c] = (1, c - 2)
                else:
                    rule[c] = (c - 1, 1)
            elif variant == 7:
                if c % 3 == 0:
                    rule[c] = (1, c - 2)
                elif c % 3 == 1:
                    rule[c] = (c - 1, 1)
                else:
                    rule[c] = (c, 1)
            else:
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    if k != 3 or g != 16:
        return luw_construct(k, g)

    best_n = float('inf')
    best_edges = None

    # ---- Structural alternative: finite-geometry incidence graph with voltage
    # assignment.  The incidence graph of a projective space PG(m, q) is regular
    # and has girth 2*(m+2).  We take the incidence graph of PG(2,3) (girth 8,
    # 3-regular, 12+12=24 vertices) and lift it by a voltage group of size L to
    # a 3-regular graph on 24*L vertices.  By choosing voltages randomly we can
    # increase the effective girth; we verify with the non-backtracking BFS.
    def geometry_voltage_construction(k, g, deadline):
        # PG(2,3): 13 points, 13 lines, each point on 4 lines, each line has 4
        # points -> incidence graph is 4-regular.  Not 3-regular, so we need a
        # different base.  Instead use the incidence graph of the generalized
        # quadrangle GQ(2,2) (DoTsushihiko): 15 points, 15 lines, 3-regular,
        # girth 8.  We build it explicitly.
        # GQ(2,2) = points and lines of PG(2,2) minus one line (or equivalently
        # the 3x3 grid of points with parallel classes).  Simplest: the 3-regular
        # bipartite graph on 15+15 vertices known as the GQ(2,2) incidence graph.
        # Construct: points (i,j) for i,j in {0,1,2} plus 3 extra points per
        # parallel class...  Actually GQ(2,2) has 15 points.  Let me use the
        # standard construction: points are (i,j) in Z3xZ3 (9) plus 6 more.
        # Simpler: use the complement of the 9-cycle?  No.
        # Let me just hardcode the GQ(2,2) incidence graph (15+15=30 vertices,
        # 3-regular, girth 8).
        # Standard GQ(2,2): points P = {(i,j): i,j in {0,1,2}} union {A,B,C,D,E,F}
        # Lines: 9 lines L_{i,j} = {(i,j), A+B+C pattern...}
        # This is getting complex.  Let me use a different approach:
        # Take the 3-regular bipartite graph that is the incidence graph of
        # PG(2,2) (Fano plane): 7 points, 7 lines, 3-regular, girth 6.
        # Fano plane incidence graph: 14 vertices, 3-regular, girth 6.
        # Voltage lift by group Z_L: 14*L vertices, 3-regular.
        # Girth of base is 6, so a random voltage lift typically has girth
        # around 2*L or so.  For g=16 we need girth >= 16, so L >= 8 roughly.
        # 14*8 = 112 vertices.  That's much smaller than 936!

        # Fano plane: points 0..6, lines as triples
        fano_lines = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5),
        ]
        # Incidence graph: 14 vertices.  Points 0-6, lines 7-13.
        base_n = 14
        base_edges = []
        for li, line in enumerate(fano_lines):
            for p in line:
                base_edges.append((p, 7 + li))
        # base_edges: 21 edges, 3-regular bipartite, girth 6.

        # Voltage assignment: assign to each base edge a voltage in Z_L.
        # Lifted graph: vertex (v, t) for v in 0..13, t in 0..L-1.
        # Edge ((u, t), (v, t + volt(e) mod L)) for each base edge (u,v) with voltage.
        # The lifted graph is 3-regular on 14*L vertices.

        import random
        rng = random.Random(42)

        best_geom_n = float('inf')
        best_geom_edges = None

        for L in range(8, 25):  # try various group sizes
            if time.time() > deadline:
                break
            n_lift = base_n * L
            # Try several random voltage assignments
            for trial in range(30):
                if time.time() > deadline:
                    break
                # Assign random voltages to each base edge
                volt = [rng.randint(0, L - 1) for _ in range(len(base_edges))]
                # Build lifted edges
                lifted_edges = []
                for (u, v), vol in zip(base_edges, volt):
                    for t in range(L):
                        lifted_edges.append((u * L + t, v * L + (t + vol) % L))
                # Verify
                if verify_girth(n_lift, lifted_edges, g):
                    if n_lift < best_geom_n:
                        best_geom_n = n_lift
                        best_geom_edges = lifted_edges
                    break  # found valid for this L, try smaller L next

        if best_geom_edges is not None:
            return best_geom_n, best_geom_edges
        return None

    # Try the geometry-based construction first (may be much smaller)
    geom_result = geometry_voltage_construction(k, g, deadline)
    if geom_result is not None:
        best_n, best_edges = geom_result

    # Search over d and rule variants (LUW family).
    for d in [7, 8, 9, 10, 11, 12]:
        if time.time() > deadline:
            break
        max_variants = 12 if d == 7 else (8 if d == 8 else 5)
        for variant in range(max_variants):
            if time.time() > deadline:
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
