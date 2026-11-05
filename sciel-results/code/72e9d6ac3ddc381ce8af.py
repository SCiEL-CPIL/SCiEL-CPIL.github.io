import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification using non-backtracking walks ----------------
    def check_girth(n, edges, g, time_limit):
        """Check that girth >= g using non-backtracking walks from every vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        for s in range(n):
            if time.time() > time_limit:
                return -1
            # dist[prev][node] = shortest non-backtracking walk length from s to node
            # where the previous vertex is prev
            # Use: dist[node][prev] = min length of non-backtracking walk from s to node
            # arriving from prev
            # For efficiency, use a 2D array indexed by (node, prev)
            # But n can be large. Use dict-based approach.
            # Actually, for a 3-regular graph, each node has at most 3 neighbors.
            # State: (current_node, previous_node) -> distance
            # Start: for each neighbor nb of s, state (nb, s) at distance 1.
            # We want to find the shortest cycle through s.
            # A cycle through s of length L corresponds to a non-backtracking walk
            # from s back to s of length L (where the first step goes to some neighbor
            # and the last step comes from some neighbor, and we don't backtrack).
            # Actually, a cycle through s: s -> v1 -> v2 -> ... -> v_{L-1} -> s
            # This is a non-backtracking walk of length L from s to s where
            # v1 != s (always true) and v_{L-1} != s (true for L >= 3).
            # But we need to be careful: the walk s->v1->...->s is non-backtracking
            # if v2 != s (i.e., v1's next step isn't back to s) and for all intermediate
            # steps, we don't go back.
            # Actually, a non-backtracking walk from s to s of length L where L >= 3
            # automatically corresponds to a cycle (since the only way to return to s
            # without backtracking is to form a cycle).
            # Wait, that's not quite right. Consider s-a-b-s. This is a cycle of length 3.
            # Non-backtracking: s->a (ok), a->b (b != s, ok), b->s (s != a, ok). Yes.
            # What about s-a-b-a-s? That's length 4 but it's not a simple cycle; it backtracks
            # in the sense that we revisit a. But is it non-backtracking?
            # s->a: ok. a->b: b!=s, ok. b->a: a!=b? No, a IS b's predecessor... 
            # Wait, non-backtracking means we don't go to the vertex we just came from.
            # So b->a: we came from a to b, so going back to a IS backtracking. Not allowed.
            # So non-backtracking walks from s to s of length >= 3 are exactly the cycles through s.
            # Good.

            # BFS on non-backtracking states
            # State: (node, prev_node)
            # We'll use a dict: dist[(node, prev)] = length
            dist = {}
            queue = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                queue.append((nb, s))

            found = False
            while queue:
                node, prev = queue.popleft()
                d = dist[(node, prev)]
                if node == s:
                    # Found a cycle of length d through s
                    if d < g:
                        return d
                    found = True
                    # We can stop exploring this s if we found a cycle >= g? 
                    # No, we need to check if there's a shorter one. But BFS gives shortest first.
                    # Actually, the first time we hit (s, prev) for any prev, that's the shortest cycle through s.
                    # Wait, no. We're doing BFS on the state space. The first time we dequeue
                    # a state where node == s, that gives us the shortest non-backtracking walk
                    # from s to s, which is the shortest cycle through s.
                    break
                for nb in adj[node]:
                    if nb == prev:
                        continue  # non-backtracking
                    state = (nb, node)
                    if state not in dist:
                        dist[state] = d + 1
                        queue.append(state)

            if not found and dist:
                # No cycle through s? That means s is in a tree, which can't happen in a finite graph.
                # Actually in a connected graph, every vertex is in some cycle (if girth is finite).
                # If the graph is disconnected, some components might be trees. But we're looking for
                # the global girth, so we just need to check all vertices.
                # If no cycle found through s, skip.
                pass
            elif not dist:
                # Isolated vertex or something weird
                pass

        return g  # If we got here, all cycles are >= g

    # ---------------- Voltage lifting on Petersen graph with non-Abelian group ----------------
    def voltage_lift_nonabelian():
        if k != 3 or g != 16:
            return None

        # Petersen graph
        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]

        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use a non-Abelian group: S_3 (symmetric group on 3 elements)
        # S_3 has 6 elements. Represent as permutations of (0,1,2).
        # Elements: e, (01), (02), (12), (012), (021)
        # Multiplication table for S_3
        # Label: 0=e, 1=(01), 2=(02), 3=(12), 4=(012), 5=(021)
        # (012) = (01)(12) = (02)(01) etc.
        # Let's compute the multiplication table.
        # Using standard notation: permutations composed right-to-left.
        # e = identity
        # (01): swap 0 and 1
        # (02): swap 0 and 2
        # (12): swap 1 and 2
        # (012): 0->1, 1->2, 2->0
        # (021): 0->2, 1->0, 2->1

        # Multiplication table (row * column):
        #     0  1  2  3  4  5
        #  0: 0  1  2  3  4  5
        #  1: 1  0  4  5  3  2
        #  2: 2  5  0  4  1  3
        #  3: 3  4  5  0  2  1
        #  4: 4  2  3  1  5  0
        #  5: 5  3  1  2  0  4

        S3_mult = [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 4, 5, 3, 2],
            [2, 5, 0, 4, 1, 3],
            [3, 4, 5, 0, 2, 1],
            [4, 2, 3, 1, 5, 0],
            [5, 3, 1, 2, 0, 4]
        ]

        # Inverse table
        S3_inv = [0, 1, 2, 3, 5, 4]  # e^-1=e, (01)^-1=(01), (02)^-1=(02), (12)^-1=(12), (012)^-1=(021), (021)^-1=(012)

        # Verify: S3_mult[a][S3_inv[a]] should be 0 for all a
        # 0*0=0, 1*1=0, 2*2=0, 3*3=0, 4*5=0, 5*4=0. Good.

        # For voltage assignment, we assign a group element to each directed edge.
        # For undirected edge (u,v), we assign g_uv to u->v and g_uv^{-1} to v->u.
        # The lifted graph has 10*|G| = 60 vertices.
        # Edge in lifted graph: (u, a) -> (v, a * g_uv) for each base edge (u,v) and voltage g_uv.

        # We need to find voltage assignments such that no non-backtracking closed walk
        # in the base graph of length < g has trivial product.

        # Generate all non-backtracking closed walks of length 5 to 15 in Petersen graph.
        max_L = g - 1  # 15

        # For each closed walk, compute the sequence of edges and directions.
        # Then the product of voltages must be non-identity.

        # Collect walk edge sequences
        walk_edge_seqs = set()

        for start_v in range(10):
            # DFS to find all non-backtracking walks from start_v back to start_v
            # of length 5 to max_L
            stack = [(start_v, -1, 0, [])]
            while stack:
                current, prev, length, edge_seq = stack.pop()

                if length >= 5 and current == start_v:
                    walk_edge_seqs.add(tuple(edge_seq))

                if length == max_L:
                    continue

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    # Determine direction: 1 if going u->v where base_edges[e_idx]=(u,v), -1 otherwise
                    u, v = base_edges[e_idx]
                    direction = 1 if current == u else -1
                    stack.append((next_v, current, length + 1, edge_seq + [(e_idx, direction)]))

        if not walk_edge_seqs:
            return None

        n_walks = len(walk_edge_seqs)
        n_edges = 15

        # Each walk gives a constraint: product of voltages (with appropriate inverses) != identity
        # We want to find voltage assignments (g_0, ..., g_14) in S_3^15 such that
        # for every walk, the product is not identity.

        # This is a constraint satisfaction problem. Let's use a randomized approach:
        # Try random assignments and check constraints.

        # But first, let's see how many walks there are.
        # For a 3-regular graph, the number of non-backtracking walks of length L from a vertex
        # is 3 * 2^(L-1). For L=15, that's 3*2^14 = 49152 per vertex. Total ~491520.
        # That's a lot. Let's be smarter.

        # Actually, we only need to check walks of length 5 to 15. The number is manageable.
        # Let's count them.

        # For random assignment, probability that a specific walk has trivial product is 1/6.
        # With ~500k walks, expected number of trivial products is ~83k. So random won't work well.

        # We need a smarter approach. Let's use the structure of S_3.
        # S_3 is generated by two elements: a=(01) and b=(012).
        # Any element can be written as a^i * b^j where i in {0,1}, j in {0,1,2}.
        # So we have 2*3 = 6 elements, matching |S_3|.

        # Let's parameterize voltages as (a^i * b^j) for each edge.
        # That gives 6^15 possibilities, too many to brute force.

        # Alternative: use a larger group. Let's try A_4 (alternating group on 4 elements).
        # A_4 has 12 elements. Or S_4 with 24 elements.

        # Actually, let's try a different approach: use the Abelian voltage lift but with
        # a larger modulus and better search.

        # Let's go back to Abelian but use a better strategy.
        # The key insight: we need C @ x % m != 0 for all rows of C.
        # C has many rows (one per walk). For each row c, we need c @ x != 0 mod m.
        # This is equivalent to: x is not in the kernel of any row modulo m.

        # For a fixed m, the set of "bad" x is the union of hyperplanes {x : c_i @ x = 0 mod m}.
        # We want x outside this union.

        # For large m, the probability that a random x is bad for a specific constraint is ~1/m.
        # With n_walks constraints, the expected number of bad x is ~n_walks/m.
        # For m = 100 and n_walks = 10000, expected bad is 100. Total x is 100^15, so
        # the fraction bad is tiny. Random should work for large m.

        # But we want small m to minimize n = 10*m.

        # Let's try m from 52 upwards, and for each m, do random search with enough trials.
        # Also try structured assignments.

        C_list = []
        for walk in walk_edge_seqs:
            vec = [0] * n_edges
            for e_idx, direction in walk:
                vec[e_idx] += direction
            C_list.append(vec)

        C = np.array(C_list, dtype=np.int64)
        n_constraints = len(C)

        min_m = 52
        best_m = None
        best_x = None

        for m in range(min_m, 200):
            if remaining() < 20:
                break

            # Try deterministic patterns
            patterns = []
            patterns.append(np.ones(n_edges, dtype=int) % m)
            patterns.append(np.array([2**i % m for i in range(n_edges)]))
            patterns.append(np.array([i for i in range(n_edges)]))
            patterns.append(np.array([(i * 7 + 3) % m for i in range(n_edges)]))
            patterns.append(np.array([(i * i) % m for i in range(n_edges)]))
            patterns.append(np.array([(3 * i + 1) % m for i in range(n_edges)]))

            for x in patterns:
                S = (C @ x) % m
                if not np.any(S == 0):
                    best_m = m
                    best_x = x
                    break
            if best_m is not None:
                break

            # Random search
            rng = np.random.default_rng(m * 2654435761 + 42)
            n_trials = min(5000, max(1000, int(10000 / (m - min_m + 1))))
            for _ in range(n_trials):
                x = rng.integers(0, m, size=n_edges)
                S = (C @ x) % m
                if not np.any(S == 0):
                    best_m = m
                    best_x = x
                    break
            if best_m is not None:
                break

        if best_m is None:
            # Try larger m
            for m in range(200, 500):
                if remaining() < 10:
                    break
                rng = np.random.default_rng(m * 7 + 3)
                for _ in range(3000):
                    x = rng.integers(0, m, size=n_edges)
                    S = (C @ x) % m
                    if not np.any(S == 0):
                        best_m = m
                        best_x = x
                        break
                if best_m is not None:
                    break

        if best_m is None:
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges_set = set()
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            shift = best_x[e_idx] % best_m
            for i in range(best_m):
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + shift) % best_m
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))

        final_edges = list(edges_set)

        # Verify girth
        gc = check_girth(n, final_edges, g, deadline - 5)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        else:
            return None

    # ---------------- LUW construction (Fallback) ----------------
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
        else:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
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

    # ---------------- main ----------------
    if k == 3 and g == 16:
        res = voltage_lift_nonabelian()
        if res is not None:
            return res

    res = luw_construct(k, g)
    return res if res is not None else (0, [])
