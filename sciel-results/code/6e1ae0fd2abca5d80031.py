import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 130  # safety margin before 150 s kill

    # --- LUW construction (guaranteed valid, ~15000 vertices) ---
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # --- Circulant graph search ---
    # A cycle of length l in a circulant graph with step set S corresponds to
    # finding signs e_1..e_l in {+1,-1} and steps s_1..s_l in S such that
    # sum e_i * s_i = 0 (mod n), with no two consecutive terms cancelling
    # (e_i * s_i != -e_{i+1} * s_{i+1} mod n, i.e. the walk doesn't backtrack).
    # We check this via a linear system over Z/nZ.

    def circulant_girth_ok(n, S):
        """Return True if no cycle of length < g exists in the circulant graph."""
        # Precompute: for each step s in S, the set of signed steps {+s, -s} mod n
        # We need to check for each l in [3, g-1] whether there exist
        # (s_1, e_1), ..., (s_l, e_l) with s_i in S, e_i in {+1,-1}
        # such that sum e_i * s_i = 0 mod n and no backtracking.
        #
        # Backtracking condition: e_i * s_i + e_{i+1} * s_{i+1} != 0 mod n
        # i.e. the partial sum doesn't return to the previous vertex.
        #
        # Efficient check: enumerate all l-tuples of signed steps (from the set
        # T = {+s, -s : s in S}), check if their sum is 0 mod n, and check
        # no consecutive pair sums to 0.
        # |T| = 2*|S| = 2k = 10 for k=5.
        # For l=8, 10^8 = 100M — too slow in Python.
        #
        # Better: use the linear algebra approach.
        # For a given l, we want to know if there exist x_1..x_l in T (the signed step set)
        # such that sum x_i = 0 mod n and x_i + x_{i+1} != 0 for all i (cyclic).
        #
        # Alternative: use the fact that a cycle of length l exists iff there's a
        # non-trivial solution to the system. We can use a meet-in-the-middle or
        # simply check all combinations for small l.
        #
        # For l=3: check if any three signed steps sum to 0. |T|=10, C(10,3)*6 = 1200. Fast.
        # For l=4: C(10,4)*24 = 5040. Fast.
        # For l=5: C(10,5)*120 = 30240. Fast.
        # For l=6: C(10,6)*720 = 151200. OK.
        # For l=7: C(10,7)*5040 = 604800. OK.
        # For l=8: C(10,8)*40320 = 1612800. OK.
        # Total: ~2.3M checks per candidate. With n up to 600 and maybe 50 candidates, ~115M. Borderline.
        #
        # Even better: precompute for each l the set of all possible sums of l distinct
        # signed steps (allowing repetition since steps can repeat in a cycle? No, in a
        # simple cycle, consecutive vertices are distinct, but non-consecutive can repeat.
        # Actually in a cycle of length l, all l vertices are distinct, so all l edges
        # are distinct. But the same step value can appear multiple times in the cycle.
        #
        # Let me just do the direct enumeration for l=3..8.

        T = set()
        for s in S:
            T.add(s % n)
            T.add((-s) % n)
        T = list(T)
        # Remove 0 if present (shouldn't be, since s != 0)
        T = [t for t in T if t != 0]

        # For each l, check if any l-tuple from T (with replacement, since same step
        # can be used multiple times in a cycle) sums to 0 mod n with no backtracking.
        # But we need to be careful: a cycle of length l means l distinct vertices
        # v_0, v_1, ..., v_{l-1} with v_i adjacent to v_{i+1 mod l}.
        # The steps are d_i = v_{i+1} - v_i mod n, each in T.
        # Sum of all d_i = 0 mod n.
        # Backtracking: d_i + d_{i+1} != 0 mod n for all i (cyclic).
        # Also, all partial sums v_0, v_0+d_1, ..., v_0+d_1+...+d_{l-1} must be distinct.
        #
        # For efficiency, I'll check the sum condition and backtracking condition.
        # The distinct-vertex condition is implied by no backtracking for l >= 3?
        # Not exactly, but it's a necessary condition. Let me just check sum + backtracking.

        from itertools import product

        for l in range(3, g):
            # Check if there exist d_0, ..., d_{l-1} in T such that:
            # sum(d_i) = 0 mod n, and d_i + d_{i+1} != 0 mod n for all i (cyclic).
            # Use product with repetition.
            # For l=3: 10^3 = 1000
            # For l=4: 10^4 = 10000
            # For l=5: 10^5 = 100000
            # For l=6: 10^6 = 1M
            # For l=7: 10^7 = 10M  -- too slow
            # For l=8: 10^8 = 100M -- too slow
            #
            # Need a smarter approach for l >= 6.
            #
            # Meet-in-the-middle for l=6: split into two halves of 3.
            # For l=7: split into 3 and 4.
            # For l=8: split into 4 and 4.

            if l <= 5:
                for d in product(T, repeat=l):
                    if sum(d) % n != 0:
                        continue
                    # Check backtracking
                    ok = True
                    for i in range(l):
                        if (d[i] + d[(i + 1) % l]) % n == 0:
                            ok = False
                            break
                    if ok:
                        return False
            else:
                # Meet in the middle
                half1 = l // 2
                half2 = l - half1

                # Generate all sums of half1 elements from T
                sums1 = set()
                for d1 in product(T, repeat=half1):
                    sums1.add(sum(d1) % n)

                # For each tuple of half2 elements, check if there's a complement in sums1
                # Also need to check backtracking at the boundary.
                # This is tricky. Let me just do a full check for l=6,7,8 with
                # early termination.

                # For l=6: 10^6 = 1M iterations. Each does 6 additions + 6 checks. ~12M ops.
                # For l=7: 10^7 = 10M. ~140M ops. Too slow in Python.
                # For l=8: 10^8 = 100M. Way too slow.

                # Alternative: use the linear algebra approach.
                # A cycle of length l exists iff the system has a solution.
                # We can check this by computing the rank of a certain matrix.
                #
                # Actually, let me use a different approach: BFS on the state space
                # of (current_vertex, previous_vertex) to find the shortest cycle.
                # This is the non-backtracking BFS. For n=500, k=5, the state space
                # is n * (k-1) = 500 * 4 = 2000 states. BFS from each of the n*k/2
                # directed edges. Total: 2500 * 2000 = 5M states. Fast!

                # Let me just do non-backtracking BFS for all l.
                # Actually, let me redo the whole check using non-backtracking BFS.
                pass

        # Fallback: use non-backtracking BFS
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in S:
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i].append(j1)
                adj[i].append(j2)

        # Non-backtracking BFS from each directed edge
        # State: (current, prev)
        # Start from each directed edge (u, v)
        # BFS to find shortest path back to u without going back to prev.

        # For efficiency, do BFS from each vertex, tracking the previous vertex.
        # dist[v][prev] = shortest non-backtracking walk from (start, prev) to v.
        # But we need to find cycles, so we check when we return to start.

        # Simpler: for each start vertex s, do BFS in the state space (v, prev)
        # where prev is the previous vertex. Initialize with (s, neighbor) for each
        # neighbor of s. When we reach (s, prev2) for some prev2 != initial prev,
        # we have a cycle.

        # Actually, the standard approach:
        # For each directed edge (u, v), do BFS in the non-backtracking graph.
        # The non-backtracking graph has states (a, b) where a~b, and edges
        # (a, b) -> (b, c) where c != a.
        # A cycle of length l corresponds to a path of length l-1 in this graph
        # that returns to the starting state.

        # Let me just do BFS from each vertex in the original graph, tracking
        # the parent, and check for the shortest cycle through each vertex.

        # For each vertex s:
        #   BFS with state (v, parent) where parent is the vertex we came from.
        #   dist[(v, parent)] = distance from s.
        #   When we reach (s, parent2) for parent2 != initial parent, we have a cycle
        #   of length dist + 1.
        #   But we need to be careful: the initial state is (s, neighbor) for each
        #   neighbor of s. We want to find the shortest cycle through s.

        # This is O(n * n * k) per start vertex = O(n^2 * k) total. For n=500, k=5:
        # 500 * 500 * 5 = 1.25M. Fast!

        for s in range(n):
            # BFS in state space (v, prev)
            # prev is the vertex we came from (for non-backtracking)
            # dist[(v, prev)] = shortest non-backtracking walk from s to v via prev
            from collections import deque as dq
            dist = {}
            queue = dq()
            for nb in adj[s]:
                dist[(s, nb)] = 1
                queue.append((s, nb))

            found = False
            while queue and not found:
                v, prev = queue.popleft()
                d = dist[(v, prev)]
                if d >= g:
                    continue
                for nb2 in adj[v]:
                    if nb2 == prev:
                        continue  # non-backtracking
                    new_state = (nb2, v)
                    if new_state in dist:
                        continue
                    if nb2 == s:
                        # Found a cycle of length d + 1
                        if d + 1 < g:
                            return False
                        found = True
                        break
                    dist[new_state] = d + 1
                    queue.append(new_state)

        return True

    def find_circulant(k, g, max_n, deadline):
        # Moore bound for k=5, g=9 is 426
        # n must be even (5n must be even)
        # Try n from 426 to max_n, even values only
        random.seed(42)

        # Precompute valid step sets: |S| = k/2 = 3 steps (each gives 2 neighbours)
        # Wait, for k=5, we need 5 neighbours. Each step s gives neighbours at +s and -s.
        # So we need 5 distinct signed steps, which means either:
        #   - 2 full steps (4 neighbours) + 1 step where s = -s mod n (i.e., 2s = 0 mod n)
        #   - Or 5 steps where some coincide? No, they must be distinct neighbours.
        #
        # Actually, for a circulant graph to be k-regular, the step set S must satisfy:
        # |{s, -s mod n : s in S}| = k
        # For k=5 (odd), we need at least one s where s = -s mod n, i.e., 2s = 0 mod n.
        # This requires n to be even and s = n/2.
        # So S must include n/2, plus 2 other steps (each contributing 2 neighbours).
        # Total: 1 + 2*2 = 5 neighbours. Good.

        n_min = 426
        if n_min % 2 != 0:
            n_min += 1

        for n in range(n_min, max_n + 1, 2):
            if time.time() > deadline:
                return None

            # S must include n/2 (the "diameter" step)
            # Plus 2 other steps s1, s2 with 0 < s < n/2, s != n/2
            # And all of {s1, -s1, s2, -s2, n/2} must be distinct mod n
            # This is automatic if 0 < s1, s2 < n/2 and s1 != s2.

            # Try random pairs (s1, s2)
            candidates_per_n = 20
            for _ in range(candidates_per_n):
                if time.time() > deadline:
                    return None
                s1 = random.randint(1, n // 2 - 1)
                s2 = random.randint(1, n // 2 - 1)
                if s1 == s2:
                    continue
                S = [n // 2, s1, s2]
                # Verify degree: each vertex has neighbours at +n/2, -n/2 (same),
                # +s1, -s1, +s2, -s2. Since n/2 = -n/2 mod n, that's 1 + 2 + 2 = 5. Good.

                if circulant_girth_ok(n, S):
                    # Build edge list
                    edges = []
                    for i in range(n):
                        for s in S:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

        return None

    # --- Main logic ---
    # Try circulant search first (much smaller n)
    max_circulant_n = 600  # reasonable upper bound for search
    result = find_circulant(k, g, max_circulant_n, deadline)
    if result is not None:
        return result

    # Fallback: LUW construction
    return luw_construct(k, g)
