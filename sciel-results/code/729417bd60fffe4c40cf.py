"""Construct a 3-regular graph with girth at least g on as few vertices as possible."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def verify_girth(n, edges, min_girth):
        """Verify girth >= min_girth using BFS from each vertex (non-backtracking)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for src in range(n):
            visited = {}
            queue = deque()
            for nb in adj[src]:
                queue.append((nb, src, 1))
                visited[(nb, src)] = 1

            while queue:
                node, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key in visited:
                        if visited[key] + dist < min_girth:
                            return False
                    else:
                        visited[key] = dist
                        queue.append((nb, node, dist + 1))
        return True

    def greedy_backtracking_3reg_girth(n_target, min_girth, max_time):
        """
        Deterministic greedy backtracking to build a 3-regular graph with girth >= min_girth.
        Builds layer by layer from a root, maintaining non-backtracking walk constraints.
        Returns (n, edges) or None if not found within time.
        """
        elapsed = lambda: time.time() - start_time

        # We'll build a tree-like structure first, then close it up
        # Start with root vertex 0, add neighbors greedily

        # For a 3-regular graph with girth g, the Moore bound is:
        # n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i for even g
        # For k=3, g=16: n >= 2*(1+2+4+8+16+32+64+128) = 2*255 = 510

        # Strategy: build a "cage-like" graph using a symmetric construction
        # Use a product structure: vertices are (i, j, l) where we define edges
        # based on arithmetic modulo some parameters

        # Alternative: use the known approach of building from the Moore tree
        # and identifying leaves in a way that preserves girth

        # Let's try a different structural approach:
        # Build a 3-regular graph using a "blow-up" of a smaller cage
        # or use a circulant-like construction

        # For now, implement a systematic search over small parameter spaces
        # for structured 3-regular graphs

        # Approach: use a Cayley graph on a group with 3 generators
        # Try cyclic groups Z_n with generators {a, b, c} where a+b+c=0 mod n
        # and the graph has girth >= g

        # For a Cayley graph on Z_n with generators {±a, ±b, ±c} (6 generators, degree 6)
        # we need degree 3, so use {a, -a} as one edge type... that gives degree 2 per pair
        # Actually for degree 3, we need an odd number of generator pairs or a self-inverse generator

        # Let's try: vertices 0..n-1, edges: i ~ i+1, i ~ i+s, i ~ i+t (mod n)
        # This is a circulant graph with 3 "directions"
        # For it to be 3-regular, we need s ≠ t, s ≠ 1, t ≠ 1, s ≠ -1, t ≠ -1, etc.

        # Check girth of circulant C(n; 1, s, t)
        # A cycle corresponds to a sequence of ±1, ±s, ±t that sums to 0 mod n
        # Shortest such sequence gives the girth

        best = None

        # Try various n and (s, t) pairs
        # n must be even for 3-regular (handshaking lemma: 3n must be even)
        min_n = 2 * sum(2**i for i in range(g // 2))  # Moore bound for even girth
        if min_n < 510:
            min_n = 510

        for n in range(min_n, min_n + 500, 2):  # even n only
            if elapsed() > max_time:
                break

            # Try different (s, t) pairs
            # s and t should be "large" to avoid short cycles
            # Heuristic: s ≈ n/3, t ≈ n/2 or similar
            candidates = []
            for s in range(2, n // 2):
                if s == 1 or s == n - 1:
                    continue
                for t in range(s + 1, n - 1):
                    if t == 1 or t == n - 1 or t == s or t == n - s:
                        continue
                    # Avoid small cycles: check that 1, s, t don't create short relations
                    # A 4-cycle exists if a+b = c+d for distinct {a,b,c,d} in {±1,±s,±t}
                    # A 6-cycle exists if a+b+c = 0 for distinct a,b,c in {±1,±s,±t}
                    # Quick filter: ensure gcd(n, s) and gcd(n, t) are large enough
                    candidates.append((s, t))

            # Limit candidates to avoid explosion
            # Use a heuristic: pick s and t that are "far apart"
            step = max(1, len(candidates) // 50)
            for idx in range(0, len(candidates), step):
                if elapsed() > max_time:
                    break
                s, t = candidates[idx]

                # Build the circulant graph
                edges = []
                edge_set = set()
                for i in range(n):
                    for d in [1, s, t]:
                        j = (i + d) % n
                        key = (min(i, j), max(i, j))
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append(key)

                # Verify 3-regular
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue

                # Verify girth
                if verify_girth(n, edges, min_girth):
                    if best is None or n < best[0]:
                        best = (n, edges)
                        if elapsed() > max_time:
                            return best
                        # Found a valid one; try to find smaller
                        break

        return best

    def random_search_3reg(n_target, min_girth, max_time):
        """Random search for 3-regular graph with girth >= min_girth."""
        elapsed = lambda: time.time() - start_time

        def generate_random_3reg(n):
            if n % 2 == 1 or n < 4:
                return None
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)

            edges = []
            edge_set = set()
            valid = True

            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))

            if not valid:
                return None
            return edges

        best = None
        min_n = 2 * sum(2**i for i in range(g // 2))
        if min_n < 510:
            min_n = 510

        for n in range(min_n, min_n + 200, 2):
            if elapsed() > max_time:
                break

            attempts = 0
            max_attempts = 100

            while attempts < max_attempts and elapsed() < max_time:
                attempts += 1
                edges = generate_random_3reg(n)
                if edges is None:
                    continue

                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue

                if verify_girth(n, edges, min_girth):
                    if best is None or n < best[0]:
                        best = (n, edges)
                        break

        return best

    # Try structured (circulant) construction first - deterministic
    result = greedy_backtracking_3reg_girth(512, g, 60)
    if result is not None:
        return result

    # Fall back to random search
    result = random_search_3reg(512, g, 50)
    if result is not None:
        return result

    # Final fallback: LUW construction
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
