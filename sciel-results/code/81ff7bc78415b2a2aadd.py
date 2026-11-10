import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency list
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

def is_valid(colors):
    """Check: adjacent differ + every color-pair subgraph is a forest."""
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    # For each color pair, check forest via union-find
    pair_edges = defaultdict(list)
    for u, v in EDGES:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            pair_edges[key].append((u, v))
    for edges in pair_edges.values():
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
    return True

def construct():
    start = time.time()
    deadline = start + 80.0

    # Precompute: for each edge, the set of color-pairs that would be formed
    # if we assign colors. We'll use a smarter approach.

    # Strategy: The graph is 4-regular. Let's try to find a small number of
    # colors. Key insight: if we use k colors, each color-pair subgraph must be
    # a forest, meaning for each pair (a,b), the edges between vertices colored
    # a and b must form a forest.

    # Approach: Use a constraint-based search with good heuristics.
    # We'll try k=3, then k=4, etc.

    # First, let's understand the graph structure better.
    # It looks like a toroidal grid or similar structure.
    # Let's try a systematic approach.

    # For k colors, we need to assign colors to 50 vertices such that:
    # 1. Adjacent vertices have different colors
    # 2. For each pair of colors, the induced subgraph is a forest

    # A key observation: if we use k colors, there are k*(k-1)/2 color pairs.
    # Each edge belongs to exactly one color pair. The total number of edges is 100.
    # For each color pair, the subgraph must be a forest, so it can have at most
    # (number of vertices of those two colors) - (number of connected components) edges.

    # Let's try a backtracking search with constraint propagation.

    def solve_k(k):
        """Try to find a valid k-coloring using backtracking."""
        nonlocal deadline
        colors = [-1] * N
        # Order vertices by degree (all 4 here) but use a good ordering
        # Try to color in an order that maximizes constraints early
        order = list(range(N))

        # For efficiency, precompute for each vertex the constraints
        # When we assign color c to vertex v, for each neighbor u:
        #   - u cannot have color c
        #   - the edge (u,v) with colors (color_u, c) must not create a cycle
        #     in the subgraph of that color pair

        # We'll use a simpler but effective approach:
        # Assign colors one by one, checking validity incrementally.

        # To check if adding an edge to a color-pair creates a cycle,
        # we maintain union-find per color-pair.

        # For k=3, let's be smart about it.

        if k <= 3:
            # Specialized search for small k
            return solve_small_k(k, deadline)

        # General backtracking for larger k
        # Use forward checking
        def backtrack(idx):
            nonlocal deadline
            if time.time() > deadline:
                return False
            if idx == N:
                return True

            v = order[idx]
            # Try each color
            for c in range(k):
                # Check if color c is valid for vertex v
                valid = True
                for u in ADJ[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue

                # Tentatively assign
                colors[v] = c

                # Check that no color-pair subgraph has a cycle
                # For each neighbor u, the edge (u,v) with colors (colors[u], c)
                # must not create a cycle in that color-pair's subgraph
                # We check by seeing if u and v are already connected in the
                # subgraph of colors (colors[u], c)
                # This is expensive to maintain, so let's just do a full check
                # but only on the affected edges.

                # Actually, let's just do a quick check: for each neighbor u,
                # check if there's a path from u to v using only vertices with
                # colors in {colors[u], c} (excluding the direct edge).
                # This is BFS/DFS which is O(n) per edge, so O(deg*n) per assignment.
                # For 50 vertices, this is fine.

                ok = True
                for u in ADJ[v]:
                    cu = colors[u]
                    if cu == c:
                        ok = False
                        break
                    # Check if u and v are connected in the subgraph induced by
                    # vertices with colors in {cu, c}, excluding the edge (u,v)
                    # We do BFS from u avoiding v, using only vertices with color cu or c
                    visited = set()
                    queue = deque([u])
                    visited.add(u)
                    while queue:
                        cur = queue.popleft()
                        for nb in ADJ[cur]:
                            if nb == v:
                                continue  # skip the direct edge
                            if nb in visited:
                                continue
                            if colors[nb] in (cu, c):
                                visited.add(nb)
                                queue.append(nb)
                    if v in visited:
                        ok = False
                        break
                    if not ok:
                        break

                if not ok:
                    colors[v] = -1
                    continue

                if backtrack(idx + 1):
                    return True

                colors[v] = -1

            return False

        if backtrack(0):
            return colors[:]
        return None

    def solve_small_k(k, deadline):
        """Specialized solver for k=2 or k=3."""
        if k == 2:
            # Bipartite check
            colors = [-1] * N
            for start in range(N):
                if colors[start] != -1:
                    continue
                colors[start] = 0
                queue = deque([start])
                while queue:
                    v = queue.popleft()
                    for u in ADJ[v]:
                        if colors[u] == -1:
                            colors[u] = 1 - colors[v]
                            queue.append(u)
                        elif colors[u] == colors[v]:
                            return None
            # Check for bicolored cycles
            if is_valid(colors):
                return colors
            return None

        if k == 3:
            # Try all 3^50 is too much. Use backtracking with good ordering.
            # Order: color vertices in an order that creates constraints early.
            # Use a greedy + backtracking approach.

            colors = [-1] * N

            # Try to find a valid 3-coloring using backtracking
            # with constraint propagation (forward checking)

            # Order vertices: try to use a good ordering
            # For a 4-regular graph, let's try ordering by some heuristic
            # Actually, let's try multiple orderings

            def bt(idx, colors, deadline):
                if time.time() > deadline:
                    return False
                if idx == N:
                    return True

                v = idx
                # Try colors 0, 1, 2
                for c in range(3):
                    # Check neighbor constraints
                    valid = True
                    for u in ADJ[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue

                    colors[v] = c

                    # Check for bicolored cycles involving vertex v
                    # For each neighbor u, check if edge (u,v) creates a cycle
                    ok = True
                    for u in ADJ[v]:
                        cu = colors[u]
                        # Check if there's a path from u to v in the subgraph
                        # induced by colors {cu, c}, not using the direct edge
                        visited = set()
                        queue = deque([u])
                        visited.add(u)
                        while queue and ok:
                            cur = queue.popleft()
                            for nb in ADJ[cur]:
                                if nb == v:
                                    continue
                                if nb in visited:
                                    continue
                                if colors[nb] in (cu, c):
                                    visited.add(nb)
                                    queue.append(nb)
                        if v in visited:
                            ok = False
                            break

                    if not ok:
                        colors[v] = -1
                        continue

                    if bt(idx + 1, colors, deadline):
                        return True

                    colors[v] = -1

                return False

            # Try with natural ordering
            if bt(0, colors, deadline):
                return colors[:]

            # Try with a different ordering: reverse
            colors = [-1] * N

            def bt_rev(idx, colors, deadline):
                if time.time() > deadline:
                    return False
                if idx == N:
                    return True
                v = N - 1 - idx
                for c in range(3):
                    valid = True
                    for u in ADJ[v]:
                        if colors[u] == c:
                            valid = False
                            break
                    if not valid:
                        continue
                    colors[v] = c
                    ok = True
                    for u in ADJ[v]:
                        cu = colors[u]
                        visited = set()
                        queue = deque([u])
                        visited.add(u)
                        while queue and ok:
                            cur = queue.popleft()
                            for nb in ADJ[cur]:
                                if nb == v:
                                    continue
                                if nb in visited:
                                    continue
                                if colors[nb] in (cu, c):
                                    visited.add(nb)
                                    queue.append(nb)
                        if v in visited:
                            ok = False
                            break
                    if not ok:
                        colors[v] = -1
                        continue
                    if bt_rev(idx + 1, colors, deadline):
                        return True
                    colors[v] = -1
                return False

            if bt_rev(0, colors, deadline):
                return colors[:]

            return None

    # Try k=3 first
    result = solve_k(3)
    if result is not None:
        assert is_valid(result)
        return [int(c) for c in result]

    # Try k=4
    result = solve_k(4)
    if result is not None:
        assert is_valid(result)
        return [int(c) for c in result]

    # Fallback: k=5 with pattern
    for pattern in [
        [i % 5 for i in range(N)],
        [(i // 10) * 2 + (i % 10) % 2 for i in range(N)],
    ]:
        if is_valid(pattern):
            return [int(c) for c in pattern]

    # Ultimate fallback
    return list(range(N))

