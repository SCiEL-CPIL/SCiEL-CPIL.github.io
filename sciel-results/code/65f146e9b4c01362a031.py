import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Baseline: two K_{3,3} components, cycle lengths {4, 6} -> raw = 2
    def make_k33(offset):
        edges = []
        for u in range(3):
            for v in range(3):
                edges.append([offset + u, offset + 3 + v])
        return edges

    best_edges = make_k33(0) + make_k33(6)
    best_raw = 2

    # Fast cycle-length detection using adjacency matrix powers
    def count_distinct_cycle_lengths_fast(adj):
        """adj: list of frozensets. Returns (raw, lengths_set)."""
        n = len(adj)
        # Build adjacency matrix as list of bitmasks for speed
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask

        # A^2: number of length-2 walks from i to j (i.e., common neighbors)
        # A^2[i][j] = popcount(adj_mask[i] & adj_mask[j])
        # A^2 diagonal = degree of i

        # For 4-cycles: (A^4)[i][i] counts closed walks of length 4 from i to i.
        # A 4-cycle contributes 4 to the diagonal (4 starting points).
        # Also, back-and-forth walks i-a-i-a contribute: for each neighbor a,
        # there are 2 such walks (i-a-i-a and i-a-i-a going back). Actually:
        # Closed walks of length 4 at vertex i:
        #   - i-a-i-a-i: for each neighbor a, this is 1 walk. Count = deg(i).
        #     Wait: i->a->i->a->i. That's one specific walk. For each neighbor a, 1 walk.
        #   - i-a-b-a-i where a != b, a ~ i, b ~ a, b ~ i: this is a 4-cycle i-a-b-c-i? No.
        #     Let me think more carefully.
        #
        # A closed walk of length 4 at vertex i: i -> v1 -> v2 -> v3 -> i
        # where v1 ~ i, v2 ~ v1, v3 ~ v2, i ~ v3.
        #
        # Cases:
        # 1. v1 = v3, v2 = i: i->v1->i->v1->i. For each neighbor v1, 1 walk. Total: deg(i).
        # 2. v1 = v3, v2 != i: i->v1->v2->v1->i where v2 ~ v1, v2 != i.
        #    For each neighbor v1 of i, for each neighbor v2 of v1 (v2 != i), 1 walk.
        #    Sum over v1~i of (deg(v1) - 1).
        # 3. v1 != v3, v2 = i: i->v1->i->v3->i where v1~i, v3~i.
        #    For each pair (v1, v3) of neighbors of i (ordered), 1 walk. Total: deg(i)^2.
        #    But wait, v1 and v3 can be the same? If v1=v3, we get i->v1->i->v1->i which is case 1.
        #    So for v1 != v3: deg(i)*(deg(i)-1) walks.
        # 4. v1 != v3, v2 != i: i->v1->v2->v3->i where v1~i, v2~v1, v3~v2, v3~i, v1!=v3, v2!=i.
        #    This includes 4-cycles and other structures.
        #
        # This is getting complex. Let me just use matrix multiplication with bitmasks.

        # Actually, let me use a different approach: enumerate cycles by BFS from each vertex
        # but with a tight depth limit and early exit.

        # Better: use the fact that for triangle-free graphs, we can count cycles efficiently.
        # Let me just do a bounded DFS for cycles of length 4, 5, 6, 7, 8.

        found_lengths = set()

        # For each vertex as start, do DFS to find simple cycles
        # To avoid duplicates, only count cycles where start is the minimum vertex in the cycle

        def find_cycles_from(start, max_len=8):
            """Find all simple cycles starting and ending at start, where start is min vertex."""
            cycles = []

            def dfs(current, visited, path_len):
                if path_len >= max_len:
                    return
                for neighbor in adj[current]:
                    if neighbor < start:
                        continue
                    if neighbor == start and path_len >= 3:
                        cycles.append(path_len)
                    elif neighbor not in visited and neighbor > start:
                        visited.add(neighbor)
                        dfs(neighbor, visited, path_len + 1)
                        visited.remove(neighbor)

            visited = {start}
            dfs(start, visited, 1)
            return cycles

        for start in range(n):
            if time.time() > deadline:
                return None, None
            cycles = find_cycles_from(start)
            for c in cycles:
                found_lengths.add(c)

        if not found_lengths:
            return 0, set()
        return len(found_lengths), found_lengths

    # Systematic search: enumerate 3-regular bipartite graphs on 6+6 vertices
    # Left: 0-5, Right: 6-11
    # Each left vertex connects to exactly 3 right vertices
    # Each right vertex connects to exactly 3 left vertices
    # Represent by 6 permutations (one for each left vertex) of subsets of size 3

    import itertools
    import random

    # All possible 3-subsets of {6,7,8,9,10,11}
    right_vertices = [6, 7, 8, 9, 10, 11]
    all_subsets = list(itertools.combinations(right_vertices, 3))
    # 20 subsets total

    # For a 3-regular bipartite graph, we need to assign to each left vertex (0-5)
    # a 3-subset of right vertices such that each right vertex appears exactly 3 times.

    # This is equivalent to a 6x6 (0,1)-matrix with row sums 3 and column sums 3.

    # Let's enumerate using backtracking: assign subsets to left vertices 0,1,2,3,4,5
    # such that column sums are all 3.

    def search_3regular_bipartite():
        """Search for 3-regular bipartite graphs on 6+6 with minimal distinct cycle lengths."""
        nonlocal best_edges, best_raw, deadline

        # Precompute: for each subset, which right vertices it covers
        subset_to_mask = {}
        for s in all_subsets:
            mask = 0
            for v in s:
                mask |= (1 << (v - 6))  # right vertex index 0-5
            subset_to_mask[s] = mask

        # We need to pick 6 subsets (with repetition allowed) such that
        # each right vertex (column) is covered exactly 3 times.

        # Use backtracking: for left vertices 0..5, pick a subset
        # Track column counts

        col_counts = [0] * 6
        assignment = [None] * 6
        found = []

        def backtrack(left_idx):
            if time.time() > deadline:
                return
            if left_idx == 6:
                if all(c == 3 for c in col_counts):
                    # Valid 3-regular bipartite graph
                    # Check if it has any cycles and count distinct lengths
                    adj = [set() for _ in range(12)]
                    for i in range(6):
                        for v in assignment[i]:
                            adj[i].add(v)
                            adj[v].add(i)

                    # Quick check: does it have a cycle?
                    # In a 3-regular bipartite graph on 12 vertices, it always has cycles
                    # unless it's a tree, but a tree on 12 vertices has 11 edges,
                    # while 3-regular on 12 vertices has 18 edges. So always has cycles.

                    raw, lengths = count_distinct_cycle_lengths_fast(adj)
                    if raw is not None and 0 < raw < best_raw:
                        found.append((raw, lengths, assignment[:]))
                        if raw == 1:
                            return True  # Can't do better than 1
                    return False

            for subset in all_subsets:
                mask = subset_to_mask[subset]
                # Check if adding this subset keeps column counts <= 3
                valid = True
                for j in range(6):
                    if (mask & (1 << j)) and col_counts[j] >= 3:
                        valid = False
                        break
                if not valid:
                    continue

                # Apply
                for j in range(6):
                    if mask & (1 << j):
                        col_counts[j] += 1
                assignment[left_idx] = subset

                if backtrack(left_idx + 1):
                    # Undo before returning
                    for j in range(6):
                        if mask & (1 << j):
                            col_counts[j] -= 1
                    return True

                # Undo
                for j in range(6):
                    if mask & (1 << j):
                        col_counts[j] -= 1

            return False

        backtrack(0)
        return found

    # Run the systematic search
    results = search_3regular_bipartite()

    if results:
        # Pick the best result
        results.sort(key=lambda x: x[0])
        best_raw, _, best_assignment = results[0]

        # Convert assignment to edges
        best_edges = []
        for i in range(6):
            for v in best_assignment[i]:
                best_edges.append([i, v])

    # If we found raw=1, great. Otherwise, also try some random non-bipartite constructions
    # but the bipartite search should be thorough enough.

    # Additional: try to find graphs with only 4-cycles (no 6-cycles)
    # This is the key: in a 3-regular bipartite graph, 4-cycles correspond to
    # pairs of left vertices sharing 2 right neighbors. 6-cycles are harder to avoid.

    # Let's also try a targeted search: modify the K_{3,3} structure slightly
    # to eliminate 6-cycles while keeping 4-cycles.

    # Actually, let me try one more thing: the complete bipartite graph K_{3,3} has
    # 3 4-cycles and 3 6-cycles. If we can find a 3-regular bipartite graph on 12
    # vertices with no 6-cycles, we'd have raw=1.

    # The existence of such a graph is equivalent to a 6x6 (0,1)-matrix with row/col
    # sums 3 and no 2x2 submatrix with all entries 1 in a "6-cycle pattern".

    # This is related to the concept of "C4-free" or "C6-free" bipartite graphs.
    # A bipartite graph with no 4-cycles has the property that any two left vertices
    # share at most 1 right neighbor. With 6 left vertices each of degree 3, the total
    # number of pairs of left vertices is C(6,2)=15, and each pair can share at most 1
    # right neighbor. The total number of shared-neighbor pairs is sum over right vertices
    # of C(deg(r), 2) = 6*C(3,2) = 18. But 18 > 15, so by pigeonhole, some pair of left
    # vertices must share at least 2 right neighbors, creating a 4-cycle. So 4-cycles are
    # unavoidable. But we can potentially avoid 6-cycles.

    # For now, the systematic search above should find the best 3-regular bipartite graph.

    # Final validation of best_edges
    def validate(edges):
        if not edges:
            return False
        # No loops, no duplicates
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if not isinstance(u, int) or not isinstance(v, int):
                return False
            if u < 0 or u >= 12 or v < 0 or v >= 12:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)

        # Degree check
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(12):
            if len(adj[i]) < 3:
                return False

        # Triangle-free check
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False

        # Has a cycle
        raw, lengths = count_distinct_cycle_lengths_fast(adj)
        if raw is None or raw == 0:
            return False

        return True

    if not validate(best_edges):
        # Fall back to baseline
        best_edges = make_k33(0) + make_k33(6)
        best_raw = 2

    # Convert to proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])

    return result
