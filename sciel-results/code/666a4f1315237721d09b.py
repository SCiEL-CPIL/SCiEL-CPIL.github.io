import time
import random
import sys
from collections import defaultdict
from scipy.sparse.csgraph import maximum_bipartite_matching
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))

    def can_merge(L1, R1, L2, R2):
        for u in L1:
            for v in R2:
                if not adj[u][v]:
                    return False
        for u in L2:
            for v in R1:
                if not adj[u][v]:
                    return False
        return True

    def greedy_partition(remaining_edges, seed_order=None):
        rem = set(remaining_edges)
        bics = []
        if not rem:
            return bics
        # Pick edges in a specific order for diversity
        if seed_order is not None:
            ordered = sorted(rem, key=lambda e: seed_order.get(e, 0))
        else:
            ordered = sorted(rem)
        for edge in ordered:
            if edge not in rem:
                continue
            u, v = edge
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for x in range(n):
                    if x in L:
                        continue
                    # x must be connected to all in R and edges must be in rem
                    ok = True
                    for y in R:
                        if not adj[x][y]:
                            ok = False
                            break
                        e = (min(x, y), max(x, y))
                        if e not in rem:
                            ok = False
                            break
                    if ok:
                        L.add(x)
                        changed = True
                # Try adding to R
                for y in range(n):
                    if y in R:
                        continue
                    ok = True
                    for x in L:
                        if not adj[x][y]:
                            ok = False
                            break
                        e = (min(x, y), max(x, y))
                        if e not in rem:
                            ok = False
                            break
                    if ok:
                        R.add(y)
                        changed = True
            b_edges = set((min(u, v), max(u, v)) for u in L for v in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

    def solve_min_cover(bicliques):
        """Given a list of bicliques that partition the edges, find the minimum
        number of bicliques by merging compatible ones. This is equivalent to
        finding a minimum vertex cover in the compatibility graph, which is
        NP-hard in general, but for small k we can use the bipartite matching
        approach on the complement: maximum matching in the incompatibility
        graph gives us the minimum number of bicliques needed.
        
        Actually, the correct approach: we want to merge as many bicliques as
        possible. Two bicliques can be merged if they are compatible. We want
        to find the maximum set of pairwise-compatible bicliques to merge into
        one, plus the rest. This is a clique cover problem on the compatibility
        graph.
        
        For small k, we can use the following: the minimum number of bicliques
        after merging equals the size of the minimum clique cover of the
        compatibility graph. By the complement, this equals the chromatic
        number of the incompatibility graph.
        
        For k <= 15, we can try all possible assignments of bicliques to groups
        using backtracking with pruning.
        """
        k = len(bicliques)
        if k <= 1:
            return bicliques

        # Build compatibility matrix
        compat = [[False] * k for _ in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                compat[i][j] = compat[j][i] = can_merge(L1, R1, L2, R2)

        # Try to find minimum clique cover using backtracking
        # State: assignment of each biclique to a group (clique)
        best = k  # Start with no merging
        best_assignment = list(range(k))

        def backtrack(idx, groups, current_size):
            nonlocal best, best_assignment
            if current_size >= best:
                return
            if idx == k:
                if current_size < best:
                    best = current_size
                    best_assignment = groups[:]
                return

            # Try assigning biclique idx to an existing group
            for g in range(current_size):
                # Check if idx is compatible with all in group g
                ok = True
                for j in range(k):
                    if groups[j] == g:
                        if not compat[idx][j]:
                            ok = False
                            break
                if ok:
                    groups[idx] = g
                    backtrack(idx + 1, groups, current_size)

            # Try creating a new group
            if current_size < best - 1:  # Pruning
                groups[idx] = current_size
                backtrack(idx + 1, groups, current_size + 1)

        groups = [-1] * k
        backtrack(0, groups, 0)

        # Reconstruct merged bicliques from best_assignment
        group_map = defaultdict(list)
        for i in range(k):
            group_map[best_assignment[i]].append(i)

        merged = []
        for g, members in group_map.items():
            L = set()
            R = set()
            for i in members:
                L.update(bicliques[i][0])
                R.update(bicliques[i][1])
            merged.append([list(L), list(R)])

        return merged

    def validate_and_convert(bicliques):
        covered_edges = set()
        for L, R in bicliques:
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered_edges:
                        return None  # Overlap
                    covered_edges.add(e)
                    if not adj[u][v]:
                        return None  # Not an edge
        if covered_edges != edge_set:
            return None  # Missing edges
        result = []
        for L, R in bicliques:
            result.append([[int(x) for x in L], [int(x) for x in R]])
        return result

    # Strategy: Try multiple initial partitions with different edge orderings,
    # then solve the minimum cover problem for each.
    best_result = None
    best_count = float('inf')

    # First, try the standard greedy
    initial = greedy_partition(edge_set)
    merged = solve_min_cover(initial)
    result = validate_and_convert(merged)
    if result is not None:
        best_result = result
        best_count = len(result)

    # Try different random orderings for diversity
    random.seed(42)
    num_tries = 20
    for attempt in range(num_tries):
        if time.time() > deadline:
            break
        # Random edge ordering
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        seed_order = {e: i for i, e in enumerate(edge_list)}
        initial = greedy_partition(edge_set, seed_order)
        if len(initial) >= best_count:
            continue  # Can't do better
        merged = solve_min_cover(initial)
        result = validate_and_convert(merged)
        if result is not None and len(result) < best_count:
            best_result = result
            best_count = len(result)

    if best_result is None:
        # Fallback: return individual edges as bicliques (valid but large)
        best_result = [[[u], [v]] for u, v in edge_set]

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
