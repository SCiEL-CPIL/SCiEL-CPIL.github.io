"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque
import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    # Precompute all-pairs shortest paths (unweighted)
    dist = [[n * 2 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        queue = deque([i])
        while queue:
            cur = queue.popleft()
            for nb in adj[cur]:
                if dist[cur][nb] > dist[cur][cur] + 1:
                    dist[cur][nb] = dist[cur][cur] + 1
                    queue.append(nb)

    # Precompute common neighbors for length-2 path check
    common_neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common_neighbors[u] |= adj[v]
                common_neighbors[v] |= adj[u]
                common_neighbors[u].discard(u)
                common_neighbors[v].discard(v)

    def find_path_bfs(u, v, forbidden):
        """BFS from u to v avoiding forbidden vertices."""
        if u == v:
            return None
        parent = [-1] * n
        parent[u] = u
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                path = []
                c = v
                while c != u:
                    path.append(c)
                    c = parent[c]
                path.append(u)
                path.reverse()
                return path
            for nb in adj[cur]:
                if parent[nb] == -1 and nb not in forbidden:
                    parent[nb] = cur
                    queue.append(nb)
        return None

    def find_path_bfs_shortest(u, v, forbidden):
        """BFS returning shortest path, with tie-breaking by preferring vertices with higher degree."""
        if u == v:
            return None
        parent = [-1] * n
        parent[u] = u
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                path = []
                c = v
                while c != u:
                    path.append(c)
                    c = parent[c]
                path.append(u)
                path.reverse()
                return path
            neighbors = sorted([nb for nb in adj[cur] if parent[nb] == -1 and nb not in forbidden],
                               key=lambda x: degrees[x], reverse=True)
            for nb in neighbors:
                parent[nb] = cur
                queue.append(nb)
        return None

    def check_and_get_paths(B, num_restarts=200):
        """Try to find vertex-disjoint paths for all pairs in B using bipartite matching for length-2 paths."""
        B_set = set(B)
        k = len(B_set)
        if k < 2:
            return False, None

        pairs = list(itertools.combinations(sorted(B_set), 2))
        num_pairs = len(pairs)

        # Precompute for each pair: shortest path length in G - (B - {u,v})
        pair_costs = []
        for u, v in pairs:
            forbidden = B_set - {u, v}
            d = n * 2
            if dist[u][v] <= n:
                parent = [-1] * n
                parent[u] = u
                queue = deque([u])
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    if cur == v:
                        found = True
                        break
                    for nb in adj[cur]:
                        if parent[nb] == -1 and nb not in forbidden:
                            parent[nb] = cur
                            queue.append(nb)
                if found:
                    d = 0
                    c = v
                    while c != u:
                        c = parent[c]
                        d += 1
                else:
                    d = n * 2
            pair_costs.append((d, u, v))

        pair_costs.sort()

        for attempt in range(num_restarts):
            used_interiors = set()
            paths = {}
            success = True

            if attempt == 0:
                ordered_pairs = [(u, v) for _, u, v in pair_costs]
            elif attempt == 1:
                ordered_pairs = [(u, v) for _, u, v in reversed(pair_costs)]
            else:
                ordered_pairs = [(u, v) for _, u, v in pair_costs]
                random.shuffle(ordered_pairs)

            for u, v in ordered_pairs:
                forbidden = (B_set - {u, v}) | used_interiors
                path = find_path_bfs(u, v, forbidden)
                if path is None:
                    path = find_path_bfs_shortest(u, v, forbidden)
                if path is None:
                    success = False
                    break
                paths[(u, v)] = path
                for node in path[1:-1]:
                    used_interiors.add(node)

            if success:
                return True, paths

        return False, None

    def check_with_matching(B):
        """Use bipartite matching to find length-2 paths for all pairs simultaneously.
        This is faster and more reliable for dense graphs where most pairs have common neighbors."""
        B_set = set(B)
        k = len(B_set)
        if k < 2:
            return False, None

        pairs = list(itertools.combinations(sorted(B_set), 2))
        num_pairs = len(pairs)
        non_branches = set(range(n)) - B_set

        # Build bipartite graph: left = pairs, right = non-branch vertices
        # Edge (pair_i, vertex_v) exists if v is a common neighbor of the pair
        # and v is not a branch

        # For each pair, compute which non-branch vertices can serve as middle
        pair_to_mid = [set() for _ in range(num_pairs)]
        for i, (u, v) in enumerate(pairs):
            candidates = common_neighbors[u] & common_neighbors[v] & non_branches
            # Also need to ensure the path u-v-mid is valid (u-mid and mid-v edges exist)
            # common_neighbors already ensures this
            pair_to_mid[i] = candidates

        # Check Hall's condition: for every subset of pairs, the union of their middle candidates must be at least as large
        # This is a necessary condition. We'll use maximum bipartite matching to check if a perfect matching exists.

        # Build adjacency list for matching
        # Left side: 0..num_pairs-1
        # Right side: 0..len(non_branches)-1
        non_branch_list = list(non_branches)
        mid_to_idx = {v: i for i, v in enumerate(non_branch_list)}
        num_mids = len(non_branch_list)

        if num_mids < num_pairs:
            return False, None

        # Build bipartite graph adjacency
        bipartite_adj = [[] for _ in range(num_pairs)]
        for i in range(num_pairs):
            for mid in pair_to_mid[i]:
                bipartite_adj[i].append(mid_to_idx[mid])

        # Use maximum_bipartite_matching from scipy
        # It expects a sparse matrix or adjacency list
        # Let's build a sparse matrix
        from scipy.sparse import csr_matrix
        row = []
        col = []
        data = []
        for i in range(num_pairs):
            for j in bipartite_adj[i]:
                row.append(i)
                col.append(j)
                data.append(1)

        if not row:
            return False, None

        A = csr_matrix((data, (row, col)), shape=(num_pairs, num_mids))
        match = maximum_bipartite_matching(A)

        # match is a matrix where match[i, j] = 1 if left i is matched to right j
        # Check if all pairs are matched
        matched_pairs = 0
        pair_to_assigned_mid = {}
        for i in range(num_pairs):
            matches = np.where(match[i, :])[0]
            if len(matches) > 0:
                matched_pairs += 1
                pair_to_assigned_mid[i] = non_branch_list[matches[0]]
            else:
                return False, None

        if matched_pairs != num_pairs:
            return False, None

        # All pairs have a unique middle vertex
        paths = {}
        for i, (u, v) in enumerate(pairs):
            mid = pair_to_assigned_mid[i]
            paths[(u, v)] = [u, mid, v]

        return True, paths

    def check_hybrid(B, num_restarts=50):
        """Try matching first, fall back to BFS if matching fails."""
        valid, paths = check_with_matching(B)
        if valid:
            return True, paths
        return check_and_get_paths(B, num_restarts=num_restarts)

    # --- Step 1: Find a good initial branch set ---
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    best_B = set()
    best_paths = None
    best_score = 0

    seeds_to_try = []
    seeds_to_try.append(set(vertices_by_degree[:5]))

    max_clique = []

    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline - 15.0:
            return
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        if len(P) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        union_p_x = P.union(X)
        if not union_p_x:
            return
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            uncovered = len(P.difference(adj[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
        if best_pivot is None:
            return
        for v in list(P.difference(adj[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)

    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    if len(max_clique) >= 2:
        seeds_to_try.append(set(max_clique))

    def connectivity_score(v):
        if not adj[v]:
            return 0
        return degrees[v] * sum(degrees[u] for u in adj[v]) / len(adj[v])

    vertices_by_connectivity = sorted(range(n), key=lambda x: connectivity_score(x), reverse=True)
    seeds_to_try.append(set(vertices_by_connectivity[:5]))

    for seed in seeds_to_try:
        if time.time() > deadline - 10.0:
            break
        valid, paths = check_hybrid(seed, num_restarts=50)
        if valid and len(seed) > best_score:
            best_B = set(seed)
            best_paths = paths
            best_score = len(seed)

    if best_score < 2:
        for u, v in edges:
            best_B = {u, v}
            best_paths = {(u, v): [u, v]}
            best_score = 2
            break

    # --- Step 2: Greedy expansion ---
    current_B = set(best_B)
    current_paths = best_paths

    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False

        candidates = [v for v in vertices_by_degree if v not in current_B]

        for v in candidates:
            if time.time() > deadline - 8.0:
                break
            new_B = current_B | {v}
            valid, paths = check_hybrid(new_B, num_restarts=100)
            if valid:
                current_B = new_B
                current_paths = paths
                if len(current_B) > best_score:
                    best_B = set(current_B)
                    best_paths = paths
                    best_score = len(current_B)
                improved = True
                break

        if not improved:
            if len(current_B) > 2:
                b_sorted = sorted(current_B, key=lambda x: degrees[x])
                candidates_not_in_b = [v for v in vertices_by_degree if v not in current_B]

                for remove_v in b_sorted[:2]:
                    for add_v in candidates_not_in_b[:5]:
                        if time.time() > deadline - 8.0:
                            break
                        swap_B = (current_B - {remove_v}) | {add_v}
                        valid, paths = check_hybrid(swap_B, num_restarts=50)
                        if valid:
                            current_B = swap_B
                            current_paths = paths
                            improved = True
                            break
                    if improved:
                        break

    # --- Step 3: Try larger sets via random sampling ---
    if time.time() < deadline - 10.0:
        for target_size in [best_score + 1, best_score + 2]:
            if target_size > n:
                break
            for _ in range(30):
                if time.time() > deadline - 10.0:
                    break
                sample = random.sample(range(n), target_size)
                valid, paths = check_hybrid(sample, num_restarts=30)
                if valid and target_size > best_score:
                    best_B = set(sample)
                    best_paths = paths
                    best_score = target_size

    # --- Final verification and output ---
    if best_paths is None:
        valid, paths = check_hybrid(best_B, num_restarts=500)
        if valid:
            best_paths = paths
        else:
            best_B = set(max_clique) if len(max_clique) >= 2 else {0, 1}
            best_paths = {}
            for u, v in itertools.combinations(sorted(best_B), 2):
                if v in adj[u]:
                    best_paths[(u, v)] = [u, v]
                else:
                    p = find_path_bfs(u, v, set())
                    if p:
                        best_paths[(u, v)] = p
                    else:
                        best_paths[(u, v)] = [u, v]

    branches = sorted([int(v) for v in best_B])
    paths_list = []
    for u, v in itertools.combinations(branches, 2):
        key = (u, v)
        if key not in best_paths:
            key = (v, u)
        if key in best_paths:
            p = best_paths[key]
        else:
            p = [u, v]
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}

# EVOLVE-BLOCK-END
