import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin below the 90s kill

    p = 101
    invs = [pow(x, -1, p) for x in range(1, 11)]

    # Enumerate all non-empty subsets of {1..10}
    # For each subset, record: mask, size, sum_mod_p, denominators list
    all_subsets = []  # list of (size, sum_mod_p, denoms_list)
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        s = 0
        denoms = []
        for i in range(10):
            if mask & (1 << i):
                s = (s + invs[i]) % p
                denoms.append(i + 1)
        all_subsets.append((size, s, denoms))

    # Group by size
    by_size = defaultdict(list)
    for size, s, denoms in all_subsets:
        by_size[size].append((s, denoms))

    # Hopcroft-Karp algorithm for bipartite matching
    def hopcroft_karp(adj, n_left, n_right):
        """
        adj: list of lists, adj[i] = list of right nodes connected to left node i
        n_left: number of left nodes
        n_right: number of right nodes
        Returns: (match_left, match_right) where match_left[i] = right node matched to left i, or -1
        """
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == INF:
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found

        def dfs(u):
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return pair_u, pair_v

    # Binary search on maximum allowed size K
    # For a given K, build bipartite graph:
    #   Left: 101 residues (0..100)
    #   Right: all subsets with size <= K
    #   Edge: residue r connected to subset if subset's sum mod p == r
    # Check if perfect matching exists (all 101 residues matched)

    # First, find min_size[r] = minimum subset size that achieves residue r
    min_size = [float('inf')] * p
    for size in range(1, 11):
        for s, denoms in by_size[size]:
            if size < min_size[s]:
                min_size[s] = size

    # The answer is at least max(min_size)
    min_possible = max(min_size) if all(min_size[r] < float('inf') for r in range(p)) else 11

    # Try K from min_possible upward
    best_result = None
    for K in range(min_possible, 11):
        if time.time() > deadline:
            break

        # Build adjacency for Hopcroft-Karp
        # Left nodes: residues 0..100
        # Right nodes: subsets with size <= K
        # We need to map each subset to a unique right-node index

        # Collect all subsets with size <= K
        right_subsets = []  # list of (sum_mod_p, denoms)
        for size in range(1, K + 1):
            for s, denoms in by_size[size]:
                right_subsets.append((s, denoms))

        n_right = len(right_subsets)

        # Build adjacency: for each residue r, which right nodes have sum == r
        adj = [[] for _ in range(p)]
        for j, (s, denoms) in enumerate(right_subsets):
            adj[s].append(j)

        # Run Hopcroft-Karp
        pair_u, pair_v = hopcroft_karp(adj, p, n_right)

        # Check if all residues are matched
        if all(pair_u[r] != -1 for r in range(p)):
            # Perfect matching found! Construct the result
            result = [None] * p
            for r in range(p):
                j = pair_u[r]
                result[r] = right_subsets[j][1]
            best_result = result
            break

    # If no perfect matching found (shouldn't happen for K <= 10), fall back
    if best_result is None:
        # Use greedy: for each residue, pick the smallest subset
        best_result = [None] * p
        for r in range(p):
            for size in range(1, 11):
                for s, denoms in by_size[size]:
                    if s == r:
                        best_result[r] = denoms
                        break
                if best_result[r] is not None:
                    break
            if best_result[r] is None:
                best_result[r] = list(range(1, 11))

    # Deadline guard
    if time.time() > deadline:
        pass  # We already have a result

    return best_result
