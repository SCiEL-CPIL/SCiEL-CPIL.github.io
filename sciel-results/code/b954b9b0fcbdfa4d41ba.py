import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    m = len(edges)

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Edge index map: (u,v) with u<v -> index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        edge_index[(u, v)] = i

    def popcount(x):
        return bin(x).count('1')

    # Enumerate all maximal bicliques using Eppstein's algorithm
    # A biclique is (L, R) where L and R are vertex sets, all cross edges present.
    # We enumerate maximal ones (can't add any vertex to either side).
    # Approach: for each vertex, find its maximal bicliques where it's the "first" vertex.
    
    # Simpler: enumerate all maximal bicliques by iterating over pairs
    # and using a DFS-based approach.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Generate candidate bicliques via greedy expansion from random seeds
    # 2. Also try to find large bicliques via specific strategies
    
    # For efficiency, let's generate bicliques by:
    # - For each edge (u,v), try to expand to a maximal biclique
    # - Use a randomized approach with multiple restarts
    
    def expand_biclique(u, v, adj_mask):
        """Expand (u,v) to a maximal biclique using greedy expansion."""
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L: add u' if connected to all in R
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            for u_prime in range(n):
                if u_prime not in L:
                    if (adj_mask[u_prime] & mask_R) == mask_R:
                        L.add(u_prime)
                        changed = True
            # Expand R: add v' if connected to all in L
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            for v_prime in range(n):
                if v_prime not in R:
                    if all((adj_mask[lu] & (1 << v_prime)) != 0 for lu in L):
                        R.add(v_prime)
                        changed = True
        return L, R

    # Generate candidate bicliques
    candidates = []
    seen = set()
    
    # For each edge, try to expand
    for u, v in edges:
        L, R = expand_biclique(u, v, adj)
        # Normalize: sort L and R
        key = (tuple(sorted(L)), tuple(sorted(R)))
        if key not in seen:
            seen.add(key)
            candidates.append((L, R))
    
    # Also try some random perturbations
    for _ in range(200):
        if time.time() > deadline:
            break
        # Pick random edge
        u, v = random.choice(edges)
        # Randomly remove some edges from adj to get a sub-biclique
        adj_mod = adj[:]
        for _ in range(random.randint(0, 5)):
            ru, rv = random.choice(edges)
            adj_mod[ru] &= ~(1 << rv)
            adj_mod[rv] &= ~(1 << ru)
        L, R = expand_biclique(u, v, adj_mod)
        key = (tuple(sorted(L)), tuple(sorted(R)))
        if key not in seen:
            seen.add(key)
            candidates.append((L, R))

    # For each candidate, compute which edges it covers
    # Edge is covered if both endpoints are in different sides and edge exists
    candidate_edge_masks = []
    for L, R in candidates:
        mask = 0
        for u in L:
            for v in R:
                if u < v:
                    idx = edge_index.get((u, v))
                    if idx is not None:
                        mask |= (1 << idx)
                else:
                    idx = edge_index.get((v, u))
                    if idx is not None:
                        mask |= (1 << idx)
        candidate_edge_masks.append(mask)

    # Remove dominated candidates (if A's edges are subset of B's edges, A is worse)
    # Keep only non-dominated
    non_dominated = []
    for i, mask_i in enumerate(candidate_edge_masks):
        dominated = False
        for j, mask_j in enumerate(candidate_edge_masks):
            if i != j and (mask_i & ~mask_j) == 0 and mask_i != mask_j:
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)

    # Sort by size (larger first) for better pruning
    non_dominated.sort(key=lambda i: popcount(candidate_edge_masks[i]), reverse=True)

    # Greedy set cover to get initial solution
    def greedy_cover():
        remaining = (1 << m) - 1
        cover = []
        while remaining:
            best_idx = -1
            best_count = 0
            for i in non_dominated:
                cnt = popcount(remaining & candidate_edge_masks[i])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx == -1:
                break
            cover.append(best_idx)
            remaining &= ~candidate_edge_masks[best_idx]
        return cover

    initial_cover = greedy_cover()
    best_count = len(initial_cover)
    best_cover = initial_cover[:]

    # Branch and bound
    def branch_and_bound(remaining, current_cover, depth):
        nonlocal best_count, best_cover

        if time.time() > deadline:
            return

        if not remaining:
            if len(current_cover) < best_count:
                best_count = len(current_cover)
                best_cover = current_cover[:]
            return

        # Lower bound: at least ceil(popcount(remaining) / max_candidate_size)
        max_size = popcount(candidate_edge_masks[non_dominated[0]]) if non_dominated else 1
        lb = (popcount(remaining) + max_size - 1) // max_size
        if len(current_cover) + lb >= best_count:
            return

        # Try each candidate that covers at least one remaining edge
        for i in non_dominated:
            if len(current_cover) + 1 + lb >= best_count:
                break
            mask = candidate_edge_masks[i]
            if (remaining & mask) == 0:
                continue
            current_cover.append(i)
            branch_and_bound(remaining & ~mask, current_cover, depth + 1)
            current_cover.pop()

    # Run branch and bound with a time limit
    branch_and_bound((1 << m) - 1, [], 0)

    # Construct result from best_cover
    result = []
    for idx in best_cover:
        L, R = candidates[idx]
        result.append([list(L), list(R)])

    # Convert to required format
    final_result = []
    for b in result:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        final_result.append([left, right])

    return final_result
