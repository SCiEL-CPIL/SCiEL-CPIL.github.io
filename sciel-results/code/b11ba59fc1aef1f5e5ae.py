"""Rewritten program for clique partition of split graphs."""

def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)

        K = list(range(k))
        I = list(range(k, n))

        # Build adjacency masks for independent vertices to clique vertices
        k_to_bit = {u: i for i, u in enumerate(K)}
        neigh_mask = {}
        for v in I:
            mask = 0
            for u in K:
                if (min(u, v), max(u, v)) in edge_set:
                    mask |= (1 << k_to_bit[u])
            neigh_mask[v] = mask

        # Group independent vertices by their exact neighborhood mask
        groups = defaultdict(list)
        for v in I:
            groups[neigh_mask[v]].append(v)

        # Convert masks to sets of K-vertices for easier manipulation
        mask_to_set = {}
        for mask in groups:
            s = set()
            for u in K:
                if mask & (1 << k_to_bit[u]):
                    s.add(u)
            mask_to_set[mask] = s

        cliques = []
        covered_kk_edges = set()

        # Sort groups by size descending to prioritize large groups
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))

        for mask, group in sorted_groups:
            if not group:
                continue
            S = mask_to_set[mask]
            if not S:
                # No cross edges for this group, skip (no edges to cover)
                continue

            # Check if we can use the full clique S ∪ group
            # This requires that all K-K edges within S are not already covered
            can_use_full = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                    if e in covered_kk_edges:
                        can_use_full = False
                        break
                if not can_use_full:
                    break

            if can_use_full and len(S) >= 1:
                # Use the full clique S ∪ group
                clique = list(S) + group
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Mark all K-K edges within S as covered
                    for i in range(len(S_list)):
                        for j in range(i + 1, len(S_list)):
                            e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                            covered_kk_edges.add(e)
            else:
                # Use star cliques for each vertex in the group
                for v in group:
                    star = [v] + list(S)
                    if len(star) >= 2:
                        cliques.append(star)
                        # Mark K-K edges within S as covered
                        S_list = list(S)
                        for i in range(len(S_list)):
                            for j in range(i + 1, len(S_list)):
                                e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                                covered_kk_edges.add(e)

        # Now cover remaining K-K edges
        # All K-K edges are pairs (u, v) with u, v in K
        all_kk_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                all_kk_edges.add((i, j))

        remaining_kk = all_kk_edges - covered_kk_edges

        if remaining_kk:
            # Greedily cover remaining K-K edges with cliques (subsets of K)
            # Since K is a clique, any subset is a clique.
            # We need to partition the remaining edges into subsets.
            # A simple approach: use the remaining edges to form cliques.
            # Since all K-K edges form a complete graph on K, we can cover them
            # with a partition of the vertex set K into subsets, but we only
            # need to cover the remaining edges.
            
            # Greedy: repeatedly pick a vertex with the most remaining incident edges,
            # form a clique with its remaining neighbors, and remove those edges.
            remaining = set(remaining_kk)
            adj_remaining = defaultdict(set)
            for u, v in remaining:
                adj_remaining[u].add(v)
                adj_remaining[v].add(u)

            while remaining:
                # Pick vertex with max degree in remaining graph
                max_deg = -1
                pivot = None
                for u in range(k):
                    deg = len(adj_remaining[u])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = u
                
                if pivot is None or max_deg == 0:
                    break

                # Form clique with pivot and all its remaining neighbors
                clique = [pivot] + list(adj_remaining[pivot])
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Remove all edges within this clique from remaining
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                            remaining.discard(e)
                    # Update adjacency
                    for u in clique:
                        adj_remaining[u] = set()

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))

    return results
