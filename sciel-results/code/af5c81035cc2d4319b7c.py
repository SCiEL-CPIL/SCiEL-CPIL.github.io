def construct():
    import time
    import random
    from itertools import combinations

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def solve_instance(n, p, seed, time_limit=55):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Precompute adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Bitmask representation for speed
        # For n up to 64, we can use Python integers as bitmasks
        # adj_mask[i] has bit j set if (i,j) is an edge
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        def common_neighbors_mask(vertices):
            """Return bitmask of vertices connected to ALL vertices in the list."""
            if not vertices:
                return (1 << n) - 1
            mask = (1 << n) - 1
            for v in vertices:
                mask &= adj_mask[v]
            # Remove the vertices themselves
            for v in vertices:
                mask &= ~(1 << v)
            return mask

        def get_vertices_from_mask(mask):
            """Convert bitmask to list of vertex indices."""
            verts = []
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
            return verts

        def greedy_partition(remaining_edges, trial_rng, strategy="random"):
            """
            Greedily partition edges into cliques.
            remaining_edges: set of edge tuples
            strategy: 'random', 'greedy', 'greedy_large'
            """
            if not remaining_edges:
                return []

            partition = []
            current_edges = set(remaining_edges)

            while current_edges:
                # Pick a seed edge
                if strategy == "greedy" or strategy == "greedy_large":
                    # Pick edge with maximum sum of degrees in current_edges
                    deg = [0] * n
                    for u, v in current_edges:
                        deg[u] += 1
                        deg[v] += 1
                    best_edge = None
                    best_score = -1
                    for e in current_edges:
                        s = deg[e[0]] + deg[e[1]]
                        if s > best_score:
                            best_score = s
                            best_edge = e
                    seed_edge = best_edge
                else:
                    seed_edge = trial_rng.choice(list(current_edges))

                u0, v0 = seed_edge
                clique = [u0, v0]

                # Find all vertices that can join the clique
                # A vertex w can join if it's connected to all current clique members via edges in current_edges
                # We maintain a candidate mask
                cand_mask = common_neighbors_mask(clique)

                # Greedily expand
                while cand_mask:
                    # Pick a candidate to add
                    cand_verts = get_vertices_from_mask(cand_mask)

                    if strategy == "greedy_large":
                        # Pick the one with max degree in current_edges
                        deg = [0] * n
                        for u, v in current_edges:
                            deg[u] += 1
                            deg[v] += 1
                        best_w = max(cand_verts, key=lambda w: deg[w])
                    else:
                        best_w = trial_rng.choice(cand_verts)

                    # Add best_w to clique
                    clique.append(best_w)

                    # Update candidate mask
                    cand_mask = common_neighbors_mask(clique)

                # Add clique to partition
                partition.append(clique)

                # Remove all edges in this clique from current_edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = edge_key(clique[i], clique[j])
                        if e in current_edges:
                            current_edges.discard(e)

            return partition

        def validate_partition(partition, original_edges):
            """Check that partition is valid: each set is a clique, each edge in exactly one clique."""
            covered = set()
            for clique in partition:
                if len(clique) < 2:
                    return False
                # Check all pairs are edges in original graph
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = edge_key(clique[i], clique[j])
                        if e not in original_edges:
                            return False
                        if e in covered:
                            return False  # edge covered twice
                        covered.add(e)
            if covered != original_edges:
                return False
            return True

        def local_search(partition, original_edges, trial_rng, max_iterations=500):
            """
            Try to improve partition by:
            1. Merging two cliques if their union is a clique
            2. Moving vertices between cliques
            3. Splitting and re-merging
            """
            if not partition:
                return partition

            current_partition = [list(c) for c in partition]
            current_count = len(current_partition)

            # Precompute edge sets for each clique
            def clique_edges(clique):
                es = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        es.add(edge_key(clique[i], clique[j]))
                return es

            def is_clique(vertices):
                """Check if vertices form a clique in original graph."""
                for i in range(len(vertices)):
                    for j in range(i + 1, len(vertices)):
                        if edge_key(vertices[i], vertices[j]) not in original_edges:
                            return False
                return True

            improved = True
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                # Try merging pairs of cliques
                merged = False
                for i in range(len(current_partition)):
                    if merged:
                        break
                    for j in range(i + 1, len(current_partition)):
                        union = list(set(current_partition[i]) | set(current_partition[j]))
                        if is_clique(union):
                            current_partition[i] = union
                            current_partition.pop(j)
                            merged = True
                            improved = True
                            break

                if not improved:
                    # Try moving a vertex from one clique to another
                    for i in range(len(current_partition)):
                        if len(current_partition[i]) < 3:
                            continue
                        for v in list(current_partition[i]):
                            # Try removing v from clique i
                            remaining_i = [x for x in current_partition[i] if x != v]
                            if not remaining_i:
                                continue

                            # Try adding v to another clique j
                            for j in range(len(current_partition)):
                                if j == i:
                                    continue
                                new_j = list(current_partition[j]) + [v]
                                if is_clique(new_j):
                                    # Check that remaining_i is still a clique (it is, since we removed a vertex)
                                    current_partition[i] = remaining_i
                                    current_partition[j] = new_j
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

            return current_partition

        def local_search_v2(partition, original_edges, trial_rng, max_iterations=200):
            """
            More aggressive local search:
            - Try to merge any two cliques whose union is a clique
            - Try to move subsets of vertices
            """
            if not partition:
                return partition

            current_partition = [list(c) for c in partition]

            def is_clique(vertices):
                for i in range(len(vertices)):
                    for j in range(i + 1, len(vertices)):
                        if edge_key(vertices[i], vertices[j]) not in original_edges:
                            return False
                return True

            improved = True
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                # Try merging all pairs
                i = 0
                while i < len(current_partition):
                    j = i + 1
                    while j < len(current_partition):
                        union = list(set(current_partition[i]) | set(current_partition[j]))
                        if is_clique(union):
                            current_partition[i] = union
                            current_partition.pop(j)
                            improved = True
                            # Don't increment j, re-check with same i
                            continue
                        j += 1
                    i += 1

                if not improved:
                    # Try moving a vertex from larger clique to smaller
                    for i in range(len(current_partition)):
                        if len(current_partition[i]) < 3:
                            continue
                        # Sort by degree in remaining graph for potential moves
                        for v in current_partition[i]:
                            for j in range(len(current_partition)):
                                if j == i:
                                    continue
                                new_j = list(current_partition[j]) + [v]
                                if is_clique(new_j):
                                    remaining_i = [x for x in current_partition[i] if x != v]
                                    if len(remaining_i) >= 2:
                                        current_partition[i] = remaining_i
                                        current_partition[j] = new_j
                                        improved = True
                                        break
                                # Also try moving v to j and see if we can merge
                            if improved:
                                break
                        if improved:
                            break

            return current_partition

        best_partition = None
        best_count = float('inf')
        start_time = time.time()

        # Multiple strategies and trials
        strategies = ["random", "greedy", "greedy_large"]
        num_trials_per_strategy = 15

        for strategy in strategies:
            for trial in range(num_trials_per_strategy):
                if time.time() - start_time > time_limit:
                    break

                trial_rng = random.Random(seed + 10000 * strategy.__hash__() + trial * 7919)
                partition = greedy_partition(edges, trial_rng, strategy)

                if not validate_partition(partition, edges):
                    continue

                # Local search improvement
                partition = local_search_v2(partition, edges, trial_rng)

                if not validate_partition(partition, edges):
                    continue

                if len(partition) < best_count:
                    best_count = len(partition)
                    best_partition = [list(c) for c in partition]

        # If no valid partition found, fallback
        if best_partition is None:
            # Simple fallback: each edge as its own clique
            best_partition = [[u, v] for u, v in edges]
            best_count = len(best_partition)

        return best_partition

    # Solve both instances
    start = time.time()
    result1 = solve_instance(32, 0.6, 601, time_limit=55)
    elapsed1 = time.time() - start
    remaining = 140 - elapsed1  # Leave 10s margin for result2 and overhead
    result2 = solve_instance(64, 0.55, 602, time_limit=max(10, remaining))

    return [result1, result2]
