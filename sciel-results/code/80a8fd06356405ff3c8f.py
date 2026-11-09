"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # 120s wall-clock limit, use 110s as safety margin

    def random_graph(n, p, seed):
        """Generate a random graph with n vertices, edge probability p, and given seed."""
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def is_clique(vertices, edge_set):
        """Check if all pairs in vertices form edges."""
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                a, b = verts[i], verts[j]
                if a > b:
                    a, b = b, a
                if (a, b) not in edge_set:
                    return False
        return True

    def greedy_clique_partition(edges, n, max_time=10.0):
        """
        Greedy clique partition: repeatedly find a large clique covering many remaining edges,
        remove those edges, and continue until all edges are covered.
        
        Uses a heuristic: prioritize edges with high 'clique density' (common neighbors ratio).
        """
        edge_set = set(edges)
        if not edge_set:
            return []

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        cliques = []
        remaining = set(edge_set)
        
        # Precompute common neighbors for each edge
        def common_neighbors(u, v):
            """Return set of vertices adjacent to both u and v."""
            return adj[u] & adj[v]

        # Greedy: pick an edge, extend to maximal clique, remove those edges
        while remaining:
            if time.time() - start_time > max_time:
                break

            # Score each remaining edge by how many common neighbors it has
            # (higher = more potential for a large clique)
            best_edge = None
            best_score = -1
            # Sample a subset of edges if too many
            edge_list = list(remaining)
            if len(edge_list) > 500:
                sample_size = 200
                sampled = random.sample(edge_list, sample_size)
            else:
                sampled = edge_list

            for (u, v) in sampled:
                cn = len(common_neighbors(u, v))
                # Score: prefer edges with many common neighbors (dense local area)
                score = cn
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

            if best_edge is None:
                # Fallback: just take any edge
                best_edge = next(iter(remaining))

            # Extend best_edge to a maximal clique
            u, v = best_edge
            # Start with {u, v}
            clique = {u, v}
            # Candidates: common neighbors of all vertices in clique
            candidates = common_neighbors(u, v)
            
            # Greedily add vertices to clique
            while candidates:
                # Pick candidate with most connections to current clique
                best_cand = None
                best_conn = -1
                for c in candidates:
                    conn = len(clique & adj[c])
                    if conn > best_conn:
                        best_conn = conn
                        best_cand = c
                if best_cand is None:
                    break
                clique.add(best_cand)
                # Update candidates: must be adjacent to all in clique
                new_candidates = set()
                for c in candidates:
                    if c == best_cand:
                        continue
                    if all(c in adj[x] for x in clique):
                        new_candidates.add(c)
                candidates = new_candidates

            # Convert clique to sorted list
            clique_list = sorted(clique)
            cliques.append(clique_list)

            # Remove all edges within this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if a > b:
                        a, b = b, a
                    remaining.discard((a, b))

        # Handle any leftover edges (shouldn't happen if we ran to completion)
        if remaining:
            for edge in remaining:
                cliques.append(list(edge))

        return cliques

    def greedy_clique_partition_v2(edges, n, max_time=10.0):
        """
        Alternative greedy: build cliques by finding maximal cliques in the graph,
        prioritizing those that cover the most remaining edges.
        """
        edge_set = set(edges)
        if not edge_set:
            return []

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        cliques = []
        remaining = set(edge_set)

        # Find all maximal cliques (or a good subset) using Bron-Kerbosch variant
        # For dense graphs, this can be expensive, so we limit the search

        def find_maximal_clique(start_vertex, max_size=8):
            """Find a large clique starting from start_vertex using greedy expansion."""
            clique = {start_vertex}
            candidates = set(adj[start_vertex])
            
            while candidates and len(clique) < max_size:
                # Pick candidate with most connections to current clique
                best_cand = None
                best_conn = -1
                for c in candidates:
                    conn = len(clique & adj[c])
                    if conn > best_conn:
                        best_conn = conn
                        best_cand = c
                if best_cand is None:
                    break
                clique.add(best_cand)
                new_candidates = set()
                for c in candidates:
                    if c == best_cand:
                        continue
                    if all(c in adj[x] for x in clique):
                        new_candidates.add(c)
                candidates = new_candidates

            return clique

        # Iteratively find cliques that cover many remaining edges
        vertices = list(range(n))
        random.shuffle(vertices)

        for start_v in vertices:
            if time.time() - start_time > max_time:
                break
            if not remaining:
                break

            # Find a clique starting from start_v
            clique = find_maximal_clique(start_v, max_size=7)
            clique_list = sorted(clique)
            if len(clique_list) < 2:
                continue

            # Count how many remaining edges this clique covers
            covered = 0
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) in remaining:
                        covered += 1

            if covered == 0:
                continue

            # Only add if it covers at least 2 edges (to be a meaningful clique)
            if covered >= 2:
                cliques.append(clique_list)
                # Remove covered edges
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        if a > b:
                            a, b = b, a
                        remaining.discard((a, b))

        # Handle leftover edges
        if remaining:
            for edge in remaining:
                cliques.append(list(edge))

        return cliques

    def solve_instance(n, p, seed, time_budget=30.0):
        """Solve one instance with multiple strategies and return best result."""
        edges = random_graph(n, p, seed)
        
        best_cliques = None
        best_count = float('inf')

        # Strategy 1: Greedy with high-density edge selection
        t0 = time.time()
        cliques1 = greedy_clique_partition(edges, n, max_time=time_budget * 0.4)
        if is_valid_partition(cliques1, edges, n):
            count1 = len(cliques1)
            if count1 < best_count:
                best_count = count1
                best_cliques = cliques1

        # Strategy 2: Greedy with maximal clique search
        t0 = time.time()
        cliques2 = greedy_clique_partition_v2(edges, n, max_time=time_budget * 0.4)
        if is_valid_partition(cliques2, edges, n):
            count2 = len(cliques2)
            if count2 < best_count:
                best_count = count2
                best_cliques = cliques2

        # Strategy 3: Randomized restarts with soft-min edge selection
        t0 = time.time()
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            cliques3 = randomized_greedy_partition(edges, n, restart, max_time=time_budget * 0.1)
            if is_valid_partition(cliques3, edges, n):
                count3 = len(cliques3)
                if count3 < best_count:
                    best_count = count3
                    best_cliques = cliques3

        if best_cliques is None:
            # Fallback: each edge as its own clique
            best_cliques = [[list(e)] for e in edges]

        return best_cliques

    def is_valid_partition(cliques, edges, n):
        """Verify that the partition is valid."""
        edge_set = set(edges)
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check it's a clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edge_set:
                        return False
                    if (a, b) in covered:
                        return False
                    covered.add((a, b))
        # Check all edges are covered
        if covered != edge_set:
            return False
        return True

    def randomized_greedy_partition(edges, n, seed, max_time=5.0):
        """Randomized greedy with soft-min edge selection."""
        edge_set = set(edges)
        if not edge_set:
            return []

        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        cliques = []
        remaining = set(edge_set)

        while remaining:
            if time.time() - start_time > max_time:
                break

            # Score edges with randomization
            edge_list = list(remaining)
            scored_edges = []
            for (u, v) in edge_list:
                cn = len(adj[u] & adj[v])
                # Add noise for randomization
                score = cn + rng.random() * 2
                scored_edges.append((score, (u, v)))

            # Pick the highest-scoring edge
            scored_edges.sort(key=lambda x: -x[0])
            best_edge = scored_edges[0][1]

            # Extend to maximal clique
            u, v = best_edge
            clique = {u, v}
            candidates = adj[u] & adj[v]

            while candidates and len(clique) < 6:
                # Pick candidate with most connections, with some randomness
                best_cand = None
                best_conn = -1
                for c in candidates:
                    conn = len(clique & adj[c]) + rng.random()
                    if conn > best_conn:
                        best_conn = conn
                        best_cand = c
                if best_cand is None:
                    break
                clique.add(best_cand)
                new_candidates = set()
                for c in candidates:
                    if c == best_cand:
                        continue
                    if all(c in adj[x] for x in clique):
                        new_candidates.add(c)
                candidates = new_candidates

            clique_list = sorted(clique)
            cliques.append(clique_list)

            # Remove covered edges
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if a > b:
                        a, b = b, a
                    remaining.discard((a, b))

        # Handle leftovers
        if remaining:
            for edge in remaining:
                cliques.append(list(edge))

        return cliques

    # Solve both instances
    result = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time > DEADLINE - 20:
            # Not enough time, use simple fallback
            edges = random_graph(n, p, seed)
            result.append([list(e) for e in edges])
            continue

        time_budget = min(40.0, DEADLINE - 20 - (time.time() - start_time))
        cliques = solve_instance(n, p, seed, time_budget=time_budget)
        result.append(cliques)

    return result
# EVOLVE-BLOCK-END
