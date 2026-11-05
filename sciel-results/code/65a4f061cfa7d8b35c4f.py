import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(n, edges_set):
        """
        Find a clique partition using a bitmask-based greedy maximal clique extraction.
        Uses integer bitmasks for fast adjacency checks.
        """
        if not edges_set:
            return []

        # Build adjacency bitmasks
        # adj[i] is an integer where bit j is 1 if edge (i, j) exists in remaining_edges
        adj = [0] * n
        for (u, v) in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        remaining_edges = set(edges_set)
        partition = []

        def popcount(x):
            return bin(x).count("1")

        def find_maximal_clique(candidates_mask):
            """
            Greedy: repeatedly pick the vertex in candidates with the highest degree
            within candidates, add to clique, and remove non-neighbors.
            candidates_mask: bitmask of candidate vertices.
            Returns a bitmask of the clique found.
            """
            clique_mask = 0
            current = candidates_mask

            while current:
                # Find vertex in current with max degree to other vertices in current
                best_v = -1
                best_deg = -1
                # Iterate over set bits in current
                c = current
                while c:
                    # Get lowest set bit
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    # Degree of v in current: count common bits between adj[v] and current
                    deg = popcount(adj[v] & current)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v

                if best_v < 0:
                    break

                # Add v to clique
                clique_mask |= (1 << best_v)

                # Remove v and its non-neighbors from current
                # Neighbors of best_v in current: adj[best_v] & current
                # Non-neighbors: current & ~(adj[best_v] | (1 << best_v))
                non_neighbors = current & ~(adj[best_v] | (1 << best_v))
                current = current & ~(1 << best_v) & ~non_neighbors
                # Equivalent: current = (current & ~(1 << best_v)) & adj[best_v]
                # But careful: we want to keep only neighbors of best_v that are still in current
                # So: current = (current & ~(1 << best_v)) & adj[best_v]
                current = (current & ~(1 << best_v)) & adj[best_v]

            return clique_mask

        # Main loop: repeatedly find a large clique among remaining edges
        while remaining_edges:
            if time.time() - start_time > DEADLINE:
                # Fallback: remaining edges as 2-cliques
                for e in remaining_edges:
                    partition.append(list(e))
                break

            # Compute degrees in the remaining graph to pick a starting vertex
            degrees = [0] * n
            adj_rem = [0] * n
            for (u, v) in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
                adj_rem[u] |= (1 << v)
                adj_rem[v] |= (1 << u)

            # Pick vertex with max degree
            if not any(deg > 0 for deg in degrees):
                break

            v = max(range(n), key=lambda x: degrees[x])

            # Candidates for clique: neighbors of v in remaining graph
            candidates_mask = adj_rem[v]

            # Find a maximal clique among candidates
            clique_neighbors_mask = find_maximal_clique(candidates_mask)

            # Full clique includes v
            full_clique_mask = clique_neighbors_mask | (1 << v)

            # Extract vertices from bitmask
            verts = []
            cm = full_clique_mask
            while cm:
                lsb = cm & (-cm)
                vert = lsb.bit_length() - 1
                verts.append(vert)
                cm ^= lsb

            partition.append(sorted(verts))

            # Remove edges of this clique from remaining_edges
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v2 = verts[i], verts[j]
                    if u > v2:
                        u, v2 = v2, u
                    if (u, v2) in remaining_edges:
                        remaining_edges.remove((u, v2))

        return partition

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        return greedy_clique_partition(n, edges_set)

    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]

    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
