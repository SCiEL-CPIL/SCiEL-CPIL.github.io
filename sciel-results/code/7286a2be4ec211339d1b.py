import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n = 16
    best_edges = []
    best_count = 0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def is_3_colorable(adj):
        """Check if graph with adjacency bitmasks is 3-colorable using backtracking."""
        # Order vertices by degree (descending) for better pruning
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: degs[i], reverse=True)

        colors = [-1] * n
        color_masks = [0, 0, 0]

        def backtrack(idx_in_order):
            if idx_in_order == n:
                return True

            v = order[idx_in_order]

            # Check which colors are available
            avail = []
            for c in range(3):
                if (adj[v] & color_masks[c]) == 0:
                    avail.append(c)

            if not avail:
                return False

            # Try colors in order of least constraining (fewest conflicts with uncolored neighbors)
            # For simplicity, just try all available colors
            for c in avail:
                colors[v] = c
                color_masks[c] |= (1 << v)

                if backtrack(idx_in_order + 1):
                    return True

                colors[v] = -1
                color_masks[c] &= ~(1 << v)

            return False

        return backtrack(0)

    def count_edges(adj):
        """Count edges in adjacency bitmask representation."""
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def get_candidate_edges(adj):
        """Get edges that can be added without creating triangles."""
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    # Check if adding (u,v) creates a triangle
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        return candidates

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_maximal_triangle_free(seed_edges=None):
        """Build a maximal triangle-free graph using greedy approach."""
        adj = [0] * n
        edge_count = 0

        # Start from seed if provided
        if seed_edges:
            for u, v in seed_edges:
                if (adj[u] & adj[v]) == 0:
                    add_edge(adj, u, v)
                    edge_count += 1

        # Add remaining edges greedily
        remaining = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        random.shuffle(remaining)

        for u, v in remaining:
            if (adj[u] & adj[v]) == 0:
                add_edge(adj, u, v)
                edge_count += 1

        return adj, edge_count

    def simulated_annealing(adj, edge_count, temp=10.0, cooling=0.995, max_iter=5000):
        """Simulated annealing to find denser 4-chromatic triangle-free graphs."""
        current_adj = adj[:]
        current_count = edge_count
        best_adj = adj[:]
        best_count = edge_count

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            # Get candidate edges to add
            candidates = get_candidate_edges(current_adj)
            if not candidates:
                break

            # Also consider removing an edge and adding a different one
            # For now, just try adding edges
            u, v = random.choice(candidates)

            # Add the edge
            add_edge(current_adj, u, v)
            new_count = current_count + 1

            # Check if still 3-colorable
            if is_3_colorable(current_adj):
                # Bad move - graph became 3-colorable
                delta = -100  # Large penalty
            else:
                # Good move - still 4-chromatic
                delta = 1  # Reward for more edges

            # Accept or reject based on temperature
            if delta > 0:
                # Always accept improving moves
                current_count = new_count
                if new_count > best_count:
                    best_count = new_count
                    best_adj = current_adj[:]
            else:
                # Degrading move - accept with probability exp(delta/temp)
                import math
                if temp > 0.01 and random.random() < math.exp(delta / temp):
                    current_count = new_count
                else:
                    # Reject - remove the edge
                    remove_edge(current_adj, u, v)

            # Cool down
            temp *= cooling

        return best_adj, best_count

    # First, try to find a good starting point
    # Try multiple random greedy constructions
    for restart in range(50):
        if time.time() > deadline:
            break

        adj, edge_count = greedy_maximal_triangle_free()

        if not is_3_colorable(adj):
            # This is a valid 4-chromatic triangle-free graph
            if edge_count > best_count:
                best_count = edge_count
                best_edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            best_edges.append([u, v])
                        mask ^= lsb

    # Now try simulated annealing from the best found so far
    if best_edges:
        # Reconstruct adjacency from best_edges
        adj = [0] * n
        for u, v in best_edges:
            add_edge(adj, u, v)

        # Run simulated annealing
        best_adj, best_count_sa = simulated_annealing(adj, best_count, temp=20.0, cooling=0.99, max_iter=3000)

        if best_count_sa > best_count:
            best_count = best_count_sa
            best_edges = []
            for u in range(n):
                mask = best_adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        best_edges.append([u, v])
                    mask ^= lsb

    # Fallback: Clebsch graph (known 4-chromatic triangle-free on 16 vertices)
    if best_count == 0:
        clebsch_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_edges.append([u, v])
            v = u ^ 15
            if u < v:
                clebsch_edges.append([u, v])
        best_edges = clebsch_edges

    return best_edges
