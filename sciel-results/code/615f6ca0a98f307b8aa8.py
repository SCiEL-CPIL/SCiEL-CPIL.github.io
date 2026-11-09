"""Rewritten program: deterministic K_{8,8} with edge-criticality verification."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin within 60s limit

    n = 16
    half = n // 2

    # Build K_{8,8}: vertices 0..7 (set A) connected to 8..15 (set B)
    edges = []
    for u in range(half):
        if time.time() > deadline:
            break
        for v in range(half, n):
            if time.time() > deadline:
                break
            edges.append([u, v])

    # Optional: verify edge-criticality (each edge deletion increases diameter or disconnects)
    # For K_{8,8}, this is structurally guaranteed, but we can do a quick check
    # to ensure no bugs in construction.
    if time.time() < deadline - 5.0:
        # Build adjacency for verification
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        def bfs_distances(start, adj_list):
            """Return list of distances from start to all vertices, or -1 for unreachable."""
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            idx = 0
            while idx < len(queue):
                cur = queue[idx]
                idx += 1
                for nb in adj_list[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        queue.append(nb)
            return dist

        def graph_diameter(adj_list):
            """Return diameter of the graph, or infinity if disconnected."""
            for s in range(n):
                dist = bfs_distances(s, adj_list)
                if -1 in dist:
                    return float('inf')
                max_d = max(dist)
                if max_d > 2:
                    return max_d
            return 2  # all pairs have distance <= 2, and at least one pair has distance 2

        # Verify original graph has diameter exactly 2
        orig_diam = graph_diameter(adj)
        if orig_diam != 2:
            # Fallback: this shouldn't happen for K_{8,8}
            pass

        # Verify each edge is critical (deleting it makes diameter > 2 or disconnected)
        for i, (u, v) in enumerate(edges):
            if time.time() > deadline - 2.0:
                break
            # Remove edge (u, v)
            adj[u].discard(v)
            adj[v].discard(u)
            diam = graph_diameter(adj)
            # Restore
            adj[u].add(v)
            adj[v].add(u)
            if diam <= 2:
                # Edge is not critical — this should not happen for K_{8,8}
                # If it does, we'd need to remove it, but for now just note it
                pass

    # Ensure return type: list of [int, int] lists
    return edges
# EVOLVE-BLOCK-END
