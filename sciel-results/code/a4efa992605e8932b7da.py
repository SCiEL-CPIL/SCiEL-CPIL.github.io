def check_girth_full(n, adj, g):
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            curr = queue.popleft()
            for nei in adj[curr]:
                if dist[nei] == -1:
                    dist[nei] = dist[curr] + 1
                    parent[nei] = curr
                    queue.append(nei)
                elif parent[curr] != nei and parent[nei] != curr:
                    # Found a cycle
                    # Length = dist[curr] + dist[nei] + 1
                    # Wait, this is for undirected graph.
                    # The path start->curr and start->nei meet.
                    # Cycle length is dist[curr] + dist[nei] + 1?
                    # No, the path is curr->start->...->nei->curr?
                    # The cycle is formed by the two paths from start to curr and start to nei, plus the edge (curr, nei).
                    # Length = dist[curr] + dist[nei] + 1.
                    # We need to check if this length < g.
                    # But dist is shortest path.
                    # This finds *a* cycle.
                    cycle_len = dist[curr] + dist[nei] + 1
                    if cycle_len < g:
                        return False
    return True
