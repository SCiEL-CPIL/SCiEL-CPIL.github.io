import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _find_shortest_cycle(adj, n):
        """
        Find the shortest cycle in the graph represented by adj.
        Returns the cycle as a list of vertices, or None if no cycle exists.
        """
        best_cycle = None
        best_len = float('inf')
        
        # For each edge, check if there is a path between its endpoints not using that edge
        # If so, that path + the edge forms a cycle.
        # To find the shortest such path, we can run BFS.
        
        # Optimization: Only check edges where both endpoints have degree >= 2
        # Actually, just iterate all edges.
        
        for u, v in adj:
            if u not in adj or v not in adj:
                continue
            # Temporarily remove edge (u,v) to find path u->v
            # Since we are just checking existence and length, we can just BFS avoiding the direct edge
            
            # BFS from u to v
            visited = {u: 0}
            parent = {u: None}
            queue = deque([u])
            found = False
            path_len = -1
            end_node = None
            
            while queue:
                curr = queue.popleft()
                if curr == v:
                    found = True
                    path_len = visited[curr]
                    end_node = curr
                    break
                for neighbor in adj[curr]:
                    # Avoid the direct edge (curr, v) if curr == u? 
                    # Actually, we want to avoid the specific edge (u,v).
                    # If curr == u and neighbor == v, skip.
                    if curr == u and neighbor == v:
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = visited[curr] + 1
                        parent[neighbor] = curr
                        queue.append(neighbor)
            
            if found:
                cycle_len = path_len + 1
                if cycle_len < best_len:
                    # Reconstruct path
                    path = []
                    curr = end_node
                    while curr != u:
                        path.append(curr)
                        curr = parent[curr]
                    path.append(u)
                    path.reverse()
                    # The cycle is path + [v]? No, path is u -> ... -> v.
                    # So cycle is u, ..., v, u.
                    # Vertices: u, path[1], ..., path[-1], v
                    # Wait, path is from u to v.
                    # Cycle vertices: u, path[1], ..., path[-1] (which is v)
                    # So cycle is path.
                    best_cycle = path
                    best_len = cycle_len
        
        return best_cycle

    def _max_cycle_packing_shortest(edges, n, rng):
        """
        Find a set of edge-disjoint cycles using iterative shortest-cycle extraction.
        """
        if not edges:
            return [], set()
        
        cycles = []
        current_edges = set(edges)
        adj = defaultdict(set)
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Iteratively extract cycles
        max_iterations = 100
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            if not current_edges:
                break
            
            # Check if any vertex has degree >= 2
            has_cycle_potential = any(len(adj[v]) >= 2 for v in adj)
            if not has_cycle_potential:
                break
            
            # Find the shortest cycle
            cycle = _find_shortest_cycle(adj, n)
            
            if cycle:
                cycles.append(cycle)
                # Remove edges from current_edges
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    e = (min(u,v), max(u,v))
                    current_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                break
        
        return cycles, current_edges

    def _decompose(edges, n):
        """
        Decompose using iterative shortest-cycle extraction with multiple restarts.
        """
        if not edges:
            return []
        
        best_pieces = None
        best_score = float('inf')
        
        # Try multiple random orderings (though shortest cycle is deterministic, 
        # we can randomize tie-breaking or just run it once as it's fast)
        # However, different random seeds in instance generation are fixed.
        # The graph is fixed. The algorithm is deterministic.
        # So we only need to run it once?
        # But maybe we can try to find a 2-factor if shortest cycle fails to improve?
        # For now, let's just use the deterministic shortest cycle extraction.
        
        rng = random.Random(1000)
        cycles, remaining = _max_cycle_packing_shortest(edges, n, rng)
        
        score = len(cycles) + len(remaining)
        
        best_pieces = []
        for c in cycles:
            best_pieces.append(list(c))
        for e in remaining:
            best_pieces.append([e[0], e[1]])
        
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append(_decompose(edges, n))
        
        # Check time
        if time.time() - start_time > DEADLINE:
            break

    # Ensure we return 3 lists
    while len(results) < 3:
        results.append([])

    return results
