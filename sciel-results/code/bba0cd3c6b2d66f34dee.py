"""Path decomposition for Gallai's conjecture - improved approach."""
import time
from collections import defaultdict, deque
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bfs_path(adj, s, t):
        """Find shortest path from s to t in adj."""
        if s == t:
            return [s]
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if neighbor == t:
                        # Reconstruct
                        path = []
                        cur = t
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return path
                    queue.append(neighbor)
        return None

    def find_eulerian_cycle(adj, start):
        """Find an Eulerian cycle starting/ending at start using Hierholzer's."""
        adj_copy = defaultdict(set)
        for u, vs in adj.items():
            adj_copy[u] = set(vs)
        
        stack = [start]
        cycle = []
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                w = next(iter(adj_copy[v]))
                adj_copy[v].discard(w)
                adj_copy[w].discard(v)
                stack.append(w)
            else:
                cycle.append(stack.pop())
        
        cycle.reverse()
        # cycle[0] == cycle[-1] for a proper cycle, but Hierholzer gives
        # the path where first == last only if we check. Actually the returned
        # list has start at both ends only if we add it. Let's check:
        # The algorithm pops when no more edges, so the first element is start
        # and the last element is also start (since we came back).
        # Actually no - let's trace: stack=[start], we push neighbors...
        # The final reversed list starts with start and ends with start.
        return cycle

    def decompose_graph(edges):
        """
        Decompose a graph into paths:
        1. Find all odd-degree vertices
        2. Pair them up, find paths between each pair
        3. Remove those edges
        4. Remaining graph has all even degrees -> decompose into Eulerian circuits
        5. Each circuit is a closed walk; convert to a path by removing the repeated endpoint
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        remaining = set(edges)
        paths = []
        
        # Step 1: Pair odd vertices and find paths between them
        # Use a greedy pairing: sort by vertex id and pair consecutively
        # But we can try to minimize path lengths by trying different pairings
        # For now, use simple consecutive pairing
        
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            # Build adjacency from remaining edges for BFS
            rem_adj = build_adj(remaining)
            path = bfs_path(rem_adj, s, t)
            
            if path is None:
                # Shouldn't happen in a connected graph, but fallback
                continue
            
            # Remove edges along this path
            for k in range(len(path) - 1):
                u, v = path[k], path[k + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            if len(path) >= 2:
                paths.append(path)
        
        # Step 2: Decompose remaining graph (all even degrees) into Eulerian circuits
        while remaining:
            rem_adj = build_adj(remaining)
            
            # Find a vertex with non-zero degree
            start = None
            for u, vs in rem_adj.items():
                if vs:
                    start = u
                    break
            if start is None:
                break
            
            # Find Eulerian cycle
            cycle = find_eulerian_cycle(rem_adj, start)
            
            # Remove the cycle's edges from remaining
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            # The cycle is closed: last vertex connects back to first
            if len(cycle) > 1:
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            
            # Convert cycle to path: remove the last vertex (which equals the first)
            # Actually, cycle from Hierholzer: first element is start, and the walk
            # returns to start. So cycle[0] == cycle[-1].
            # To make it a simple path, we remove the last element.
            if len(cycle) > 1:
                path = cycle[:-1] if cycle[0] == cycle[-1] else cycle
                if len(path) >= 2:
                    paths.append(path)
        
        return paths

    def verify_decomposition(edges, paths):
        """Verify that paths form a valid edge-disjoint decomposition covering all edges."""
        used = set()
        for path in paths:
            # Check simplicity
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e in used:
                    return False
                if e not in edges:
                    return False
                used.add(e)
        return used == set(edges)

    def local_search_refine(edges, paths, time_limit=5.0):
        """
        Try to reduce the number of paths by merging adjacent paths.
        Two paths can be merged if the end of one connects to the start of the other
        (or vice versa) via an unused edge, and the merged walk is simple.
        """
        deadline = time.time() + time_limit
        current_paths = [list(p) for p in paths]
        
        while time.time() < deadline:
            improved = False
            n = len(current_paths)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    # Try to merge path j into path i
                    # Check if end of i connects to start of j
                    if len(current_paths[i]) < 2 or len(current_paths[j]) < 2:
                        continue
                    
                    # Try: end of i -> start of j
                    a, b = current_paths[i][-1], current_paths[j][0]
                    e = (min(a, b), max(a, b))
                    if e in edges:
                        # Check if e is not already used in either path
                        used_edges_i = set()
                        for k in range(len(current_paths[i]) - 1):
                            u, v = current_paths[i][k], current_paths[i][k+1]
                            used_edges_i.add((min(u,v), max(u,v)))
                        used_edges_j = set()
                        for k in range(len(current_paths[j]) - 1):
                            u, v = current_paths[j][k], current_paths[j][k+1]
                            used_edges_j.add((min(u,v), max(u,v)))
                        
                        if e not in used_edges_i and e not in used_edges_j:
                            merged = current_paths[i] + current_paths[j][1:]
                            # Check simplicity
                            if len(set(merged)) == len(merged):
                                # Check all edges are valid
                                valid = True
                                for k in range(len(merged) - 1):
                                    u, v = merged[k], merged[k+1]
                                    ee = (min(u,v), max(u,v))
                                    if ee not in edges:
                                        valid = False
                                        break
                                if valid:
                                    new_paths = [p for idx, p in enumerate(current_paths) if idx != i and idx != j]
                                    new_paths.append(merged)
                                    current_paths = new_paths
                                    improved = True
                                    break
                    if improved:
                        break
                    
                    # Try: end of j -> start of i
                    a, b = current_paths[j][-1], current_paths[i][0]
                    e = (min(a, b), max(a, b))
                    if e in edges:
                        used_edges_i = set()
                        for k in range(len(current_paths[i]) - 1):
                            u, v = current_paths[i][k], current_paths[i][k+1]
                            used_edges_i.add((min(u,v), max(u,v)))
                        used_edges_j = set()
                        for k in range(len(current_paths[j]) - 1):
                            u, v = current_paths[j][k], current_paths[j][k+1]
                            used_edges_j.add((min(u,v), max(u,v)))
                        
                        if e not in used_edges_i and e not in used_edges_j:
                            merged = current_paths[j] + current_paths[i][1:]
                            if len(set(merged)) == len(merged):
                                valid = True
                                for k in range(len(merged) - 1):
                                    u, v = merged[k], merged[k+1]
                                    ee = (min(u,v), max(u,v))
                                    if ee not in edges:
                                        valid = False
                                        break
                                if valid:
                                    new_paths = [p for idx, p in enumerate(current_paths) if idx != i and idx != j]
                                    new_paths.append(merged)
                                    current_paths = new_paths
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current_paths

    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 15:
            # Fallback: single edges
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        
        try:
            paths = decompose_graph(edges)
            
            # Verify
            if not verify_decomposition(set(edges), paths):
                # Fallback to single edges
                paths = [[u, v] for u, v in edges]
            else:
                # Try local search refinement
                if time.time() < DEADLINE - 10:
                    refined = local_search_refine(set(edges), paths, time_limit=5.0)
                    if verify_decomposition(set(edges), refined):
                        paths = refined
                
                results.append(paths)
                continue
        except Exception:
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    return results
