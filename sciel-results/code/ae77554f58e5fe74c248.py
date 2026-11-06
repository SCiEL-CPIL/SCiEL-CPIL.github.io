"""Path decomposition for Gallai's conjecture - greedy maximal path approach."""
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

    def greedy_path_decomposition(edges):
        """
        Decompose graph into paths by greedily building maximal simple paths.
        Each path is extended as far as possible at both ends.
        """
        if not edges:
            return []
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Build adjacency from remaining edges
            adj = build_adj(remaining)
            
            # Find a starting vertex: prefer vertices with highest degree
            start = None
            max_deg = -1
            for u, vs in adj.items():
                if vs and len(vs) > max_deg:
                    max_deg = len(vs)
                    start = u
            
            if start is None:
                break
            
            # Build a maximal simple path starting from 'start'
            # We'll extend in both directions
            path = [start]
            used_edges = set()
            
            # Extend forward from start
            while True:
                current = path[-1]
                # Find an unused neighbor
                next_node = None
                for neighbor in sorted(adj[current]):
                    e = (min(current, neighbor), max(current, neighbor))
                    if e in remaining and e not in used_edges:
                        # Check if neighbor is not already in path (simplicity)
                        if neighbor not in path:
                            next_node = neighbor
                            break
                if next_node is None:
                    break
                path.append(next_node)
                e = (min(current, next_node), max(current, next_node))
                used_edges.add(e)
            
            # Extend backward from start
            while True:
                current = path[0]
                next_node = None
                for neighbor in sorted(adj[current]):
                    e = (min(current, neighbor), max(current, neighbor))
                    if e in remaining and e not in used_edges:
                        if neighbor not in path:
                            next_node = neighbor
                            break
                if next_node is None:
                    break
                path.insert(0, next_node)
                e = (min(current, next_node), max(current, next_node))
                used_edges.add(e)
            
            # Remove used edges from remaining
            for e in used_edges:
                remaining.discard(e)
            
            if len(path) >= 2:
                paths.append(path)
        
        return paths

    def verify_decomposition(edges, paths):
        """Verify that paths form a valid edge-disjoint decomposition covering all edges."""
        used = set()
        for path in paths:
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
                    
                    # Try: end of i -> start of j
                    if len(current_paths[i]) >= 2 and len(current_paths[j]) >= 2:
                        a, b = current_paths[i][-1], current_paths[j][0]
                        e = (min(a, b), max(a, b))
                        if e in edges:
                            # Check if e is not already used
                            used_edges = set()
                            for k in range(len(current_paths[i]) - 1):
                                u, v = current_paths[i][k], current_paths[i][k+1]
                                used_edges.add((min(u,v), max(u,v)))
                            for k in range(len(current_paths[j]) - 1):
                                u, v = current_paths[j][k], current_paths[j][k+1]
                                used_edges.add((min(u,v), max(u,v)))
                            
                            if e not in used_edges:
                                merged = current_paths[i] + current_paths[j][1:]
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
                    
                    # Try: end of j -> start of i
                    if len(current_paths[i]) >= 2 and len(current_paths[j]) >= 2:
                        a, b = current_paths[j][-1], current_paths[i][0]
                        e = (min(a, b), max(a, b))
                        if e in edges:
                            used_edges = set()
                            for k in range(len(current_paths[i]) - 1):
                                u, v = current_paths[i][k], current_paths[i][k+1]
                                used_edges.add((min(u,v), max(u,v)))
                            for k in range(len(current_paths[j]) - 1):
                                u, v = current_paths[j][k], current_paths[j][k+1]
                                used_edges.add((min(u,v), max(u,v)))
                            
                            if e not in used_edges:
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

    def decompose_with_eulerian(edges):
        """
        Alternative: pair odd vertices with paths, then Eulerian decompose.
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        deg = {u: len(vs) for u, vs in adj.items()}
        odd_vertices = sorted([u for u, d in deg.items() if d % 2 == 1])
        
        remaining = set(edges)
        paths = []
        
        # Pair odd vertices
        for i in range(0, len(odd_vertices), 2):
            if i + 1 >= len(odd_vertices):
                break
            s, t = odd_vertices[i], odd_vertices[i + 1]
            
            # BFS to find path
            rem_adj = build_adj(remaining)
            if s == t:
                continue
            visited = {s}
            parent = {s: None}
            queue = deque([s])
            found = False
            while queue and not found:
                node = queue.popleft()
                for neighbor in rem_adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = node
                        if neighbor == t:
                            found = True
                            break
                        queue.append(neighbor)
            
            if found:
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                
                if len(path) >= 2:
                    paths.append(path)
        
        # Eulerian decomposition of remaining
        while remaining:
            rem_adj = build_adj(remaining)
            start = None
            for u, vs in rem_adj.items():
                if vs:
                    start = u
                    break
            if start is None:
                break
            
            # Hierholzer
            adj_copy = defaultdict(set)
            for u, vs in rem_adj.items():
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
            
            for k in range(len(cycle) - 1):
                u, v = cycle[k], cycle[k + 1]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            if len(cycle) > 1:
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                remaining.discard(e)
            
            if len(cycle) > 1:
                path = cycle[:-1] if cycle[0] == cycle[-1] else cycle
                if len(path) >= 2:
                    paths.append(path)
        
        return paths

    # Main logic
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 15:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        edge_set = set(edges)
        
        try:
            # Try greedy maximal path decomposition
            paths = greedy_path_decomposition(edge_set)
            
            if not verify_decomposition(edge_set, paths):
                # Try Eulerian-based approach
                paths = decompose_with_eulerian(edge_set)
            
            if not verify_decomposition(edge_set, paths):
                # Fallback
                paths = [[u, v] for u, v in edges]
            else:
                # Try local search refinement
                if time.time() < DEADLINE - 10:
                    refined = local_search_refine(edge_set, paths, time_limit=5.0)
                    if verify_decomposition(edge_set, refined):
                        paths = refined
                
                results.append(paths)
                continue
        except Exception:
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
    
    return results
