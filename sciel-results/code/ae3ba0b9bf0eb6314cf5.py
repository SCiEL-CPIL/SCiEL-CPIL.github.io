import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(adj):
        return [u for u in adj if len(adj[u]) % 2 != 0]

    def find_simple_path(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        # BFS
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in adj[current]:
                edge = (min(current, neighbor), max(current, neighbor))
                if edge not in remaining_edges:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    def decompose_cycles(remaining_edges, adj):
        """Decompose remaining edges (all even degree) into paths."""
        paths = []
        # Work on a copy of remaining edges
        rem_edges = set(remaining_edges)
        rem_adj = defaultdict(set)
        for u, v in rem_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        while rem_edges:
            if time.time() > deadline - 5:
                # Fallback: just return remaining edges as single-edge paths
                for u, v in rem_edges:
                    paths.append([u, v])
                return paths
                
            start_edge = next(iter(rem_edges))
            current = start_edge[0]
            path = [current]
            
            # Walk to form a cycle
            while True:
                next_vertex = None
                for neighbor in list(rem_adj[current]):
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in rem_edges:
                        next_vertex = neighbor
                        break
                
                if next_vertex is None:
                    break
                
                path.append(next_vertex)
                
                a, b = current, next_vertex
                edge = (min(a, b), max(a, b))
                rem_edges.remove(edge)
                rem_adj[a].remove(b)
                rem_adj[b].remove(a)
                
                current = next_vertex
                
                # If we've returned to start and covered all edges in this component, stop
                if current == path[0] and not any(edge in rem_edges for edge in 
                   [(min(path[0], nb), max(path[0], nb)) for nb in rem_adj[path[0]]]):
                    # Check if this component is done
                    # Actually, just break if we've used up all edges from start in this walk?
                    # Simpler: if we closed the cycle, break.
                    if all(nb not in rem_adj[path[0]] for nb in [n for n in rem_adj[path[0]] if (min(path[0], n), max(path[0], n)) in rem_edges]):
                         # This logic is messy. Just break if we closed cycle.
                         # But we might have multiple cycles.
                         # Let's just break when we return to start.
                         pass
                         
                # Break if we've closed the cycle
                if current == path[0]:
                    break
            # Also break if no more edges from current
            if not rem_adj[current] or all((min(current, n), max(current, n)) not in rem_edges for n in rem_adj[current]):
                break

            # We have a cycle (or path if not closed? But degrees are even, so it should close)
            # If it's a cycle, split it into 2 paths
            if len(path) > 1 and path[0] == path[-1]:
                # Cycle: v0, v1, ..., vk, v0
                # Path 1: v0 ... v_{k-1}
                # Path 2: v_{k-1} ... v0 (using remaining edges)
                # Actually, just split the list
                cycle = path[:-1] # v0...vk
                if len(cycle) > 1:
                    # Split at some point
                    mid = len(cycle) // 2
                    p1 = cycle[:mid+1]
                    p2 = cycle[mid:]
                    # Reverse p2? No, just list of vertices
                    # Check if p1 and p2 are simple paths
                    if len(set(p1)) == len(p1) and len(set(p2)) == len(p2):
                        paths.append(p1)
                        paths.append(p2)
                    else:
                        # Fallback: single edge paths
                        for i in range(len(cycle)):
                            u, v = cycle[i], cycle[(i+1)%len(cycle)]
                            paths.append([u, v])
            else:
                # Should not happen for even degree graph, but handle path
                if len(path) > 1:
                    paths.append(path)
        
        return paths

    def solve_graph(edges):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        odd_verts = get_odd_vertices(adj)
        n_odd = len(odd_verts)
        
        best_cover = None
        best_count = float('inf')
        
        # If no odd vertices, it's Eulerian
        # Min paths is at least 1 (if we allow cycle as path? No, simple path).
        # For Eulerian, min paths = 1? No, 2 for a cycle.
        # Generally, min paths for Eulerian is 1 if we consider "closed path" but task says "simple path".
        # A simple path cannot be a cycle.
        # So we must break cycles.
        
        rng = random.Random(12345)
        
        # Strategy 1: Pair odd vertices, find disjoint paths, decompose rest
        # Run multiple times with different random pairings
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
                
            remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
            paths = []
            
            # Copy adj for this iteration
            iter_adj = defaultdict(set)
            for u, v in remaining_edges:
                iter_adj[u].add(v)
                iter_adj[v].add(u)
                
            # Shuffle odd vertices
            odd_set = list(odd_verts)
            rng.shuffle(odd_set)
            
            success = True
            for i in range(0, len(odd_set) - 1, 2):
                u, v = odd_set[i], odd_set[i+1]
                path = find_simple_path(u, v, iter_adj, remaining_edges)
                if path:
                    paths.append(path)
                    # Remove edges
                    for j in range(len(path) - 1):
                        a, b = path[j], path[j+1]
                        edge = (min(a, b), max(a, b))
                        if edge in remaining_edges:
                            remaining_edges.remove(edge)
                            iter_adj[a].remove(b)
                            iter_adj[b].remove(a)
                else:
                    success = False
                    break
            
            if success and remaining_edges:
                # Decompose remaining cycles
                cycle_paths = decompose_cycles(remaining_edges, iter_adj)
                paths.extend(cycle_paths)
            
            if success:
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_cover = paths
                    
            # Early stop if we hit the lower bound
            # Lower bound: n_odd/2 + (number of cycles in remaining?)
            # We don't know cycles in advance.
            # But n_odd/2 is a hard lower bound for the paths connecting odd vertices.
            # Total paths >= n_odd/2.
            # If we reach n_odd/2, it's optimal (assuming no cycles left, or cycles merged).
            # Actually, if cycles remain, count > n_odd/2.
            # So lower bound is not just n_odd/2.
            # But if we find a solution with count == n_odd/2, it's perfect.
            if best_count == n_odd // 2:
                break

        # Fallback: Greedy
        if best_cover is None:
            remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
            iter_adj = defaultdict(set)
            for u, v in remaining_edges:
                iter_adj[u].add(v)
                iter_adj[v].add(u)
            paths = []
            while remaining_edges:
                start_edge = next(iter(remaining_edges))
                current = start_edge[0]
                path = [current]
                visited = {current}
                while True:
                    next_v = None
                    for nb in iter_adj[current]:
                        edge = (min(current, nb), max(current, nb))
                        if edge in remaining_edges and nb not in visited:
                            next_v = nb
                            break
                    if next_v is None:
                        break
                    path.append(next_v)
                    visited.add(next_v)
                    a, b = current, next_v
                    edge = (min(a, b), max(a, b))
                    remaining_edges.remove(edge)
                    iter_adj[a].remove(b)
                    iter_adj[b].remove(a)
                    current = next_v
                if len(path) > 1:
                    paths.append(path)
            best_cover = paths
            best_count = len(paths)

        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline - 5:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = solve_graph(edges)
        results.append(cover)
        
    return results
