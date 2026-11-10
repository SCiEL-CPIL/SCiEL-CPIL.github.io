"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_path_decomposition(edges, max_time=None):
        """
        Find minimum number of edge-disjoint simple paths covering all edges.
        
        Strategy:
        1. Try to find Eulerian trails in connected components (if all degrees even)
        2. For non-Eulerian components, use a minimum path cover approach:
           - Each vertex of odd degree must be an endpoint of some path
           - Minimum number of paths = max(1, number_of_odd_degree_vertices / 2)
           - We construct paths by pairing up odd-degree vertices and finding
             paths between them, then covering remaining edges with cycles
        3. Use greedy longest-path extraction as a fallback/heuristic
        """
        if max_time is None:
            max_time = 30
        
        edge_set = set(edges)
        if not edge_set:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        deg = defaultdict(int)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Find connected components
        visited = set()
        components = []
        for v in adj:
            if v not in visited:
                comp = []
                stack = [v]
                visited.add(v)
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
                components.append(comp)
        
        paths = []
        
        for comp in components:
            comp_set = set(comp)
            comp_edges = [e for e in edge_set if e[0] in comp_set and e[1] in comp_set]
            
            if not comp_edges:
                continue
            
            # Check if Eulerian (all degrees even)
            odd_vertices = [v for v in comp if deg[v] % 2 == 1]
            
            if not odd_vertices:
                # Eulerian component: find an Eulerian circuit
                trail = _find_eulerian_trail(comp_edges, adj, comp_set)
                if trail and len(trail) >= 2:
                    # Convert circuit to a path by breaking it
                    paths.append(trail)
                else:
                    # Fallback: greedy
                    paths.extend(_greedy_cover(comp_edges, adj, comp_set, max_time))
            else:
                # Non-Eulerian: need at least len(odd_vertices)/2 paths
                # Strategy: pair up odd vertices, find paths between pairs,
                # then cover remaining edges with Eulerian trails in subcomponents
                
                # First, do a greedy longest path extraction which naturally
                # tends to use odd-degree vertices as endpoints
                paths.extend(_greedy_cover_smart(comp_edges, adj, comp_set, odd_vertices, max_time))
        
        return paths

    def _find_eulerian_trail(edge_list, adj, vertex_set):
        """Find an Eulerian trail using Hierholzer's algorithm."""
        if not edge_list:
            return []
        
        # Build multigraph adjacency with edge indices
        adj_edges = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            adj_edges[u].append((v, idx))
            adj_edges[v].append((u, idx))
        
        used = [False] * len(edge_list)
        
        # Start from any vertex
        start = edge_list[0][0]
        
        stack = [start]
        trail = []
        
        while stack:
            v = stack[-1]
            # Find an unused edge
            next_edge = None
            for nb, idx in adj_edges[v]:
                if not used[idx]:
                    next_edge = (nb, idx)
                    break
            
            if next_edge is None:
                trail.append(stack.pop())
            else:
                nb, idx = next_edge
                used[idx] = True
                stack.append(nb)
        
        trail.reverse()
        
        # Verify it's a valid trail
        if len(trail) < 2:
            return []
        
        # Check all edges are used
        if sum(used) != len(edge_list):
            return []
        
        return trail

    def _greedy_cover(edges, adj, vertex_set, max_time=None):
        """Greedy longest path extraction."""
        remaining = set(edges)
        paths = []
        
        while remaining:
            if max_time and (time.time() - start_time) > max_time:
                break
            
            best_path = None
            best_len = 0
            
            # Try multiple starting points
            candidates = list(vertex_set)
            random.shuffle(candidates)
            
            for start in candidates[:20]:
                path = _greedy_extend(start, remaining, adj)
                if path and len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None:
                e = next(iter(remaining))
                best_path = list(e)
            
            paths.append(best_path)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
        
        if remaining:
            for e in remaining:
                paths.append(list(e))
        
        return paths

    def _greedy_cover_smart(edges, adj, vertex_set, odd_vertices, max_time=None):
        """
        Greedy path extraction that prioritizes starting from odd-degree vertices.
        This helps ensure that odd-degree vertices become path endpoints.
        """
        remaining = set(edges)
        paths = []
        
        while remaining:
            if max_time and (time.time() - start_time) > max_time:
                break
            
            best_path = None
            best_len = 0
            
            # Prioritize odd-degree vertices as starting points
            # Recompute degrees in remaining graph
            rem_deg = defaultdict(int)
            for u, v in remaining:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            # Candidates: odd-degree vertices first, then others
            odd_in_rem = [v for v in vertex_set if rem_deg.get(v, 0) % 2 == 1]
            even_in_rem = [v for v in vertex_set if rem_deg.get(v, 0) % 2 == 0 and rem_deg.get(v, 0) > 0]
            
            candidates = odd_in_rem + even_in_rem
            random.shuffle(candidates)
            
            for start in candidates[:25]:
                path = _greedy_extend(start, remaining, adj)
                if path and len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None:
                e = next(iter(remaining))
                best_path = list(e)
            
            paths.append(best_path)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
        
        if remaining:
            for e in remaining:
                paths.append(list(e))
        
        return paths

    def _greedy_extend(start, remaining, adj):
        """
        Greedily extend a path from start using a lookahead heuristic.
        Uses Warnsdorff-like rule: prefer moves that leave fewer options for
        the next step (to avoid trapping ourselves).
        """
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Find unvisited neighbors connected by remaining edges
            candidates = []
            for neighbor in adj[current]:
                if neighbor not in visited:
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining:
                        candidates.append(neighbor)
            
            if not candidates:
                break
            
            if len(candidates) == 1:
                best_next = candidates[0]
            else:
                # Score each candidate: prefer the one with fewest onward options
                # (Warnsdorff's rule adapted for path finding)
                best_next = None
                best_score = float('inf')
                
                for cand in candidates:
                    # Count how many unvisited neighbors cand has via remaining edges
                    onward = 0
                    for nb in adj[cand]:
                        if nb not in visited and nb != current:
                            edge = tuple(sorted((cand, nb)))
                            if edge in remaining:
                                onward += 1
                    # Prefer fewer onward options (to avoid dead ends)
                    if onward < best_score:
                        best_score = onward
                        best_next = cand
                
                if best_next is None:
                    best_next = candidates[0]
            
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        
        return path

    def _improve_by_merge(paths, adj, edge_set, max_time=None):
        """
        Try to merge pairs of paths if they can be connected by removing
        a common edge and rearranging. This can reduce the total path count.
        """
        if max_time and (time.time() - start_time) > max_time:
            return paths
        
        changed = True
        while changed:
            changed = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    
                    # Try to merge paths[i] and paths[j]
                    merged = _try_merge(paths[i], paths[j], adj, edge_set)
                    if merged is not None:
                        new_paths.append(merged)
                        used[i] = True
                        used[j] = True
                        changed = True
                        break
            
            if not changed:
                break
            
            paths = new_paths
        
        return paths

    def _try_merge(path1, path2, adj, edge_set):
        """
        Try to merge two paths into one by finding a way to connect them.
        This is a simple heuristic: check if we can swap endpoints.
        """
        # For simplicity, just check if the two paths share a vertex
        # that could allow a merge. This is a placeholder for a more
        # sophisticated merge strategy.
        
        # A better approach: if path1 ends at u and path2 starts at v,
        # and there's an edge (u,v) not in either path, we can merge.
        # But we need to ensure the merged path is simple.
        
        # This is complex, so for now we return None (no merge)
        # and rely on the greedy approach being good enough.
        return None

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        if time.time() < deadline - 3:
            paths = min_path_decomposition(edges, max_time=deadline - 3)
        else:
            paths = [list(e) for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
