import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def min_path_decomposition(n, edges):
        """Find minimum number of edge-disjoint paths covering all edges using min-cost max-matching."""
        m = len(edges)
        if m == 0:
            return []
        
        # Build edge index mapping
        edge_idx = {}
        for i, e in enumerate(edges):
            edge_idx[e] = i
        
        # For each edge, determine its two endpoints
        # Left node i represents edge i as a "tail" (starting point of chain)
        # Right node j represents edge j as a "head" (ending point of chain)
        # We can chain edge i -> edge j if they share a vertex
        
        # Build cost matrix for matching
        # Cost[i][j] = 0 if edge i and edge j can be chained (share a vertex), INF otherwise
        # We want to maximize the number of chains (minimize cost where cost=0 is good)
        
        # Actually, let's think differently:
        # Each edge has two endpoints. An edge can be "started" at one endpoint and "ended" at the other.
        # We need to decide for each edge which endpoint is the start and which is the end.
        # Then we match: edge i ends at vertex v, edge j starts at vertex v -> chain them.
        
        # Simpler approach: 
        # Create a bipartite graph where:
        # - Left side: one node per edge (representing the "end" of that edge)
        # - Right side: one node per edge (representing the "start" of that edge)
        # - Edge from left_i to right_j exists if edge i and edge j share a vertex
        # - We find maximum matching to maximize chains
        # - Number of paths = m - (size of maximum matching)
        
        # But we also need to know WHICH vertex is shared to reconstruct paths.
        
        # Let's build the compatibility matrix
        # compat[i][j] = vertex if edges i and j share vertex, else -1
        
        compat = [[-1] * m for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(m):
                if i == j:
                    continue
                u2, v2 = edges[j]
                if u1 == u2:
                    compat[i][j] = u1
                elif u1 == v2:
                    compat[i][j] = u1
                elif v1 == u2:
                    compat[i][j] = v1
                elif v1 == v2:
                    compat[i][j] = v1
        
        # Cost matrix: 0 if compatible, large value if not
        BIG = 10**9
        cost = [[BIG if compat[i][j] == -1 else 0 for j in range(m)] for i in range(m)]
        
        # Find minimum cost maximum matching
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Count how many actual matches (cost 0) we got
        num_chains = 0
        match_map = {}  # match_map[i] = j means edge i chains to edge j
        for r, c in zip(row_ind, col_ind):
            if cost[r][c] == 0:
                num_chains += 1
                match_map[r] = c
        
        # Number of paths = m - num_chains
        num_paths = m - num_chains
        
        # Reconstruct paths
        # match_map[i] = j means edge i's end connects to edge j's start
        # So in a path, edge i comes before edge j
        
        # Build next pointers: next[i] = j if edge i chains to edge j
        next_edge = {}
        for i, j in match_map.items():
            next_edge[i] = j
        
        # Find path starts: edges that are not the target of any match
        # i.e., edges j where there's no i such that match_map[i] = j
        matched_as_target = set(match_map.values())
        path_starts = [i for i in range(m) if i not in matched_as_target]
        
        # Reconstruct each path
        paths = []
        for start in path_starts:
            path = []
            curr = start
            visited_in_path = set()
            while curr is not None and curr not in visited_in_path:
                visited_in_path.add(curr)
                u, v = edges[curr]
                # Determine which endpoint is the "start" and which is the "end"
                # The end of edge curr should match the start of next_edge[curr]
                if curr in next_edge:
                    next_e = next_edge[curr]
                    u_next, v_next = edges[next_e]
                    # The shared vertex is the end of curr and start of next_e
                    if compat[curr][next_e] == u:
                        # u is the end, so start is v
                        path.append(v)
                        path.append(u)
                    else:
                        # v is the end, so start is u
                        path.append(u)
                        path.append(v)
                    curr = next_e
                else:
                    # This is the last edge in the path
                    # We need to determine orientation
                    # The start of this edge should match the end of the previous edge
                    # For the first edge, we don't have previous info
                    # Let's just add both vertices in some order
                    # Actually, we need consistent orientation
                    # For now, add u then v
                    path.append(u)
                    path.append(v)
                    curr = None
            
            # Remove duplicate consecutive vertices
            if len(path) > 1:
                clean_path = [path[0]]
                for v in path[1:]:
                    if v != clean_path[-1]:
                        clean_path.append(v)
                paths.append(clean_path)
        
        return paths

    def greedy_fallback(n, edges):
        """Fallback greedy approach if matching fails or takes too long."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        curr_edges = set(edges)
        curr_paths = []
        
        while curr_edges:
            active_vertices = [v for v in range(n) if adj[v] and any((min(v, nb), max(v, nb)) in curr_edges for nb in adj[v])]
            
            if not active_vertices:
                break
            
            best_path_in_iter = []
            
            for s in active_vertices:
                path = [s]
                visited = {s}
                curr_node = s
                
                while True:
                    candidates = []
                    for nb in adj[curr_node]:
                        if nb in visited:
                            continue
                        e = (min(curr_node, nb), max(curr_node, nb))
                        if e in curr_edges:
                            unvis_deg = 0
                            for m in adj[nb]:
                                if m in visited:
                                    continue
                                e2 = (min(nb, m), max(nb, m))
                                if e2 in curr_edges:
                                    unvis_deg += 1
                            candidates.append((unvis_deg, nb))
                    
                    if not candidates:
                        break
                    
                    candidates.sort(key=lambda x: x[0])
                    min_deg = candidates[0][0]
                    min_candidates = [c for c in candidates if c[0] == min_deg]
                    next_node = min_candidates[0][1]
                    
                    path.append(next_node)
                    visited.add(next_node)
                    curr_node = next_node
                
                if len(path) > len(best_path_in_iter):
                    best_path_in_iter = path[:]
            
            if best_path_in_iter and len(best_path_in_iter) > 1:
                curr_paths.append(best_path_in_iter)
                for i in range(len(best_path_in_iter) - 1):
                    u, v = best_path_in_iter[i], best_path_in_iter[i + 1]
                    e = (min(u, v), max(u, v))
                    curr_edges.discard(e)
            else:
                if curr_edges:
                    e = next(iter(curr_edges))
                    curr_paths.append(list(e))
                    curr_edges.discard(e)
        
        return curr_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Try matching-based approach first
        if time.time() < deadline - 30.0:
            try:
                paths = min_path_decomposition(n, edges)
                # Validate: check that paths are valid
                valid = True
                used_edges = set()
                for path in paths:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in set(edges) or e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break
                
                if valid and len(used_edges) == len(edges):
                    all_results.append(paths)
                    continue
            except Exception:
                pass
        
        # Fallback to greedy
        paths = greedy_fallback(n, edges)
        all_results.append(paths)

    return all_results
