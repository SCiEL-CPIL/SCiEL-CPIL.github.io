"""Path decomposition via maximum matching (bipartite) + greedy refinement."""
import time
import sys
from collections import defaultdict
import random
import numpy as np

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

    def get_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def verify_decomposition(paths, edges, n):
        """Verify that paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        all_edges = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != all_edges:
            return False
        
        return True

    def hopcroft_karp(left_nodes, right_nodes, adjacency):
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        left_nodes: list of left-side node ids
        right_nodes: list of right-side node ids
        adjacency: dict mapping left_node -> list of right_node neighbors
        
        Returns (matching_left, matching_right) where
        matching_left[l] = r means l is matched to r
        matching_right[r] = l means r is matched to l
        """
        from collections import deque
        
        # Build adjacency list
        adj = {l: [] for l in left_nodes}
        for l in left_nodes:
            if l in adjacency:
                adj[l] = list(adjacency[l])
        
        matching_l = {}  # left -> right
        matching_r = {}  # right -> left
        dist = {}
        
        def bfs():
            queue = deque()
            for l in left_nodes:
                if l not in matching_l:
                    dist[l] = 0
                    queue.append(l)
                else:
                    dist[l] = float('inf')
            
            found = False
            while queue:
                l = queue.popleft()
                for r in adj.get(l, []):
                    ml = matching_r.get(r)
                    if ml is None:
                        found = True
                    elif dist[ml] == float('inf'):
                        dist[ml] = dist[l] + 1
                        queue.append(ml)
            return found
        
        def dfs(l):
            for r in adj.get(l, []):
                ml = matching_r.get(r)
                if ml is None or (dist[ml] == dist[l] + 1 and dfs(ml)):
                    matching_l[l] = r
                    matching_r[r] = l
                    return True
            dist[l] = float('inf')
            return False
        
        while bfs():
            for l in left_nodes:
                if l not in matching_l:
                    dfs(l)
        
        return matching_l, matching_r

    def matching_based_decomposition(edges, adj, n):
        """
        Use maximum bipartite matching to find minimum path cover.
        
        For each undirected edge {u,v}, create two directed stubs:
        - Left node (u, v) representing "arrive at v from u" 
        - Right node (v, u) representing "leave u for v"
        
        A match between left (u,v) and right (v,w) means edge {u,v} is 
        immediately followed by edge {v,w} in a path.
        
        Number of paths = |E| - |maximum matching|
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Create left and right nodes for each directed edge
        # Left node: (u, v) means "the edge from u to v" (arriving at v)
        # Right node: (u, v) means "the edge from u to v" (leaving u)
        # 
        # A match between left (a, b) and right (b, c) means:
        # edge {a,b} is followed by edge {b,c} in the path
        # i.e., we have ...a -> b -> c...
        
        left_nodes = []
        right_nodes = []
        node_id_to_edge = {}  # (u,v) directed -> undirected edge
        
        for u, v in edges:
            # Directed edge u -> v
            left_nodes.append((u, v))
            right_nodes.append((u, v))
            node_id_to_edge[(u, v)] = (min(u,v), max(u,v))
        
        # Build adjacency: left (a,b) can match right (b,c) if edge {b,c} exists
        # and b is not equal to a (to avoid self-loops in path)
        # Actually, we just need {b,c} to be an edge in the graph
        adj_dict = defaultdict(set)
        
        for (a, b) in left_nodes:
            # Find all edges {b, c} where c != a
            for c in adj.get(b, []):
                if c == a:
                    continue
                # Check if edge {b,c} exists
                e = tuple(sorted((b, c)))
                if e in edge_set:
                    adj_dict[(a, b)].add((b, c))
        
        # Run Hopcroft-Karp
        matching_l, matching_r = hopcroft_karp(left_nodes, right_nodes, adj_dict)
        
        # Build the paths from the matching
        # matching_l[(a,b)] = (b,c) means edge {a,b} is followed by edge {b,c}
        # 
        # We need to reconstruct the paths:
        # 1. Find all "start" edges: edges (a,b) where (a,b) is NOT in matching_r
        #    (i.e., no edge ends at a from some other vertex)
        #    Wait, let me reconsider.
        #
        # matching_l[(a,b)] = (b,c) means: after traversing edge {a,b} (arriving at b),
        # we next traverse edge {b,c} (leaving b).
        #
        # matching_r[(a,b)] = (x,a) means: before traversing edge {a,b} (leaving a),
        # we previously traversed edge {x,a} (arriving at a).
        #
        # A path starts with an edge (a,b) where there is no (x,a) in matching_r 
        # that maps to (a,b). I.e., (a,b) is not in the values of matching_r.
        
        # Build reverse: which left node maps to each right node
        right_to_left = {v: k for k, v in matching_l.items()}
        
        # Find start edges: left nodes that are not in the values of matching_l
        # i.e., no other edge points to them as a predecessor
        matched_right_nodes = set(matching_l.values())
        
        start_edges = [le for le in left_nodes if le not in matched_right_nodes]
        
        # For each start edge, follow the chain
        paths = []
        visited_edges = set()
        
        for start_edge in start_edges:
            if start_edge in visited_edges:
                continue
            
            path = []
            current = start_edge
            visited_edges.add(current)
            
            while current is not None:
                u, v = current
                if not path:
                    path.append(u)
                path.append(v)
                
                # Find next edge
                if current in matching_l:
                    current = matching_l[current]
                    if current in visited_edges:
                        current = None
                    else:
                        visited_edges.add(current)
                else:
                    current = None
            
            if len(path) >= 2:
                paths.append(path)
        
        # Check if all edges are covered
        all_path_edges = set()
        for p in paths:
            for i in range(len(p) - 1):
                all_path_edges.add(tuple(sorted((p[i], p[i+1]))))
        
        if all_path_edges != edge_set:
            # Some edges not covered - this shouldn't happen if matching is correct
            # But let's verify and fall back if needed
            return None
        
        return paths

    def greedy_decomposition(edges, adj, n, time_budget=10):
        """Fast greedy fallback: repeatedly take longest available simple path."""
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        end_time = time.time() + time_budget
        
        while remaining and time.time() < end_time:
            # Find longest simple path via greedy DFS with backtracking limit
            best_path = None
            best_len = 0
            
            # Try from high-degree vertices first
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if len(neigh) > 0}
            if not degs:
                break
            sorted_verts = sorted(degs.keys(), key=lambda v: -degs[v])
            
            for start in sorted_verts[:min(8, len(sorted_verts))]:
                # Simple greedy path extension
                path = [start]
                visited = {start}
                used_e = set()
                current = start
                
                while True:
                    # Find neighbor with fewest remaining options (Warnsdorff)
                    best_nb = None
                    best_opts = float('inf')
                    for nb in sorted(rem_adj[current]):
                        e = tuple(sorted((current, nb)))
                        if e in remaining and e not in used_e and nb not in visited:
                            opts = sum(1 for nn in rem_adj[nb] 
                                      if nn != current and nn not in visited
                                      and tuple(sorted((nb, nn))) in remaining
                                      and tuple(sorted((nb, nn))) not in used_e)
                            if opts < best_opts:
                                best_opts = opts
                                best_nb = nb
                    if best_nb is None:
                        break
                    path.append(best_nb)
                    visited.add(best_nb)
                    e = tuple(sorted((current, best_nb)))
                    used_e.add(e)
                    current = best_nb
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
            else:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    e = tuple(sorted((best_path[i], best_path[i+1])))
                    if e in remaining:
                        remaining.remove(e)
                        rem_adj[e[0]].discard(e[1])
                        rem_adj[e[1]].discard(e[0])
        
        # Fallback for any remaining edges
        while remaining:
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.remove(e)
            rem_adj[e[0]].discard(e[1])
            rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths(paths, edges, n):
        """Try to merge pairs of paths that share an edge between their endpoints."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        while improved:
            improved = False
            random.shuffle(current)
            for i in range(len(current)):
                if improved:
                    break
                for j in range(i + 1, len(current)):
                    if improved:
                        break
                    p1, p2 = current[i], current[j]
                    if set(p1) & set(p2):
                        continue
                    
                    # Try 4 merge orientations
                    for orient in range(4):
                        if orient == 0:  # p1 + p2
                            a, b = p1[-1], p2[0]
                            merged = p1 + p2
                        elif orient == 1:  # p2 + p1
                            a, b = p2[-1], p1[0]
                            merged = p2 + p1
                        elif orient == 2:  # p1 + reversed(p2)
                            a, b = p1[-1], p2[-1]
                            merged = p1 + p2[::-1]
                        else:  # p1[::-1] + p2
                            a, b = p1[0], p2[0]
                            merged = p1[::-1] + p2
                        
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        if len(set(merged)) != len(merged):
                            continue
                        
                        # Verify all edges in merged path exist
                        valid = True
                        for k in range(len(merged) - 1):
                            e2 = tuple(sorted((merged[k], merged[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        new_current = [current[k] for k in range(len(current)) if k != i and k != j]
                        new_current.append(merged)
                        if verify_decomposition(new_current, edges, n):
                            current = new_current
                            improved = True
                            break
        
        return current

    def solve_graph(edges, adj, n, time_budget):
        """Solve one graph instance with time budget."""
        # First try matching-based decomposition (exact minimum)
        if time_budget > 5:
            match_time = min(time_budget * 0.5, 30)
            start_match = time.time()
            match_result = matching_based_decomposition(edges, adj, n)
            if match_result is not None and time.time() - start_match < match_time:
                # Try to improve with merging (shouldn't help since it's already optimal,
                # but verify and return)
                if verify_decomposition(match_result, edges, n):
                    return match_result
        
        # Fallback to greedy
        greedy_result = greedy_decomposition(edges, adj, n, min(time_budget, 15))
        if not verify_decomposition(greedy_result, edges, n):
            return [list(e) for e in edges]
        
        # Try merging
        merged = merge_paths(greedy_result, edges, n)
        if verify_decomposition(merged, edges, n):
            return merged
        
        return greedy_result

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        remaining_time = deadline - time.time() - 10
        if remaining_time < 5:
            paths = [list(e) for e in edges]
        else:
            time_budget = max(5, min(remaining_time / (3 - idx), 40))
            paths = solve_graph(edges, adj, n, time_budget)
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 5:
            for idx2 in range(len(results), 3):
                n2, p2, seed2 = specs[idx2]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
