"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import defaultdict, deque
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
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
            # Check simple path
            if len(set(path)) != len(path):
                return False
            # Check edges exist
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                if e not in all_edges:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges covered
        if used_edges != all_edges:
            return False
        
        return True

    def greedy_trail_decomposition(edges, adj, n, max_time=40):
        """
        Decompose graph into edge-disjoint paths using greedy trail walking.
        Each "path" is a trail (no repeated edges) that may revisit vertices.
        We then convert trails to simple paths by splitting at repeated vertices.
        
        Actually, for this task we need SIMPLE paths. So let's use a different approach:
        Repeatedly find a longest simple path in the remaining graph.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Build adjacency for remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining:
            # Find vertex with highest degree in remaining graph
            if not remaining:
                break
            
            degs = {v: len(neigh) for v, neigh in rem_adj.items()}
            if not degs:
                break
            start = max(degs.keys(), key=lambda v: degs[v])
            
            # Find a long simple path starting from 'start'
            # Use iterative DFS with backtracking, but limit depth
            best_path = None
            best_len = 0
            
            # Try multiple starting points for diversity
            candidates = sorted(degs.keys(), key=lambda v: -degs[v])[:5]
            
            for cand_start in candidates:
                if not remaining:
                    break
                
                # DFS to find longest simple path from cand_start
                visited = set()
                used_e = set()
                current_path = [cand_start]
                visited.add(cand_start)
                
                # Use stack-based DFS with greedy extension
                stack = [(cand_start, [cand_start], set())]
                found_path = None
                
                # Simple greedy: walk as far as possible
                path = [cand_start]
                visited_set = {cand_start}
                current = cand_start
                used_edges_local = set()
                
                while True:
                    extended = False
                    # Sort neighbors by remaining degree (descending)
                    neighbors = list(rem_adj[current])
                    neighbors.sort(key=lambda nb: -len(rem_adj[nb]))
                    
                    for nb in neighbors:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and e not in used_edges_local and nb not in visited_set:
                            path.append(nb)
                            visited_set.add(nb)
                            used_edges_local.add(e)
                            current = nb
                            extended = True
                            break
                    
                    if not extended:
                        break
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            
            if best_path is None or len(best_path) < 2:
                # Shouldn't happen if remaining is non-empty, but handle it
                # Pick any edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                rem_adj[e[0]].discard(e[1])
                rem_adj[e[1]].discard(e[0])
                continue
            
            # Add the path and remove its edges
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                if e in remaining:
                    remaining.remove(e)
                    rem_adj[e[0]].discard(e[1])
                    rem_adj[e[1]].discard(e[0])
        
        return paths

    def merge_paths(paths, edges, adj, n, max_iterations=100):
        """Try to merge pairs of paths to reduce total count."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        for _ in range(max_iterations):
            merged = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    p1, p2 = current[i], current[j]
                    
                    # Try 4 ways to merge
                    for a, b, orientation in [
                        (p1[-1], p2[0], 'normal'),
                        (p1[0], p2[-1], 'normal'),
                        (p1[-1], p2[-1], 'reverse'),
                        (p1[0], p2[0], 'reverse')
                    ]:
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        
                        # Check no vertex overlap
                        if set(p1) & set(p2):
                            continue
                        
                        # Build merged path
                        if orientation == 'normal':
                            if a == p1[-1] and b == p2[0]:
                                merged_path = p1 + p2
                            elif a == p1[0] and b == p2[-1]:
                                merged_path = p2 + p1
                        else:  # reverse
                            if a == p1[-1] and b == p2[-1]:
                                merged_path = p1 + p2[::-1]
                            elif a == p1[0] and b == p2[0]:
                                merged_path = p1[::-1] + p2
                        
                        # Verify simplicity
                        if len(set(merged_path)) != len(merged_path):
                            continue
                        
                        # Verify all edges exist
                        valid = True
                        for k in range(len(merged_path) - 1):
                            e2 = tuple(sorted((merged_path[k], merged_path[k+1])))
                            if e2 not in edge_set:
                                valid = False
                                break
                        
                        if valid:
                            # Create new list without i and j
                            new_current = []
                            for idx in range(len(current)):
                                if idx != i and idx != j:
                                    new_current.append(current[idx])
                            new_current.append(merged_path)
                            
                            # Verify full decomposition
                            if verify_decomposition(new_current, edges, n):
                                current = new_current
                                merged = True
                                break
                    
                    if merged:
                        break
                
                if merged:
                    break
            
            if not merged:
                break
        
        return current

    def reduce_paths_by_splitting(paths, edges, n):
        """
        If a path is too long, we might be able to split it differently.
        This is complex; skip for now.
        """
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # Get initial decomposition
        paths = greedy_trail_decomposition(edges, adj, n)
        
        # Verify it's valid
        if not verify_decomposition(paths, edges, n):
            # Fallback: each edge as a path
            paths = [list(e) for e in edges]
        
        # Try to merge paths to reduce count
        if len(paths) > 1:
            merged = merge_paths(paths, edges, adj, n, max_iterations=50)
            if verify_decomposition(merged, edges, n):
                paths = merged
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 10:
            # Run out of time, use fallback for remaining
            for idx in range(len(results), 3):
                n2, p2, seed2 = specs[idx]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
