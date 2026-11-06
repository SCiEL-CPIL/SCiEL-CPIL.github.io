"""Path decomposition: Greedy + local search with path-merging via augmenting paths."""
import time
import sys
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

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
        return used_edges == all_edges

    def greedy_extract_paths(remaining_edges, adj, n, rng, max_starts=20):
        """Extract long simple paths greedily using Warnsdorff's heuristic."""
        remaining = set(remaining_edges)
        paths = []
        rem_adj = defaultdict(set)
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        while remaining:
            # Find a good starting vertex
            degs = {v: len(neigh) for v, neigh in rem_adj.items() if len(neigh) > 0}
            if not degs:
                break
            
            # Try multiple starting vertices
            candidates = sorted(degs.keys(), key=lambda v: (-degs[v], rng.random()))[:max_starts]
            
            best_path = None
            best_len = 0
            
            for start in candidates:
                if not remaining:
                    break
                
                path = [start]
                visited = {start}
                current = start
                
                while True:
                    best_nb = None
                    best_opts = float('inf')
                    for nb in rem_adj[current]:
                        e = tuple(sorted((current, nb)))
                        if e in remaining and nb not in visited:
                            # Warnsdorff: prefer neighbor with fewest options
                            opts = sum(1 for nn in rem_adj[nb] 
                                      if nn != current and nn not in visited
                                      and tuple(sorted((nb, nn))) in remaining)
                            if opts < best_opts:
                                best_opts = opts
                                best_nb = nb
                    if best_nb is None:
                        break
                    path.append(best_nb)
                    visited.add(best_nb)
                    current = best_nb
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
            
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
        
        # Handle any remaining edges
        while remaining:
            e = next(iter(remaining))
            paths.append([e[0], e[1]])
            remaining.remove(e)
            rem_adj[e[0]].discard(e[1])
            rem_adj[e[1]].discard(e[0])
        
        return paths

    def try_merge_paths(paths, edges, n, max_rounds=100):
        """Try to merge pairs of paths by finding a valid connection."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        
        improved = True
        round_num = 0
        while improved and round_num < max_rounds:
            improved = False
            round_num += 1
            
            for i in range(len(current)):
                if improved:
                    break
                for j in range(i + 1, len(current)):
                    if improved:
                        break
                    
                    p1, p2 = current[i], current[j]
                    if set(p1) & set(p2):
                        continue
                    
                    # Try all 4 orientations
                    for orient in range(4):
                        if orient == 0:
                            a, b = p1[-1], p2[0]
                            merged = p1 + p2
                        elif orient == 1:
                            a, b = p2[-1], p1[0]
                            merged = p2 + p1
                        elif orient == 2:
                            a, b = p1[-1], p2[-1]
                            merged = p1 + p2[::-1]
                        else:
                            a, b = p1[0], p2[0]
                            merged = p1[::-1] + p2
                        
                        e = tuple(sorted((a, b)))
                        if e not in edge_set:
                            continue
                        if len(set(merged)) != len(merged):
                            continue
                        
                        # Verify all edges exist
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

    def local_search_reduce_paths(paths, edges, adj, n, rng, time_budget=5):
        """Local search to reduce number of paths by splitting and recombining."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        current = [list(p) for p in paths]
        best = current[:]
        best_count = len(current)
        
        end_time = time.time() + time_budget
        no_improve = 0
        
        while time.time() < end_time and no_improve < 20:
            # Pick a random path and try to split it
            if len(current) < 2:
                break
            
            idx = rng.randint(0, len(current) - 1)
            path = current[idx]
            
            if len(path) < 3:
                no_improve += 1
                continue
            
            # Try splitting at a random position
            split_pos = rng.randint(1, len(path) - 2)
            part1 = path[:split_pos + 1]
            part2 = path[split_pos:]
            
            if len(part1) < 2 or len(part2) < 2:
                no_improve += 1
                continue
            
            # Create new set of paths with the split
            new_paths = [current[k] for k in range(len(current)) if k != idx]
            new_paths.append(part1)
            new_paths.append(part2)
            
            # Try to merge other paths with these two parts
            new_paths = try_merge_paths(new_paths, edges, n, max_rounds=20)
            
            if len(new_paths) < best_count:
                best = new_paths[:]
                best_count = len(new_paths)
                no_improve = 0
            else:
                no_improve += 1
        
        return best

    def solve_graph(edges, adj, n, time_budget, rng):
        """Solve one graph instance."""
        # Phase 1: Multiple greedy attempts, keep best
        greedy_time = min(time_budget * 0.3, 15)
        best_greedy = None
        best_count = float('inf')
        
        num_attempts = max(3, int(greedy_time / 2))
        attempt_budget = greedy_time / num_attempts
        
        for attempt in range(num_attempts):
            if time.time() > start_time + time_budget - 5:
                break
            paths = greedy_extract_paths(edges, adj, n, rng, max_starts=15)
            if verify_decomposition(paths, edges, n):
                # Try merging
                merged = try_merge_paths(paths, edges, n, max_rounds=50)
                if verify_decomposition(merged, edges, n):
                    if len(merged) < best_count:
                        best_count = len(merged)
                        best_greedy = merged[:]
        
        if best_greedy is None:
            return [list(e) for e in edges]
        
        # Phase 2: Local search
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 3:
            ls_time = min(remaining_time * 0.7, 20)
            refined = local_search_reduce_paths(best_greedy, edges, adj, n, rng, ls_time)
            if verify_decomposition(refined, edges, n) and len(refined) < len(best_greedy):
                best_greedy = refined
        
        # Phase 3: Final merge
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 2:
            merged = try_merge_paths(best_greedy, edges, n, max_rounds=min(100, int(remaining_time * 10)))
            if verify_decomposition(merged, edges, n) and len(merged) < len(best_greedy):
                best_greedy = merged
        
        return best_greedy

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        remaining_time = deadline - time.time() - 8
        if remaining_time < 5:
            paths = [list(e) for e in edges]
        else:
            # Allocate more time to larger graphs
            weights = [1.0, 1.1, 1.3]
            total_weight = sum(weights)
            time_budget = max(10, remaining_time * weights[idx] / total_weight)
            rng = random.Random(seed * 1000 + idx)
            paths = solve_graph(edges, adj, n, time_budget, rng)
        
        results.append(paths)
        
        if time.time() - start_time > deadline - 5:
            for idx2 in range(len(results), 3):
                n2, p2, seed2 = specs[idx2]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
