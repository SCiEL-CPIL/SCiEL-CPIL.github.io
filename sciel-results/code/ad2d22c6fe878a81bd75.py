"""Program for path decomposition of fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_decompose(edge_set, n, rng, deadline):
        """
        Greedy decomposition: repeatedly find a long path, remove its edges.
        Uses randomization to escape local minima.
        """
        if not edge_set:
            return []
        
        remaining = set(edge_set)
        paths = []
        
        # Build adjacency dynamically
        adj = build_adjacency(remaining, n)
        
        while remaining and time.time() < deadline:
            # Find a vertex with highest degree to start from
            # Prefer odd-degree vertices in remaining graph
            rem_odd = [u for u in adj if len(adj[u]) % 2 == 1 and adj[u]]
            if rem_odd:
                candidates = sorted(rem_odd, key=lambda v: -len(adj[v]))[:3]
                start = rng.choice(candidates)
            else:
                candidates = [u for u in adj if adj[u]]
                if not candidates:
                    break
                candidates = sorted(candidates, key=lambda v: -len(adj[v]))[:3]
                start = rng.choice(candidates)
            
            if not adj[start]:
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.remove(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
                continue
            
            # Greedily extend path
            path = [start]
            visited = {start}
            current = start
            
            for _ in range(n - 1):
                # Find neighbor with fewest remaining edges (to preserve connectivity)
                best_neighbor = None
                best_degree = float('inf')
                
                neighbors = [v for v in adj[current] if (min(current, v), max(current, v)) in remaining]
                if not neighbors:
                    break
                
                non_cycle = [v for v in neighbors if v not in visited]
                if non_cycle:
                    # Sort by degree, pick with some randomness
                    non_cycle.sort(key=lambda v: len(adj[v]))
                    if rng.random() < 0.3 and len(non_cycle) > 1:
                        idx = rng.randrange(1, min(3, len(non_cycle)))
                        best_neighbor = non_cycle[idx]
                    else:
                        best_neighbor = non_cycle[0]
                else:
                    break
                
                if best_neighbor is None:
                    break
                
                e = tuple(sorted((current, best_neighbor)))
                if e not in remaining:
                    break
                
                remaining.remove(e)
                adj[current].discard(best_neighbor)
                adj[best_neighbor].discard(current)
                if not adj[current]:
                    del adj[current]
                if not adj[best_neighbor]:
                    del adj[best_neighbor]
                
                path.append(best_neighbor)
                visited.add(best_neighbor)
                current = best_neighbor
            
            if len(path) > 1:
                paths.append(path)
        
        # Handle any leftover edges
        while remaining and time.time() < deadline:
            e = next(iter(remaining))
            u, v = e
            paths.append([u, v])
            remaining.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj.get(u, set()):
                adj.pop(u, None)
            if not adj.get(v, set()):
                adj.pop(v, None)
        
        return paths

    def merge_paths(paths, adj_original, deadline):
        """
        Iteratively merge paths if they share an endpoint and the union is simple.
        This is the key structural improvement from inspiration 1.
        """
        if len(paths) <= 1:
            return paths
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                
                path_i = paths[i]
                start_i = path_i[0]
                end_i = path_i[-1]
                
                merged = False
                for j in range(len(paths)):
                    if j == i or used[j]:
                        continue
                    
                    path_j = paths[j]
                    start_j = path_j[0]
                    end_j = path_j[-1]
                    
                    # Check four possible merging patterns
                    candidate = None
                    
                    # Option 1: end_i == start_j -> path_i + path_j[1:]
                    if end_i == start_j:
                        cand = path_i + path_j[1:]
                        if len(set(cand)) == len(cand):
                            candidate = cand
                    
                    # Option 2: end_i == end_j -> path_i + reversed(path_j[1:])
                    if end_i == end_j:
                        cand = path_i + list(reversed(path_j[1:]))
                        if len(set(cand)) == len(cand):
                            candidate = cand
                    
                    # Option 3: start_i == start_j -> reversed(path_i[1:]) + path_j
                    if start_i == start_j:
                        cand = list(reversed(path_i[1:])) + path_j
                        if len(set(cand)) == len(cand):
                            candidate = cand
                    
                    # Option 4: start_i == end_j -> path_j + path_i[1:]
                    if start_i == end_j:
                        cand = path_j + path_i[1:]
                        if len(set(cand)) == len(cand):
                            candidate = cand
                    
                    if candidate is not None:
                        new_paths.append(candidate)
                        used[i] = True
                        used[j] = True
                        merged = True
                        changed = True
                        break
                
                if not merged:
                    new_paths.append(path_i)
                    used[i] = True
            
            paths = new_paths
        
        return paths

    def decompose_graph(edges, n, time_budget):
        """
        Combined approach: multi-start greedy + iterative merging.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Compute degrees to get theoretical lower bound
        degree = [0] * n
        for u, v in edge_set:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        min_paths = max(1, len(odd_verts) // 2)
        
        adj_original = build_adjacency(edge_set, n)
        
        best_paths = None
        best_count = float('inf')
        
        # Multi-start with different random seeds
        max_restarts = 30
        deadline = start_time + time_budget
        
        for trial in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = Random(trial * 7919 + 13)
            raw_paths = greedy_decompose(edge_set, n, rng, deadline)
            
            # Merge paths
            merged_paths = merge_paths(raw_paths, adj_original, deadline)
            
            count = len(merged_paths)
            if count < best_count:
                best_count = count
                best_paths = merged_paths
            
            # Early termination if we hit the theoretical lower bound
            if best_count == min_paths:
                break
        
        if best_paths is None:
            # Fallback: single-edge paths
            best_paths = [[u, v] for u, v in edge_set]
        
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return remaining as single edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        # Allocate time budget per instance
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        # Distribute time: give more to later instances if time permits
        per_instance_time = min(remaining_time / 3, 30)
        
        # Try decomposition
        paths = decompose_graph(edges, n, per_instance_time)
        
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
