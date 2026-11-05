import time
import random
from collections import defaultdict, deque


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        return used_edges == edge_set

    def greedy_target(edges, k, seed_val, adj=None):
        """
        Attempts to cover all edges with exactly k simple paths.
        Returns the list of paths if successful, else None.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None

        rng = random.Random(seed_val)
        remaining = set(edges)
        
        if adj is None:
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)

        paths = [[] for _ in range(k)]

        # Pick k distinct edges to start k paths
        # Prefer edges incident to high-degree vertices or odd-degree vertices to help flow
        edge_list = list(remaining)
        rng.shuffle(edge_list)

        start_edges = edge_list[:k]

        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]

        # Extend paths greedily
        # We iterate until no more extensions are possible or remaining is empty
        changed = True
        while changed and remaining:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)

            for i in idxs:
                if not paths[i]:
                    continue

                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)

                if candidates:
                    # Heuristic: pick neighbor that leads to more remaining edges? 
                    # For now random choice, but we can weight by degree in remaining graph
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue

                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue

        if remaining:
            return None
        return paths

    def merge_paths(paths, max_iters=1000):
        """
        Iteratively merge paths that share endpoints.
        A merge is valid if the resulting path is simple (no repeated vertices).
        Returns the improved list of paths.
        """
        current_paths = [list(p) for p in paths]
        
        # Map vertex to list of (path_index, is_end)
        # is_end=True if vertex is the last element, False if first
        endpoint_map = defaultdict(list)
        
        def rebuild_endpoint_map():
            endpoint_map.clear()
            for i, p in enumerate(current_paths):
                if not p:
                    continue
                endpoint_map[p[0]].append((i, False))
                endpoint_map[p[-1]].append((i, True))
        
        rebuild_endpoint_map()
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to find a mergeable pair
            # We look for vertices that are endpoints of at least 2 paths
            merge_candidates = []
            for v, entries in endpoint_map.items():
                if len(entries) >= 2:
                    merge_candidates.append(v)
            
            if not merge_candidates:
                break
            
            # Randomize candidate order
            random.shuffle(merge_candidates)
            
            for v in merge_candidates:
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                # Get unique path indices
                path_indices = list(set([idx for idx, _ in entries]))
                
                if len(path_indices) < 2:
                    continue
                
                # Try to merge path_indices[0] and path_indices[1]
                idx1, idx2 = path_indices[0], path_indices[1]
                
                p1 = current_paths[idx1]
                p2 = current_paths[idx2]
                
                if not p1 or not p2:
                    continue
                
                # Determine how to merge
                # Case 1: p1 ends at v, p2 starts at v -> p1 + p2
                # Case 2: p1 ends at v, p2 ends at v -> p1 + reversed(p2)
                # Case 3: p1 starts at v, p2 starts at v -> reversed(p1) + p2
                # Case 4: p1 starts at v, p2 ends at v -> p2 + p1
                
                p1_end_v = (p1[-1] == v)
                p2_end_v = (p2[-1] == v)
                
                if p1_end_v and not p2_end_v:
                    new_path = p1 + p2
                elif p1_end_v and p2_end_v:
                    new_path = p1 + list(reversed(p2))
                elif not p1_end_v and not p2_end_v:
                    new_path = list(reversed(p1)) + p2
                else: # p1 starts at v, p2 ends at v
                    new_path = p2 + p1
                
                # Check simplicity
                if len(set(new_path)) == len(new_path):
                    # Valid merge
                    current_paths[idx1] = new_path
                    current_paths[idx2] = []
                    
                    improved = True
                    # Rebuild endpoint map for next iteration
                    # Actually, we can just break and rebuild to be safe
                    break
            
            if improved:
                # Clean up empty paths
                current_paths = [p for p in current_paths if p]
                rebuild_endpoint_map()
                if time.time() > deadline - 15:
                    break
        
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Compute lower bound
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, (odd_count + 1) // 2)
        
        # Allocate time per instance
        time_per_instance = 35.0
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy: 
        # 1. Try to hit the theoretical lower bound using greedy_target with multiple restarts.
        # 2. If that fails, try slightly higher targets.
        # 3. Use merge_paths on any valid solution found to see if we can reduce count further (though greedy_target should already be near optimal for that k).
        
        # We prioritize finding a solution with count == lower_bound.
        # If we find one, we stop early.
        
        max_targets = lower_bound + 5
        for target_k in range(lower_bound, min(len(edges), max_targets) + 1):
            if time.time() > inst_deadline - 2:
                break
            
            # If we already have a solution with count <= target_k, no need to try this target
            if best_count <= target_k:
                continue
                
            # Multiple restarts for this target
            # Budget restarts based on time remaining and target difficulty
            # Lower targets are harder, so maybe more restarts? Or just fixed budget.
            max_restarts = 1000
            found_for_target = False
            
            for restart in range(max_restarts):
                if time.time() > inst_deadline - 2:
                    break
                
                seed_val = seed * 100000 + restart * 7 + target_k
                
                # Greedy target
                paths = greedy_target(edges, target_k, seed_val, adj)
                
                if paths is not None:
                    # Validate to be safe
                    if validate_paths(paths, edges):
                        # Try to merge to see if we can get even fewer (unlikely if target_k is low, but possible if target_k > lower_bound)
                        # Actually, if target_k == lower_bound, we can't do better.
                        # If target_k > lower_bound, merging might reduce it.
                        merged = merge_paths(paths, max_iters=50)
                        
                        if validate_paths(merged, edges):
                            if len(merged) < best_count:
                                best_count = len(merged)
                                best_paths = merged
                                found_for_target = True
                                # If we hit the absolute lower bound, we are done
                                if best_count == lower_bound:
                                    return best_paths
                        else:
                            # Merging failed validation? Unlikely if input was valid and merge checks simplicity.
                            # But if it happens, keep original if it's better
                            if target_k < best_count:
                                best_count = target_k
                                best_paths = paths
                                found_for_target = True
                                if best_count == lower_bound:
                                    return best_paths

            # If we found a solution for this target_k, and it's the lower bound, we are done.
            # If not, we continue to next target_k? 
            # Actually, if we found a solution for k, we don't need to try k+1 unless we want to see if k+1 merges down to k-1? 
            # No, merging only reduces count. So if we have a solution of size K, trying K+1 might yield a solution that merges to K-1? 
            # It's possible. So we should keep trying higher targets if we haven't hit lower_bound.
            
        if best_paths is None:
            # Fallback: Greedy decomposition without target
            # This ensures we return *something* valid
            rng = random.Random(seed + 9999)
            remaining = set(edges)
            paths = []
            
            while remaining:
                if time.time() > inst_deadline:
                    break
                degs = defaultdict(int)
                for u, v in remaining:
                    degs[u] += 1
                    degs[v] += 1
                if not degs:
                    break
                max_d = max(degs.values())
                cands = [v for v, d in degs.items() if d == max_d]
                start_u = rng.choice(cands)
                
                neighbors = [nb for nb in adj[start_u]
                             if (start_u, nb) in remaining or (nb, start_u) in remaining]
                if not neighbors:
                    e = next(iter(remaining))
                    start_u, start_v = e
                else:
                    start_v = rng.choice(neighbors)
                
                path = [start_u, start_v]
                if (start_u, start_v) in remaining:
                    remaining.remove((start_u, start_v))
                else:
                    remaining.remove((start_v, start_u))
                
                changed = True
                while changed:
                    changed = False
                    back = path[-1]
                    cands = [nb for nb in adj[back]
                             if nb not in path and
                             ((back, nb) in remaining or (nb, back) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (back, nb) in remaining:
                            remaining.remove((back, nb))
                        else:
                            remaining.remove((nb, back))
                        path.append(nb)
                        changed = True
                        continue
                    front = path[0]
                    cands = [nb for nb in adj[front]
                             if nb not in path and
                             ((front, nb) in remaining or (nb, front) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (front, nb) in remaining:
                            remaining.remove((front, nb))
                        else:
                            remaining.remove((nb, front))
                        path.insert(0, nb)
                        changed = True
                        continue
                
                paths.append(path)
            
            if remaining:
                # Add remaining edges as single-edge paths
                for e in remaining:
                    paths.append([e[0], e[1]])
            
            # Try to merge these fallback paths
            merged = merge_paths(paths, max_iters=100)
            if validate_paths(merged, edges):
                best_paths = merged
            else:
                best_paths = paths

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            # Emergency fallback: single edges
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results


if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
