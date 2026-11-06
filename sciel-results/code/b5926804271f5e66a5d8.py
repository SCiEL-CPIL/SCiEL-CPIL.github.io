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

    def fast_greedy_decomposition(edges, adj, n):
        """
        Fast greedy decomposition:
        1. While edges remain, pick highest degree vertex.
        2. Walk along unused edges to neighbor with highest degree.
        3. Record path.
        """
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # Precompute edge lookup for fast removal
        # We need to track remaining edges efficiently
        
        while remaining:
            # Find vertex with max degree in remaining graph
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            
            if not degs:
                break
                
            start = max(degs.keys(), key=lambda v: degs[v])
            current = start
            path = [current]
            
            while True:
                # Find next vertex
                next_v = None
                best_deg = -1
                for nb in adj[current]:
                    e = tuple(sorted((current, nb)))
                    if e in remaining:
                        # Check degree of nb in remaining
                        # This is expensive if done naively, but n is small
                        # We can just pick the first one or the one with max degree
                        # Let's pick the one with max degree to try to make long paths
                        if degs[nb] > best_deg:
                            best_deg = degs[nb]
                            next_v = nb
                
                if next_v is None:
                    break
                
                # Move to next_v
                remaining.remove(tuple(sorted((current, next_v))))
                path.append(next_v)
                current = next_v
                
                # Update degrees for removed edge?
                # For simplicity, we recompute degrees at the start of the outer loop.
                # Inside the inner loop, we use the stale `degs` which is okay for heuristic.
                
            if len(path) > 1:
                paths.append(path)
            else:
                # Should not happen if remaining is not empty and start has deg > 0
                # But just in case, take one edge
                for e in list(remaining):
                    if start in e:
                        other = e[0] if e[1] == start else e[1]
                        paths.append([start, other])
                        remaining.remove(e)
                        break
        
        return paths

    def merge_paths(paths, edges, n):
        """
        Local search to merge paths that share endpoints.
        """
        current_paths = [list(p) for p in paths]
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Map vertex to list of (path_index, is_end)
        # We only care about endpoints
        endpoint_map = defaultdict(list)
        
        def update_map(path_idx, path, add=True):
            if not path: return
            start, end = path[0], path[-1]
            if add:
                endpoint_map[start].append((path_idx, True))
                endpoint_map[end].append((path_idx, False))
            else:
                # Remove
                endpoint_map[start] = [(i, t) for i, t in endpoint_map[start] if i != path_idx]
                endpoint_map[end] = [(i, t) for i, t in endpoint_map[end] if i != path_idx]
        
        for i, p in enumerate(current_paths):
            update_map(i, p, True)
            
        merged = True
        while merged:
            merged = False
            # Try to find a merge
            for v in list(endpoint_map.keys()):
                entries = endpoint_map[v]
                if len(entries) < 2:
                    continue
                
                # Find two entries that can be merged
                # One must be an end (False) and one a start (True) to merge directly?
                # Actually, if P1 ends at v and P2 starts at v, we can merge P1 + P2[1:]
                # If P1 ends at v and P2 ends at v, we can merge P1 + reverse(P2)[1:]
                # If P1 starts at v and P2 starts at v, we can merge reverse(P1) + P2[1:]
                
                # Just try all pairs
                for i in range(len(entries)):
                    idx1, is_start1 = entries[i]
                    for j in range(i+1, len(entries)):
                        idx2, is_start2 = entries[j]
                        
                        p1 = current_paths[idx1]
                        p2 = current_paths[idx2]
                        
                        # Determine merge orientation
                        # We want to connect an end of p1 to a start of p2 (or vice versa)
                        # Case 1: p1 ends at v, p2 starts at v
                        if not is_start1 and is_start2:
                            # p1 = [..., v], p2 = [v, ...]
                            if v in p2[1:]: # Check if v appears elsewhere in p2 (shouldn't for simple path)
                                continue
                            if any(x in p1 for x in p2[1:]): # Check disjointness
                                continue
                            new_path = p1 + p2[1:]
                        # Case 2: p1 starts at v, p2 ends at v
                        elif is_start1 and not is_start2:
                            # p1 = [v, ...], p2 = [..., v]
                            if any(x in p2[:-1] for x in p1[1:]):
                                continue
                            new_path = p1[1:] + p2
                        # Case 3: p1 ends at v, p2 ends at v
                        elif not is_start1 and not is_start2:
                            # p1 = [..., v], p2 = [..., v]
                            if any(x in p1 for x in p2[:-1]):
                                continue
                            new_path = p1 + p2[::-1][1:]
                        # Case 4: p1 starts at v, p2 starts at v
                        else:
                            # p1 = [v, ...], p2 = [v, ...]
                            if any(x in p2 for x in p1[1:]):
                                continue
                            new_path = p1[::-1][1:] + p2
                            
                        # Verify new path is simple and valid
                        if len(set(new_path)) == len(new_path):
                            # Check edges
                            valid = True
                            for k in range(len(new_path)-1):
                                e = tuple(sorted((new_path[k], new_path[k+1])))
                                if e not in edge_set:
                                    valid = False
                                    break
                            if valid:
                                # Perform merge
                                # Remove old paths
                                # Update indices
                                # This is tricky with indices.
                                # Simpler: just create new list
                                new_current = []
                                for idx, p in enumerate(current_paths):
                                    if idx == idx1 or idx == idx2:
                                        continue
                                    new_current.append(p)
                                new_current.append(new_path)
                                
                                # Verify full decomposition
                                if verify_decomposition(new_current, edges, n):
                                    current_paths = new_current
                                    # Rebuild endpoint map
                                    endpoint_map.clear()
                                    for i, p in enumerate(current_paths):
                                        update_map(i, p, True)
                                    merged = True
                                    break
                    if merged:
                        break
            if not merged:
                break
                
        return current_paths

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

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        adj = get_adjacency(edges)
        
        # 1. Fast greedy
        paths = fast_greedy_decomposition(edges, adj, n)
        
        # 2. Merge
        if time.time() < deadline - 10:
            paths = merge_paths(paths, edges, n)
            
        # 3. Fallback if invalid (should not happen)
        if not verify_decomposition(paths, edges, n):
            paths = [list(e) for e in edges]
            
        results.append(paths)
        
        if time.time() > deadline - 5:
            for _ in range(3 - len(results)):
                n2, p2, seed2 = specs[len(results)]
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
