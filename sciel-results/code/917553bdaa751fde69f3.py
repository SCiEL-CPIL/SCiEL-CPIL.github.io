import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_initial_decomposition(edges, seed):
        """
        Greedy construction: repeatedly find the longest available path 
        starting from the highest degree vertex.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        while remaining:
            # Pick start node with highest degree in remaining graph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            # Start from a high-degree node
            start = max(deg.keys(), key=lambda x: (deg[x], rng.random()))
            
            path = [start]
            visited = {start}
            used_edges = set()
            
            while True:
                current = path[-1]
                # Find next node
                next_node = None
                best_score = -1
                # We prefer moving to a node that keeps options open (not dead end)
                # but mainly we just need to extend the path.
                candidates = []
                for v in adj[current]:
                    e = (min(current, v), max(current, v))
                    if e in remaining and v not in visited:
                        # Heuristic: prefer nodes with higher degree in remaining graph?
                        # Or just random?
                        # Let's try to extend as far as possible.
                        candidates.append(v)
                
                if not candidates:
                    break
                
                # Simple strategy: pick random valid neighbor to avoid getting stuck in bad local minima?
                # Or pick the one with highest degree to keep path "dense"?
                # For simple path cover, extending is key.
                # Let's pick the neighbor that has the most unvisited neighbors (to avoid dead ends).
                best_next = None
                max_future = -1
                for v in candidates:
                    future = 0
                    for w in adj[v]:
                        e2 = (min(v, w), max(v, w))
                        if e2 in remaining and w not in visited:
                            future += 1
                    if future > max_future:
                        max_future = future
                        best_next = v
                
                if best_next is None:
                    break
                    
                path.append(best_next)
                visited.add(best_next)
                e = (min(current, best_next), max(current, best_next))
                used_edges.add(e)
                current = best_next
                
            if len(path) > 1:
                paths.append(path)
                for e in used_edges:
                    remaining.discard(e)
            else:
                # Isolated edge
                if remaining:
                    e = next(iter(remaining))
                    paths.append(list(e))
                    remaining.discard(e)
                    
        return paths

    def merge_paths(paths, edge_set):
        """
        Try to merge two paths if their endpoints are connected by an edge 
        and the merged path is simple (no repeated vertices).
        Returns new list of paths if a merge was successful, else None.
        """
        n_paths = len(paths)
        # We look for pairs (i, j) such that an endpoint of i is connected to an endpoint of j.
        # Since we want to minimize count, any valid merge reduces count by 1.
        
        # To speed up, we can index endpoints.
        # But n is small (<30), so O(N^2) check is fine.
        
        for i in range(n_paths):
            p1 = paths[i]
            if len(p1) < 2: continue
            e1_start = p1[0]
            e1_end = p1[-1]
            set1 = set(p1)
            
            for j in range(n_paths):
                if i == j: continue
                p2 = paths[j]
                if len(p2) < 2: continue
                e2_start = p2[0]
                e2_end = p2[-1]
                
                # Check 4 possible connections
                # 1. p1_end connected to p2_start
                if (min(e1_end, e2_start), max(e1_end, e2_start)) in edge_set:
                    # Check simplicity: e1_end not in p2 (except maybe if it's start? No, p1_end is not in p1's interior, 
                    # and p1_end != e2_start because paths are edge-disjoint and simple, 
                    # wait, can they share vertices? Yes. 
                    # If e1_end == e2_start, they are already "connected" at a vertex, but we need an EDGE.
                    # If e1_end == e2_start, the edge is (e1_end, next_in_p2). 
                    # The condition above checks if edge exists between the specific endpoints.
                    # If e1_end == e2_start, the "edge" is a self-loop? No, simple graph.
                    # So e1_end != e2_start.
                    
                    # New path: p1 + p2[1:]
                    # Check if any vertex in p2[1:] is in p1.
                    # p2[1:] are vertices in p2 excluding start.
                    # We need set(p1) and set(p2[1:]) to be disjoint.
                    if not (set1 & set(p2[1:])):
                        new_p = p1 + p2[1:]
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_p)
                        return new_paths

                # 2. p1_end connected to p2_end
                if (min(e1_end, e2_end), max(e1_end, e2_end)) in edge_set:
                    # New path: p1 + reversed(p2)
                    # p1 + [e2_end, ..., e2_start]
                    # Vertices in reversed(p2) are all vertices of p2.
                    # We need set(p1) and set(p2) to be disjoint?
                    # Wait. p1 ends at e1_end. p2 ends at e2_end.
                    # Connection is e1_end -- e2_end.
                    # New path: p1 + [e2_end, ..., e2_start].
                    # e2_end is the first new vertex.
                    # We need set(p1) and set(p2) to be disjoint?
                    # If they share a vertex, the new path would repeat it.
                    # Yes, for a simple path, all vertices must be unique.
                    if not (set1 & set(p2)):
                        new_p = p1 + p2[::-1]
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_p)
                        return new_paths

                # 3. p1_start connected to p2_start
                if (min(e1_start, e2_start), max(e1_start, e2_start)) in edge_set:
                    # New path: reversed(p1) + p2
                    # [e1_start, ..., e1_end] is reversed p1? No.
                    # p1 is [v0, ..., vk]. e1_start=v0, e1_end=vk.
                    # Connection v0 -- w0 (p2_start).
                    # New path: [v0, w0, ...] ? No, edge is (v0, w0).
                    # So we can do p2 + reversed(p1)?
                    # p2: [w0, ..., wm].
                    # reversed(p1): [vk, ..., v0].
                    # Connection vk -- v0? No, edge is v0-w0.
                    # So we can do p1 + p2? No, p1 starts with v0. p2 starts with w0.
                    # If we reverse p1, it ends with v0.
                    # So reversed(p1) + p2 = [vk, ..., v0, w0, ..., wm].
                    # This is valid if edge (v0, w0) exists.
                    if not (set1 & set(p2)):
                        new_p = p1[::-1] + p2
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_p)
                        return new_paths

                # 4. p1_start connected to p2_end
                if (min(e1_start, e2_end), max(e1_start, e2_end)) in edge_set:
                    # New path: p2 + reversed(p1)
                    # p2: [w0, ..., wm=e2_end].
                    # reversed(p1): [vk, ..., v0=e1_start].
                    # Connection wm -- v0.
                    # p2 + reversed(p1) = [w0, ..., wm, vk, ..., v0].
                    if not (set1 & set(p2)):
                        new_p = p2 + p1[::-1]
                        new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                        new_paths.append(new_p)
                        return new_paths

        return None

    def local_search_decompose(edges, seed):
        """
        Starts with a greedy decomposition and improves it via random merging.
        """
        # Try a few different initial seeds to get diverse starting points
        best_paths = None
        best_count = float('inf')
        
        for init_seed in range(5):
            paths = get_initial_decomposition(edges, seed + init_seed)
            
            # Local search: repeatedly try to merge random pairs
            # We can be smarter: try all pairs? N is small.
            # If N=30, max paths ~15. Pairs ~100. Check is fast.
            
            improved = True
            while improved and time.time() < DEADLINE - 2:
                improved = False
                # Try to find any valid merge
                # To avoid bias, shuffle paths order
                current_paths = paths[:]
                random.Random(seed + init_seed + int(time.time()*1000)).shuffle(current_paths)
                
                new_paths = merge_paths(current_paths, set(edges))
                if new_paths is not None:
                    paths = new_paths
                    improved = True
                    best_count = len(paths)
                    if best_paths is None or len(paths) < len(best_paths):
                        best_paths = paths[:]
            
            if best_paths is None or len(paths) < len(best_paths):
                best_paths = paths[:]
                best_count = len(paths)

        return best_paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        
        # Run local search
        paths = local_search_decompose(edges, spec[2])
        
        # Fallback if something went wrong (shouldn't happen)
        if paths is None:
             paths = get_initial_decomposition(edges, 0)
             
        result.append(paths)
        
    return result

# EVOLVE-BLOCK-END
