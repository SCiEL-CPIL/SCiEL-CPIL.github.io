import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    # Deadline guard: 150s limit, leave 10s safety margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges, rng, strategy='random'):
        """
        Runs a greedy path decomposition algorithm.
        Tries to build long paths by extending from both ends.
        Uses randomization in tie-breaking to explore different solutions.
        """
        if not edges:
            return []
        
        # Copy edges to a mutable set for removal
        remaining = set(edges)
        
        paths = []
        
        # Pre-compute adjacency list for faster neighbor lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Helper to get neighbors of a node in the remaining graph
        def get_neighbors(u):
            neighbors = []
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in remaining:
                    neighbors.append(v)
            return neighbors

        # Helper to remove edge
        def remove_edge(u, v):
            e = tuple(sorted((u, v)))
            if e in remaining:
                remaining.remove(e)

        # Helper to extend path
        def extend_path(path):
            # path is a list of vertices
            # Try to extend from both ends until no more extension is possible
            changed = True
            while changed:
                changed = False
                # Try extending the back end
                if len(path) > 0:
                    back = path[-1]
                    neighbors = get_neighbors(back)
                    # Filter neighbors that are not already in the path
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            # Prefer neighbors with lower remaining degree
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(get_neighbors(n))
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.append(next_node)
                        remove_edge(back, next_node)
                        changed = True
                
                # Try extending the front end
                if len(path) > 0:
                    front = path[0]
                    neighbors = get_neighbors(front)
                    valid_neighbors = [n for n in neighbors if n not in path]
                    if valid_neighbors:
                        if strategy == 'random':
                            next_node = rng.choice(valid_neighbors)
                        elif strategy == 'low_deg':
                            degs = {}
                            for n in valid_neighbors:
                                degs[n] = len(get_neighbors(n))
                            min_deg = min(degs.values())
                            candidates = [n for n in valid_neighbors if degs[n] == min_deg]
                            next_node = rng.choice(candidates)
                        else:
                            next_node = rng.choice(valid_neighbors)
                        path.insert(0, next_node)
                        remove_edge(front, next_node)
                        changed = True
                        
        # Main loop: construct paths
        while remaining:
            # Pick a starting edge
            # Heuristic: Pick an edge incident to a vertex with the highest degree
            # Calculate degrees
            degrees = defaultdict(int)
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            # Find vertex with max degree
            max_deg = max(degrees.values())
            candidates = [v for v, d in degrees.items() if d == max_deg]
            start_u = rng.choice(candidates)
            
            # Pick a neighbor for the starting edge
            neighbors = get_neighbors(start_u)
            if not neighbors:
                continue 
            
            start_v = rng.choice(neighbors)
            
            # Initialize path
            path = [start_u, start_v]
            remove_edge(start_u, start_v)
            
            # Extend path
            extend_path(path)
            
            paths.append(path)
            
            # Check deadline
            if time.time() > deadline:
                break

        return paths

    def merge_paths(paths, rng):
        """
        Merge paths that can be concatenated. Two paths can be merged if:
        - Path A ends at vertex v and Path B starts at vertex v (or vice versa)
        - The merged path remains simple (no repeated vertices)
        
        Returns the merged list of paths.
        """
        if len(paths) <= 1:
            return paths
        
        paths = [list(p) for p in paths]
        changed = True
        while changed:
            changed = False
            # Index paths by their start and end vertices
            by_start = defaultdict(list)
            by_end = defaultdict(list)
            for i, p in enumerate(paths):
                if len(p) >= 2:
                    by_start[p[0]].append(i)
                    by_end[p[-1]].append(i)
            
            # Try to find mergeable pairs
            merged = False
            for v in list(by_start.keys()):
                if v in by_end:
                    start_indices = by_start[v]
                    end_indices = by_end[v]
                    for si in start_indices:
                        for ei in end_indices:
                            if si == ei:
                                continue
                            # Try merging: path ending at v + path starting at v
                            path_a = paths[ei]  # ends at v
                            path_b = paths[si]  # starts at v
                            # Merge: path_a + path_b[1:]
                            merged_path = path_a + path_b[1:]
                            # Check simplicity
                            if len(set(merged_path)) == len(merged_path):
                                # Remove both paths and add merged
                                # Need to be careful with indices
                                del paths[max(si, ei)]
                                del paths[min(si, ei)]
                                paths.append(merged_path)
                                changed = True
                                merged = True
                                break
                        if merged:
                            break
                if merged:
                    break
            
            if not changed:
                # Also try the reverse: path starting at v + path ending at v
                changed = True
                while changed:
                    changed = False
                    by_start = defaultdict(list)
                    by_end = defaultdict(list)
                    for i, p in enumerate(paths):
                        if len(p) >= 2:
                            by_start[p[0]].append(i)
                            by_end[p[-1]].append(i)
                    
                    for v in list(by_start.keys()):
                        if v in by_end:
                            start_indices = by_start[v]
                            end_indices = by_end[v]
                            for si in start_indices:
                                for ei in end_indices:
                                    if si == ei:
                                        continue
                                    # Try merging: path_b (starts at v) reversed + path_a (ends at v)
                                    # Actually, we need: path starting at v can connect to path ending at v
                                    # path_b starts at v, path_a ends at v
                                    # Merge: path_b + path_a[1:] won't work directly
                                    # We need: reverse path_a, then path_a_rev starts at v
                                    # So: path_b + reverse(path_a)[1:]
                                    path_a = paths[ei]
                                    path_b = paths[si]
                                    # path_a ends at v, so reverse(path_a) starts at v
                                    path_a_rev = path_a[::-1]
                                    merged_path = path_b + path_a_rev[1:]
                                    if len(set(merged_path)) == len(merged_path):
                                        del paths[max(si, ei)]
                                        del paths[min(si, ei)]
                                        paths.append(merged_path)
                                        changed = True
                                        break
                                if changed:
                                    break
                        if changed:
                            break
        
        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Run multiple iterations with different random seeds for the greedy choices
        # The number of iterations depends on the time budget
        # We have ~140s. Let's allocate time proportionally to graph size?
        # Or just run a fixed number of fast iterations.
        # Graphs are small (N=20-30, E~100-200). One iteration is very fast (<1ms).
        # We can run thousands of iterations.
        
        iterations = 2000
        for i in range(iterations):
            if time.time() > deadline:
                break
            
            # Alternate between random and low_deg strategies
            strategy = 'random' if i % 2 == 0 else 'low_deg'
            rng = random.Random(seed * 1000 + i)
            paths = decompose_greedy(edges, rng, strategy)
            
            # Apply path merging to reduce the number of paths
            paths = merge_paths(paths, rng)
            
            # Verify validity (simple, edge-disjoint, cover all)
            # This is important to ensure we don't return an invalid solution
            # 1. Edge-disjoint and cover all:
            edge_set = set(edges)
            used_edges = set()
            valid = True
            for path in paths:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                    # Check if edge exists in original graph
                    if e not in edge_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(used_edges) == len(edges):
                # 2. Simple paths:
                for path in paths:
                    if len(set(path)) != len(path):
                        valid = False
                        break
            
            if valid:
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
            
            # If we reached the theoretical lower bound, we can stop early
            if best_count == 1:
                break

        if best_paths is None:
            # Fallback: return single edges if greedy failed completely (shouldn't happen)
            return [[list(e)] for e in edges]
            
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        if time.time() > deadline:
            # If we ran out of time, fill remaining with trivial solution?
            # But we should have finished.
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
         # This should not happen if time management is correct
         results.append([])

    return results

if __name__ == "__main__":
    # Quick test
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
