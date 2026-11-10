"""Model-and-optimize edge path cover for three fixed graphs."""
import time
import random
import sys
from collections import defaultdict

def _instance(n, p, seed):
    """Exact copy of the verifier's instance generator."""
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False, strategy="degree"):
    """
    Perform a single greedy walk from start_v, returning the path and updating
    remaining and deg in place.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            if strategy == "degree":
                weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
            elif strategy == "mixed":
                # Mix of degree and random
                weights = [1.0 / (deg.get(u, 0) + 1) + rng.random() * 0.5 for u in valid_neighbors]
            else:
                weights = [1.0 for u in valid_neighbors]
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            next_v = valid_neighbors[-1]
            for i, u in enumerate(valid_neighbors):
                cum += weights[i]
                if r <= cum:
                    next_v = u
                    break
        else:
            # Deterministic: pick neighbor with fewest remaining degree
            next_v = min(valid_neighbors, key=lambda u: deg.get(u, 0))
        
        e = (min(v, next_v), max(v, next_v))
        remaining.discard(e)
        deg[v] -= 1
        deg[next_v] -= 1
        path.append(next_v)
        visited.add(next_v)
        
    return path

def _path_cover_single(edges, deadline, rng=None, randomize=False, strategy="degree"):
    """
    Run a single pass of the greedy path cover algorithm.
    Returns list of paths.
    """
    adj = _build_graph(edges)
    remaining = set(map(tuple, edges))
    paths = []
    
    deg = defaultdict(int)
    for e in remaining:
        deg[e[0]] += 1
        deg[e[1]] += 1
    
    while remaining and time.time() < deadline:
        # Find start vertex
        start_candidates = set()
        for e in remaining:
            start_candidates.add(e[0])
            start_candidates.add(e[1])
        
        leaves = [v for v in start_candidates if deg[v] == 1]
        if leaves:
            if randomize and rng is not None:
                start_v = rng.choice(leaves)
            else:
                start_v = leaves[0]
        else:
            min_deg = min(deg[v] for v in start_candidates)
            candidates = [v for v in start_candidates if deg[v] == min_deg]
            if randomize and rng is not None:
                start_v = rng.choice(candidates)
            else:
                start_v = min(candidates)
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize, strategy)
        paths.append(path)
    
    # If we ran out of time, add remaining edges as single-edge paths
    for e in remaining:
        paths.append(list(e))
    
    return paths

def _try_merge_paths(paths, edges_set):
    """
    Try to merge two paths into one by finding a common edge.
    Returns (merged_paths, success)
    """
    if len(paths) < 2:
        return paths, False
    
    # Build edge-to-path-index mapping
    edge_to_path = {}
    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            e = (min(path[j], path[j+1]), max(path[j], path[j+1]))
            edge_to_path[e] = i
    
    # Try to find two paths that can be merged
    # Two paths can be merged if they share a vertex and we can reorder
    # This is complex, so let's use a simpler heuristic
    # Try to find paths where the end of one is the start of another (or vice versa)
    
    # For each pair of paths, check if they can be concatenated
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1 = paths[i]
            p2 = paths[j]
            
            # Check if p1's end is p2's start
            if p1[-1] == p2[0]:
                # Verify all edges in p1 + p2 are valid
                merged = p1 + p2[1:]
                valid = True
                for k in range(len(merged) - 1):
                    e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                    if e not in edges_set:
                        valid = False
                        break
                    # Check if edge is used in another path
                    if edge_to_path.get(e) not in (i, j, None):
                        valid = False
                        break
                
                if valid:
                    new_paths = [p for idx, p in enumerate(paths) if idx not in (i, j)]
                    new_paths.append(merged)
                    return new_paths, True
            
            # Check if p1's start is p2's end
            if p1[0] == p2[-1]:
                merged = p2 + p1[1:]
                valid = True
                for k in range(len(merged) - 1):
                    e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                    if e not in edges_set:
                        valid = False
                        break
                    if edge_to_path.get(e) not in (i, j, None):
                        valid = False
                        break
                
                if valid:
                    new_paths = [p for idx, p in enumerate(paths) if idx not in (i, j)]
                    new_paths.append(merged)
                    return new_paths, True
            
            # Check if p2's end is p1's start
            if p2[-1] == p1[0]:
                merged = p2 + p1[1:]
                valid = True
                for k in range(len(merged) - 1):
                    e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                    if e not in edges_set:
                        valid = False
                        break
                    if edge_to_path.get(e) not in (i, j, None):
                        valid = False
                        break
                
                if valid:
                    new_paths = [p for idx, p in enumerate(paths) if idx not in (i, j)]
                    new_paths.append(merged)
                    return new_paths, True
            
            # Check if p2's start is p1's end
            if p2[0] == p1[-1]:
                merged = p2 + p1[1:]
                valid = True
                for k in range(len(merged) - 1):
                    e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                    if e not in edges_set:
                        valid = False
                        break
                    if edge_to_path.get(e) not in (i, j, None):
                        valid = False
                        break
                
                if valid:
                    new_paths = [p for idx, p in enumerate(paths) if idx not in (i, j)]
                    new_paths.append(merged)
                    return new_paths, True
    
    return paths, False

def _path_cover_with_restarts(edges, deadline):
    """
    Try multiple random orderings of the greedy process to find a better cover.
    """
    best_paths = None
    best_count = len(edges)
    start = time.time()
    
    # First run: deterministic
    paths = _path_cover_single(edges, deadline, randomize=False)
    best_paths = paths
    best_count = len(paths)
    
    # Try to merge paths
    edges_set = set(map(tuple, edges))
    merged_paths, success = _try_merge_paths(best_paths, edges_set)
    if success and len(merged_paths) < best_count:
        best_paths = merged_paths
        best_count = len(merged_paths)
    
    if time.time() > deadline - 5:
        return best_paths
    
    # Multiple randomized restarts
    rng = random.Random(42)
    strategies = ["degree", "mixed", "random"]
    
    for trial in range(150):
        if time.time() > deadline - 10:
            break
        
        strategy = strategies[trial % len(strategies)]
        paths = _path_cover_single(edges, deadline - 5, rng, randomize=True, strategy=strategy)
        
        # Try to merge paths
        merged_paths, success = _try_merge_paths(paths, edges_set)
        if success:
            paths = merged_paths
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            # If we've hit a good number, we can stop early
            if best_count <= 12:
                break
    
    return best_paths

def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    Uses a model-and-optimize approach with deadline guarding.
    """
    # 150s wall-clock limit. Use 140s to be safe.
    deadline = time.time() + 140
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time roughly proportional to graph size (edge count)
    time_per_graph = 45.0
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        graph_deadline = time.time() + time_per_graph
        paths = _path_cover_with_restarts(edges, graph_deadline)
        results.append(paths)
    
    return results

if __name__ == "__main__":
    result = construct()
    for i, paths in enumerate(result):
        print(f"Graph {i}: {len(paths)} paths")
    print(f"Total: {sum(len(p) for p in result)}")
