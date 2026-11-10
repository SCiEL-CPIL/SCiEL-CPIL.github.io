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

def _greedy_walk(adj, remaining, deg, start_v, rng=None, randomize=False):
    """
    Perform a single greedy walk from start_v, returning the path and updating
    remaining and deg in place.
    """
    path = [start_v]
    visited = {start_v}
    
    while True:
        v = path[-1]
        # Find valid neighbors
        valid_neighbors = []
        for u in adj[v]:
            if u not in visited:
                e = (min(v, u), max(v, u))
                if e in remaining:
                    valid_neighbors.append(u)
        
        if not valid_neighbors:
            break
        
        if randomize and rng is not None:
            # Weighted random choice: prefer low degree
            weights = [1.0 / (deg.get(u, 0) + 1) for u in valid_neighbors]
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

def _path_cover_single(edges, deadline, rng=None, randomize=False):
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
                start_v = min(candidates, key=lambda v: deg[v])
        
        path = _greedy_walk(adj, remaining, deg, start_v, rng, randomize)
        paths.append(path)
    
    # If we ran out of time, add remaining edges as single-edge paths
    for e in remaining:
        paths.append(list(e))
    
    return paths

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
    
    if time.time() > deadline - 5:
        return best_paths
    
    # Multiple randomized restarts
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        paths = _path_cover_single(edges, deadline - 5, rng, randomize=True)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
            # If we've hit a good number, we can stop early
            # Gallai's conjecture suggests ~n/2 paths. For n=20, ~10; n=25, ~13; n=30, ~15
            # If we get close to optimal, break.
            if best_count <= 12:  # Heuristic threshold for small graphs
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
    # Or just split evenly? Even split is safer.
    # Let's split the available time (140s) among the 3 graphs.
    # But some graphs might be harder.
    # Let's give each graph 45s, keeping 5s buffer.
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
