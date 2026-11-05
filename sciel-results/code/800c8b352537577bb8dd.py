"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def _instance(n, p, seed):
    """Generate the instance graph (same as verifier)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_graph(edges, time_budget):
    """
    Decompose a connected graph into edge-disjoint simple paths covering all edges.
    Strategy:
      1. Compute the theoretical lower bound: ceil(odd_degree_vertices / 2).
      2. Try to find a decomposition with exactly that many paths via randomized
         greedy / local search.
      3. If that fails, fall back to a valid (possibly suboptimal) decomposition.
    """
    start_time = time.time()
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Count odd-degree vertices
    odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
    lower_bound = (len(odd_vertices) + 1) // 2  # ceil(odd/2)
    
    # Total number of edges
    m = len(edges)
    
    # Edge set for quick lookup
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    # --- Strategy: Randomized greedy path packing ---
    # We try to greedily build long paths, then use local search to merge/adjust.
    
    def _greedy_decompose(num_paths_target, seed=0):
        """Try to decompose into exactly num_paths_target paths using greedy + repair."""
        rng = random.Random(seed)
        remaining_edges = set(edge_set)
        paths = []
        
        # Greedily build paths
        while remaining_edges:
            # Pick a random edge to start
            edge = rng.choice(list(remaining_edges))
            u, v = edge
            remaining_edges.discard(edge)
            
            # Extend the path from both ends
            path = [u, v]
            
            # Try to extend from the end (v)
            while True:
                current = path[-1]
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining_edges]
                if not neighbors:
                    break
                # Choose a neighbor that doesn't create a cycle (unless it's the start)
                chosen = None
                for nb in rng.sample(neighbors, len(neighbors)):
                    if nb == path[0] and len(path) > 2:
                        # This would form a cycle; only allow if we want to close
                        continue
                    chosen = nb
                    break
                if chosen is None:
                    # All neighbors would create cycles; pick the first valid one
                    for nb in neighbors:
                        if nb != path[0] or len(path) <= 2:
                            chosen = nb
                            break
                if chosen is None:
                    break
                remaining_edges.discard((min(current, chosen), max(current, chosen)))
                path.append(chosen)
            
            # Try to extend from the start (u)
            while True:
                current = path[0]
                neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) in remaining_edges]
                if not neighbors:
                    break
                chosen = None
                for nb in rng.sample(neighbors, len(neighbors)):
                    if nb == path[-1] and len(path) > 2:
                        continue
                    chosen = nb
                    break
                if chosen is None:
                    for nb in neighbors:
                        if nb != path[-1] or len(path) <= 2:
                            chosen = nb
                            break
                if chosen is None:
                    break
                remaining_edges.discard((min(current, chosen), max(current, chosen)))
                path.insert(0, chosen)
            
            paths.append(path)
        
        return paths
    
    def _is_valid_decomposition(paths, edge_set):
        """Check if paths are valid: simple, use edges, edge-disjoint, cover all edges."""
        used_edges = set()
        for path in paths:
            # Check simplicity
            if len(set(path)) != len(path):
                return False
            # Check edges exist and not reused
            for i in range(len(path) - 1):
                e = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        # Check all edges covered
        return used_edges == edge_set
    
    def _local_search(paths, num_paths_target, max_iters=5000):
        """Try to reduce number of paths to num_paths_target via merging/splitting."""
        paths = [list(p) for p in paths]
        
        for _ in range(max_iters):
            if len(paths) <= num_paths_target:
                break
            
            # Try to merge two paths
            merged = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    # Check if we can merge: end of p1 adjacent to end of p2, etc.
                    # Four possible connections:
                    # p1_end -> p2_start, p1_end -> p2_end, p1_start -> p2_start, p1_start -> p2_end
                    
                    connections = [
                        (p1[-1], p2[0]),
                        (p1[-1], p2[-1]),
                        (p1[0], p2[0]),
                        (p1[0], p2[-1]),
                    ]
                    
                    for a, b in connections:
                        e = (min(a, b), max(a, b))
                        if e not in edge_set:
                            continue
                        
                        # Determine the merged path
                        if a == p1[-1] and b == p2[0]:
                            new_path = p1 + p2[1:]
                        elif a == p1[-1] and b == p2[-1]:
                            new_path = p1 + list(reversed(p2))[1:]
                        elif a == p1[0] and b == p2[0]:
                            new_path = list(reversed(p1))[1:] + p2
                        else:  # a == p1[0] and b == p2[-1]
                            new_path = list(reversed(p1))[1:] + list(reversed(p2))
                        
                        # Check simplicity
                        if len(set(new_path)) != len(new_path):
                            continue
                        
                        # Check all edges in new_path are valid and were in p1 or p2
                        valid = True
                        for k in range(len(new_path) - 1):
                            ne = (min(new_path[k], new_path[k+1]), max(new_path[k], new_path[k+1]))
                            if ne not in edge_set:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Perform merge
                        paths[i] = new_path
                        paths.pop(j)
                        merged = True
                        break
                    if merged:
                        break
                if merged:
                    break
            
            if not merged:
                # Try splitting a long path to enable a different merge
                # Find a path longer than 2
                split_done = False
                for i in range(len(paths)):
                    if len(paths[i]) > 3:
                        p = paths[i]
                        # Try splitting at various positions
                        for split_pos in range(1, len(p) - 1):
                            p1 = p[:split_pos + 1]
                            p2 = p[split_pos:]
                            # Check both are valid paths (they are, since original was)
                            # But we need to check they use edges
                            valid1 = all((min(p1[k], p1[k+1]), max(p1[k], p1[k+1])) in edge_set for k in range(len(p1)-1))
                            valid2 = all((min(p2[k], p2[k+1]), max(p2[k], p2[k+1])) in edge_set for k in range(len(p2)-1))
                            if valid1 and valid2:
                                paths[i] = p1
                                paths.insert(i + 1, p2)
                                split_done = True
                                break
                        if split_done:
                            break
                
                if not split_done:
                    break
        
        return paths
    
    # Main algorithm
    best_paths = None
    best_count = m  # Worst case: each edge is its own path
    
    # First, get a baseline decomposition
    baseline = _greedy_decompose(lower_bound, seed=42)
    if _is_valid_decomposition(baseline, edge_set):
        best_paths = baseline
        best_count = len(baseline)
    
    # Try to improve with local search
    if best_paths and best_count > lower_bound:
        improved = _local_search(best_paths, lower_bound, max_iters=10000)
        if _is_valid_decomposition(improved, edge_set) and len(improved) < best_count:
            best_paths = improved
            best_count = len(improved)
    
    # If we didn't reach the lower bound, try more randomized restarts
    deadline = time_budget
    for seed in range(1, 5000):
        if time.time() - start_time > deadline:
            break
        candidate = _greedy_decompose(lower_bound, seed=seed)
        if _is_valid_decomposition(candidate, edge_set):
            if len(candidate) < best_count:
                best_paths = candidate
                best_count = len(candidate)
                if best_count <= lower_bound:
                    break
            # Try local search on this candidate
            if best_count > lower_bound:
                improved = _local_search(candidate, lower_bound, max_iters=2000)
                if _is_valid_decomposition(improved, edge_set) and len(improved) < best_count:
                    best_paths = improved
                    best_count = len(improved)
                    if best_count <= lower_bound:
                        break
    
    # Final fallback: if we still have issues, use a simple valid decomposition
    if best_paths is None or not _is_valid_decomposition(best_paths, edge_set):
        # Fallback: each edge as its own path
        best_paths = [[u, v] for u, v in edges]
    
    return best_paths


def construct():
    """
    Returns three lists of paths (vertex lists), one per fixed instance graph.
    """
    total_deadline = 140  # Leave 10s margin for 150s limit
    start_time = time.time()
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        elapsed = time.time() - start_time
        remaining = total_deadline - elapsed
        if remaining < 5:
            # Not enough time; use trivial decomposition
            edges = _instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(*spec)
        paths = _decompose_graph(edges, time_budget=remaining - 2)
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
