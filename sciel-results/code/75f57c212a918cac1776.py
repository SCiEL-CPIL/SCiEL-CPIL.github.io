"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _initial_paths(edges):
    """Start with each edge as its own path."""
    return [list(e) for e in edges]

def _try_merge(path_a, path_b, adj):
    """
    Try to merge path_a and path_b into a single simple path.
    A merge is possible if there is an edge connecting an endpoint of path_a
    to an endpoint of path_b.
    Returns the merged path (list of vertices) or None.
    """
    # Endpoints of path_a
    a_start, a_end = path_a[0], path_a[-1]
    # Endpoints of path_b
    b_start, b_end = path_b[0], path_b[-1]
    
    # Try all four combinations of connecting endpoints
    # Connection: a endpoint to b endpoint, then concatenate
    # We need to check if the connecting edge exists in adj
    
    # Option 1: a_end -- b_start  => path_a + path_b
    if b_start in adj.get(a_end, set()):
        # Check no vertex overlap
        set_a = set(path_a)
        set_b = set(path_b)
        if not set_a & set_b:
            return path_a + path_b
    
    # Option 2: a_end -- b_end  => path_a + reversed(path_b)
    if b_end in adj.get(a_end, set()):
        set_a = set(path_a)
        set_b = set(path_b)
        if not set_a & set_b:
            return path_a + path_b[::-1]
    
    # Option 3: a_start -- b_start  => reversed(path_a) + path_b
    if b_start in adj.get(a_start, set()):
        set_a = set(path_a)
        set_b = set(path_b)
        if not set_a & set_b:
            return path_a[::-1] + path_b
    
    # Option 4: a_start -- b_end  => reversed(path_a) + reversed(path_b)
    if b_end in adj.get(a_start, set()):
        set_a = set(path_a)
        set_b = set(path_b)
        if not set_a & set_b:
            return path_a[::-1] + path_b[::-1]
    
    return None

def _greedy_merge(paths, adj, deadline):
    """
    Greedily merge paths until no more merges are possible or deadline is reached.
    Uses a priority-like approach: try to merge paths, preferring shorter paths
    to keep options open, but primarily just iterate until no progress.
    """
    paths = list(paths)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(paths)
        # Sort by length to try shorter paths first (more likely to merge)
        # Actually, let's just iterate and try all pairs
        # To be more efficient, we can track which paths are "active"
        for i in range(len(paths)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(paths)):
                merged = _try_merge(paths[i], paths[j], adj)
                if merged is not None:
                    # Replace paths[i] and paths[j] with merged
                    # Remove j first if j > i, then i
                    del paths[j]
                    paths[i] = merged
                    changed = True
                    # Restart the outer loop since indices changed
                    break
            if changed:
                break
    return paths

def _local_search(paths, adj, deadline, max_iters=500):
    """
    Local search: try to reduce path count by finding a path that can be
    split and its parts merged with other paths. This is a 2-opt style move.
    
    Alternative: try to find a cycle in the union of two paths that allows
    a different decomposition. This is complex.
    
    Simpler local search: for each path, try removing an internal edge and
    seeing if the two resulting sub-paths can be merged with other paths
    to reduce total count.
    
    Actually, a better approach: try to find two paths that share no vertices
    but have endpoints that can be connected via a path of existing edges
    not in any current path... but all edges are in paths.
    
    Let's use a different local search: for each pair of paths, try to find
    a "swap" that reduces the number of paths. This is hard.
    
    Instead, let's do: for each path P, try to split it at an internal vertex
    into two sub-paths P1, P2. Then try to merge P1 and P2 with other paths
    to see if we can reduce the total count. This is expensive.
    
    Given time constraints, let's stick with greedy merge and add a simple
    1-opt: try to reorder vertices within a path? No, paths are fixed by edges.
    
    Let's implement a simple local search: for each path, try removing one
    internal edge, creating two sub-paths. Then run greedy merge on the new
    set. If the result is better, accept.
    """
    best_paths = list(paths)
    best_count = len(best_paths)
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        improved = False
        for pi in range(len(best_paths)):
            if time.time() >= deadline:
                break
            p = best_paths[pi]
            if len(p) < 3:
                continue
            # Try splitting at each internal position
            for split_pos in range(1, len(p) - 1):
                p1 = p[:split_pos + 1]
                p2 = p[split_pos:]
                # Create new path list
                new_paths = list(best_paths)
                del new_paths[pi]
                new_paths.append(p1)
                new_paths.append(p2)
                # Try to merge greedily
                merged = _greedy_merge(new_paths, adj, deadline)
                if len(merged) < best_count:
                    best_paths = merged
                    best_count = len(merged)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_paths

def _solve_one(edges, deadline):
    """Solve path decomposition for one graph."""
    adj = _build_graph(edges)
    paths = _initial_paths(edges)
    # Greedy merge
    paths = _greedy_merge(paths, adj, deadline)
    # Local search
    paths = _local_search(paths, adj, deadline)
    return paths

def construct():
    start_time = time.time()
    total_deadline = start_time + 140  # Leave 10s margin
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() >= total_deadline:
            # Fallback: return edge-per-path for remaining
            edges = _instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(n, p, seed)
        # Allocate remaining time proportionally
        remaining = total_deadline - time.time()
        graphs_left = len(specs) - len(results)
        per_graph = remaining / graphs_left
        graph_deadline = time.time() + per_graph
        
        paths = _solve_one(edges, graph_deadline)
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
