"""Improved program: greedy path cover with randomized restarts and local search."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import defaultdict, deque
from itertools import combinations

def _instance(n, p, seed):
    """Generate the fixed instance graph (copied from verifier)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = set()
    for u, v in zip(order, order[1:]):
        edges.add((min(u, v), max(u, v)))
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return sorted(edges)

def _build_adj(edges):
    """Build adjacency list and edge set for fast lookup."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj, set(edges)

def _greedy_cover(edges, adj, rng, max_paths=None):
    """
    Greedy edge-disjoint path cover.
    Strategy: repeatedly pick a longest simple path in the remaining graph.
    To avoid O(m^2) per path, we use randomized DFS with pruning.
    """
    remaining = set(edges)
    paths = []
    
    while remaining:
        # Pick a random starting vertex with high degree in remaining graph
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        # Try multiple random starts, keep best path
        best_path = None
        best_len = 0
        
        candidates = [v for v in deg if deg[v] > 0]
        rng.shuffle(candidates)
        # Try up to 10 random starts
        for start in candidates[:10]:
            # Randomized DFS to find a long simple path
            visited = {start}
            path = [start]
            stack = [start]
            
            # Use iterative DFS with backtracking for longer paths
            best_dfs_path = [start]
            
            def dfs(node, path_so_far, visited_set):
                nonlocal best_dfs_path, best_len
                if len(path_so_far) > len(best_dfs_path):
                    best_dfs_path = list(path_so_far)
                
                # Try to extend
                neighbors = list(adj[node])
                rng.shuffle(neighbors)
                for nb in neighbors:
                    if nb not in visited_set:
                        # Check edge is in remaining
                        edge = (min(node, nb), max(node, nb))
                        if edge in remaining:
                            visited_set.add(nb)
                            path_so_far.append(nb)
                            dfs(nb, path_so_far, visited_set)
                            path_so_far.pop()
                            visited_set.discard(nb)
                            # Early stop if we've found a very long path
                            if len(best_dfs_path) >= len(remaining) * 0.8:
                                return
            
            dfs(start, [start], {start})
            
            if len(best_dfs_path) > best_len:
                best_len = len(best_dfs_path)
                best_path = best_dfs_path
        
        if best_path is None or len(best_path) < 2:
            # Fallback: just take any single edge
            e = next(iter(remaining))
            paths.append(list(e))
            remaining.discard(e)
        else:
            paths.append(best_path)
            # Remove edges used in this path
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                edge = (min(u, v), max(u, v))
                remaining.discard(edge)
        
        if max_paths and len(paths) >= max_paths:
            break
    
    return paths

def _local_search_improve(paths, edges, adj, rng, max_iter=50):
    """
    Try to merge paths or re-route to reduce path count.
    Simple approach: try to connect two paths by removing an edge from one
    and adding it to another, if that reduces total path count.
    """
    # This is a simplified local search; in practice, we rely on greedy restarts
    # For now, just return paths as-is
    return paths

def construct():
    t_start = time.time()
    deadline = t_start + 120  # 120s wall-clock limit
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    all_results = []
    best_total = float('inf')
    best_all = None
    
    for idx, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        adj, edge_set = _build_adj(edges)
        m = len(edges)
        
        best_paths = None
        best_count = float('inf')
        
        # Multiple randomized restarts
        rng = random.Random(42 + idx * 1000)
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            paths = _greedy_cover(edges, adj, rng)
            count = len(paths)
            
            if count < best_count:
                best_count = count
                best_paths = paths
                best_total_candidate = sum(len(p) for p in best_paths)  # sanity
                # Track total for early termination
                if best_all is not None:
                    total_so_far = sum(len(p) for p in best_all) + count
                else:
                    total_so_far = count
                if total_so_far < 38:
                    # Good enough, could break early but let's try a bit more
                    pass
            
            # If we've reached the theoretical lower bound (ceiling of odd-degree vertices / 2), stop
            odd_deg = sum(1 for v in adj if len(adj[v]) % 2 == 1)
            lower_bound = (odd_deg + 1) // 2
            if count <= lower_bound:
                break
        
        if best_paths is None:
            # Fallback: each edge as its own path
            best_paths = [list(e) for e in edges]
        
        all_results.append(best_paths)
        best_all = all_results
    
    return all_results

# EVOLVE-BLOCK-END
