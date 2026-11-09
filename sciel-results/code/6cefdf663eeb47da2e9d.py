"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

import random

def _build_adj(added_edges):
    """Build adjacency list from base path + added edges."""
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_all(adj, start):
    """BFS from start, return distance array."""
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

def _diameter(adj):
    """Compute diameter; return -1 if disconnected."""
    n = len(adj)
    max_d = 0
    for s in range(n):
        dist = _bfs_all(adj, s)
        if -1 in dist:
            return -1
        max_d = max(max_d, max(dist))
    return max_d

def _triangle_free(adj):
    """Check triangle-free using adjacency sets."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i in range(len(adj[u])):
            v = adj[u][i]
            for j in range(i + 1, len(adj[u])):
                w = adj[u][j]
                if w in adj_set[v]:
                    return False
    return True

def _valid(added_edges, path_edge_set):
    """Full validity check."""
    edge_set = set(path_edge_set)
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    adj = _build_adj(added_edges)
    if not _triangle_free(adj):
        return False
    d = _diameter(adj)
    if d < 0 or d > 4:
        return False
    return True

def _quick_check(added_edges, path_edge_set, adj_cache=None):
    """Fast check: duplicates + triangle-free only (no diameter)."""
    edge_set = set(path_edge_set)
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    adj = _build_adj(added_edges)
    if not _triangle_free(adj):
        return False
    return True

def _greedy_removal(added_edges, path_edge_set):
    """Try to remove edges one by one, keep if still valid."""
    current = list(added_edges)
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            if _valid(candidate, path_edge_set):
                current = candidate
                changed = True
                break
    return current

def _swap_polish(added_edges, path_edge_set, all_non_path_edges, deadline):
    """1-for-1 swap: try removing one edge and adding another to reduce count or improve."""
    current = list(added_edges)
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            without_i = current[:i] + current[i+1:]
            # Try adding each non-path edge not already in without_i
            existing_set = set((min(u,v), max(u,v)) for u,v in without_i)
            for e in all_non_path_edges:
                if (e[0], e[1]) in existing_set or (e[1], e[0]) in existing_set:
                    continue
                candidate = without_i + [list(e)]
                if _valid(candidate, path_edge_set):
                    if len(candidate) < len(current):
                        current = candidate
                        improved = True
                        break
            if improved:
                break
        if time.time() > deadline:
            break
    return current

def _random_greedy_build(path_edge_set, all_non_path_edges, rng):
    """Build a solution by greedily adding edges in random order until diameter <= 4."""
    # Start with empty, check diameter of base path (it's 23, way too big)
    # We need to add edges to bring diameter down to 4.
    # Greedy: repeatedly pick the edge that most reduces the current diameter.
    
    # First, find a feasible set by adding edges that help most
    # Strategy: pick edges that connect distant vertices
    
    added = []
    added_set = set()
    
    # Precompute: which edges are "useful" - connect vertices far apart on path
    # Sort candidate edges by how much they reduce max distance
    candidates = list(all_non_path_edges)
    rng.shuffle(candidates)
    
    # Greedy: add edges while diameter > 4
    adj = _build_adj(added)
    diam = _diameter(adj)
    
    while diam > 4 and candidates:
        best_edge = None
        best_diam = diam
        # Sample a subset for speed
        sample_size = min(50, len(candidates))
        sample = candidates[:sample_size]
        for e in sample:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key in added_set:
                continue
            new_added = added + [e]
            if not _quick_check(new_added, path_edge_set):
                continue
            new_adj = _build_adj(new_added)
            new_diam = _diameter(new_adj)
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = e
        if best_edge is None:
            # Try all remaining
            for e in candidates:
                key = (min(e[0], e[1]), max(e[0], e[1]))
                if key in added_set:
                    continue
                new_added = added + [e]
                if not _quick_check(new_added, path_edge_set):
                    continue
                new_adj = _build_adj(new_added)
                new_diam = _diameter(new_adj)
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = e
        if best_edge is None:
            break
        added.append(best_edge)
        added_set.add((min(best_edge[0], best_edge[1]), max(best_edge[0], best_edge[1])))
        # Remove from candidates
        candidates = [e for e in candidates if (min(e[0],e[1]), max(e[0],e[1])) != (min(best_edge[0],best_edge[1]), max(best_edge[0],best_edge[1]))]
        adj = _build_adj(added)
        diam = _diameter(adj)
    
    return added

def _initial_solution():
    """Deterministic hub-based initial solution (10 edges)."""
    return [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    path_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # All non-path edges
    all_non_path_edges = []
    for u in range(24):
        for v in range(u + 1, 24):
            if (u, v) not in path_edge_set:
                all_non_path_edges.append([u, v])
    
    # Start with known 10-edge solution
    best = _initial_solution()
    if not _valid(best, path_edge_set):
        return []
    best_len = len(best)
    
    # Phase 1: Greedy removal on initial solution
    best = _greedy_removal(best, path_edge_set)
    best_len = len(best)
    
    # Phase 2: Swap polish
    best = _swap_polish(best, path_edge_set, all_non_path_edges, deadline)
    best_len = len(best)
    
    # Phase 3: Randomized multi-start greedy with removal + swap
    rng = random.Random(42)
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random greedy build
        candidate = _random_greedy_build(path_edge_set, all_non_path_edges, rng)
        
        if candidate:
            # Verify full validity
            if _valid(candidate, path_edge_set):
                # Greedy removal
                candidate = _greedy_removal(candidate, path_edge_set)
                if _valid(candidate, path_edge_set) and len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
                
                # Swap polish
                candidate = _swap_polish(candidate, path_edge_set, all_non_path_edges, deadline)
                if _valid(candidate, path_edge_set) and len(candidate) < best_len:
                    best = candidate
                    best_len = len(candidate)
        
        # Also try: take current best, remove a random edge, then greedily rebuild
        if best_len > 7:
            if time.time() > deadline:
                break
            # Perturbation: remove 1-2 random edges from best
            perturbed = list(best)
            remove_count = min(2, len(perturbed))
            indices_to_remove = rng.sample(range(len(perturbed)), remove_count)
            indices_to_remove.sort(reverse=True)
            for idx in indices_to_remove:
                perturbed.pop(idx)
            
            if _valid(perturbed, path_edge_set):
                perturbed = _greedy_removal(perturbed, path_edge_set)
                if _valid(perturbed, path_edge_set) and len(perturbed) < best_len:
                    best = perturbed
                    best_len = len(perturbed)
            
            # Try to fill back in with random greedy if perturbed is invalid or larger
            if not _valid(perturbed, path_edge_set) or len(perturbed) >= best_len:
                # Greedy add to fix
                if _valid(perturbed, path_edge_set):
                    pass  # already valid, just larger
                else:
                    # Need to add edges back
                    fixed = list(perturbed)
                    fixed_set = set((min(u,v), max(u,v)) for u,v in fixed)
                    adj = _build_adj(fixed)
                    diam = _diameter(adj)
                    if diam == -1 or diam > 4:
                        # Add edges to fix
                        for e in all_non_path_edges:
                            if time.time() > deadline:
                                break
                            key = (min(e[0], e[1]), max(e[0], e[1]))
                            if key in fixed_set:
                                continue
                            new_fixed = fixed + [e]
                            if _quick_check(new_fixed, path_edge_set):
                                new_adj = _build_adj(new_fixed)
                                new_diam = _diameter(new_adj)
                                if 0 < new_diam <= 4:
                                    fixed = new_fixed
                                    fixed_set.add(key)
                                    break
                        if _valid(fixed, path_edge_set):
                            fixed = _greedy_removal(fixed, path_edge_set)
                            if _valid(fixed, path_edge_set) and len(fixed) < best_len:
                                best = fixed
                                best_len = len(fixed)
    
    # Final polish on best
    best = _greedy_removal(best, path_edge_set)
    if not _valid(best, path_edge_set):
        # Shouldn't happen, but fallback
        fallback = _initial_solution()
        if _valid(fallback, path_edge_set):
            best = fallback
        else:
            best = []
    
    # Ensure return type is correct
    result = []
    for e in best:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
