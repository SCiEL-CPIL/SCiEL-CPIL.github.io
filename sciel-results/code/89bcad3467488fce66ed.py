"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(n, adj, start):
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

def _max_diameter(n, adj):
    max_d = 0
    for i in range(n):
        dists = _bfs_distances(n, adj, i)
        if -1 in dists:
            return float('inf')
        max_d = max(max_d, max(dists))
    return max_d

def _is_triangle_free(n, edges):
    adj = set()
    for u, v in edges:
        adj.add((min(u,v), max(u,v)))
    for u in range(n):
        for v in adj:
            pass
    # Check for triangles: for each edge (u,v), check if any common neighbor
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    for u in range(n):
        for i in range(len(adj_list[u])):
            for j in range(i+1, len(adj_list[u])):
                if adj_list[u][i] in adj_list[adj_list[u][j]]:
                    return False
    return True

def _check_valid(n, added_edges):
    all_edges = P['edges'] + added_edges
    # Check no duplicates
    edge_set = set()
    for u, v in all_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    # Check triangle-free
    if not _is_triangle_free(n, all_edges):
        return False
    # Check connected and diameter
    adj = _build_adj(n, all_edges)
    diam = _max_diameter(n, adj)
    return diam <= 4

def _candidate_edges(n):
    """Generate all valid candidate edges (even-odd pairs not in path)"""
    base = set()
    for u, v in P['edges']:
        base.add((min(u,v), max(u,v)))
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) in base:
                continue
            if (u % 2) == (v % 2):
                continue  # Skip same-parity to maintain bipartite/triangle-free
            candidates.append((u, v))
    return candidates

def _try_add_edges(n, added, candidates, current_edges, deadline):
    """Greedy: find minimum added edges"""
    if time.time() > deadline:
        return None
    if _check_valid(n, added):
        return list(added)
    # Find best edge to add
    best_gain = -1
    best_edge = None
    for u, v in candidates:
        if (u,v) in current_edges:
            continue
        test_added = list(added) + [(u,v)]
        test_edges = set(current_edges)
        test_edges.add((u,v))
        # Quick check: is this edge valid (triangle-free so far)?
        # We'll rely on final check, but do a quick triangle check
        # For speed, just try it
        if time.time() > deadline:
            break
        # Compute new diameter contribution - this is expensive
        # Instead, let's use a heuristic: prefer edges that connect distant parts
        # Actually, let's just do greedy by trying each and checking if it reduces diameter
        # But that's O(candidates * n^2) which might be too slow
        # For n=24, candidates ~ 144, n^2 = 576, so ~82k ops per step, should be fine
        all_test = P['edges'] + test_added
        adj = _build_adj(n, all_test)
        diam = _max_diameter(n, adj)
        if diam <= 4 and _is_triangle_free(n, all_test):
            # This alone might not suffice, but it's a good candidate
            # We want to minimize total edges, so let's be more strategic
            pass
    
    # Better approach: iterative deepening / BFS over edge sets
    # For small n, try all combinations of size k
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    n = P['n']
    base_edges = [(min(u,v), max(u,v)) for u, v in P['edges']]
    base_set = set(base_edges)
    
    # Generate candidate edges (even-odd, not in base)
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) in base_set:
                continue
            if (u % 2) != (v % 2):
                candidates.append((u, v))
    
    # Try increasing number of edges
    for k in range(1, 15):
        if time.time() > deadline:
            break
        # For small k, try combinations
        if k <= 5:
            for combo in itertools.combinations(range(len(candidates)), k):
                if time.time() > deadline:
                    break
                added = [candidates[i] for i in combo]
                all_edges = base_edges + added
                # Check triangle-free
                if not _is_triangle_free(n, all_edges):
                    continue
                # Check diameter
                adj = _build_adj(n, all_edges)
                diam = _max_diameter(n, adj)
                if diam <= 4:
                    return [[u, v] for u, v in added]
        else:
            # For larger k, use greedy heuristic
            if time.time() > deadline:
                break
            # Greedy: repeatedly add the edge that most reduces diameter
            current_added = []
            current_edges = list(base_edges)
            for _ in range(k):
                if time.time() > deadline:
                    break
                best_edge = None
                best_diam = float('inf')
                for u, v in candidates:
                    if (u,v) in current_edges:
                        continue
                    test_edges = current_edges + [(u,v)]
                    # Quick triangle check
                    if not _is_triangle_free(n, test_edges):
                        continue
                    adj = _build_adj(n, test_edges)
                    diam = _max_diameter(n, adj)
                    if diam < best_diam:
                        best_diam = diam
                        best_edge = (u,v)
                    if time.time() > deadline:
                        break
                if best_edge is None:
                    break
                current_added.append(best_edge)
                current_edges.append(best_edge)
                if best_diam <= 4:
                    # Verify full solution
                    all_edges = base_edges + current_added
                    if _is_triangle_free(n, all_edges):
                        adj = _build_adj(n, all_edges)
                        if _max_diameter(n, adj) <= 4:
                            return [[u, v] for u, v in current_added]
    
    # Fallback: return empty (should not happen)
    return []
# EVOLVE-BLOCK-END

