import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _precompute():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        base_adj[u].add(v)
        base_adj[v].add(u)

    # Base distance for path
    def base_dist(u, v):
        return abs(u - v)

    # Precompute for each pair (u, v) which base edges are "between" them
    # base_between[u][v] = set of edges (i, i+1) such that i in [min(u,v), max(u,v)-1]
    base_between = [[set() for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v:
                lo, hi = min(u, v), max(u, v)
                for i in range(lo, hi):
                    base_between[u][v].add((i, i + 1))

    return n, base_set, base_adj, base_dist, base_between


def _build_adj(n, added_set, base_adj):
    adj = [set(s) for s in base_adj]
    for u, v in added_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _bfs_dist(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= 4:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist


def _diameter_leq4(n, adj):
    for src in range(n):
        dist = _bfs_dist(n, adj, src)
        if -1 in dist:
            return False
    return True


def _is_triangle_free(n, added_set, base_adj):
    # Build full adjacency
    adj = _build_adj(n, added_set, base_adj)
    # Check all triangles
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w > v and w in adj[v]:
                        return False
    return True


def _edge_compatible(n, e, added_set, base_adj, base_set):
    """Check if adding edge e = (u,v) to added_set keeps triangle-freeness."""
    u, v = e
    # If e is already in base or added, it's a no-op but shouldn't happen
    if (min(u, v), max(u, v)) in base_set or (min(u, v), max(u, v)) in added_set:
        return False
    # A triangle is formed if there exists w such that (u,w) and (v,w) are both in (base ∪ added ∪ {e})
    # Since e is the new edge, we need (u,w) in (base ∪ added) AND (v,w) in (base ∪ added)
    # OR (u,w) in (base ∪ added ∪ {e}) but e is (u,v) so w != v
    # So: for all w != u, w != v:
    #   not ( (w in base_adj[u] or (min(u,w),max(u,w)) in added_set) and
    #         (w in base_adj[v] or (min(v,w),max(v,w)) in added_set) )
    for w in range(n):
        if w == u or w == v:
            continue
        uw_in_base = w in base_adj[u]
        vw_in_base = w in base_adj[v]
        if uw_in_base and vw_in_base:
            # Both base edges exist, adding (u,v) creates triangle
            return False
        if uw_in_base:
            # Check if (v,w) is in added
            vw_key = (min(v, w), max(v, w))
            if vw_key in added_set:
                return False
        if vw_in_base:
            uw_key = (min(u, w), max(u, w))
            if uw_key in added_set:
                return False
        # Both not in base: check if both in added
        if not uw_in_base and not vw_in_base:
            uw_key = (min(u, w), max(u, w))
            vw_key = (min(v, w), max(v, w))
            if uw_key in added_set and vw_key in added_set:
                return False
    return True


def _greedy_construct(n, base_adj, base_set, candidates, deadline):
    """Greedy: add edges one by one to minimize diameter, respecting triangle-freeness."""
    added_set = set()
    added_list = []
    
    # Sort candidates by "usefulness" - edges that bridge large gaps
    # For a path, an edge (u,v) with |u-v| > 1 is a shortcut.
    # Longer shortcuts are generally more useful for diameter reduction.
    def edge_value(e):
        u, v = e
        return abs(u - v)
    
    sorted_cands = sorted(candidates, key=edge_value, reverse=True)
    
    # We want to add edges that most improve diameter.
    # Start with no edges, diameter is 23.
    # Add edge, recompute diameter.
    # But we need a stopping condition: when diameter <= 4.
    
    # Heuristic: prioritize edges that connect distant nodes and don't create triangles.
    # We'll iterate through candidates and add if it helps.
    
    # Actually, a better approach: iteratively find the "most critical" pair of nodes
    # (those with current distance > 4) and add an edge that reduces their distance.
    
    # Let's try a different greedy: 
    # 1. Compute initial distances (path).
    # 2. While diameter > 4:
    #    a. Find all pairs (u,v) with dist > 4.
    #    b. For each such pair, find candidate edges that would reduce dist(u,v).
    #       An edge (a,b) reduces dist(u,v) if it creates a shorter path.
    #       This is complex. Simpler: just try adding candidate edges in order of 
    #       "potential" and check if diameter improves.
    
    # Let's use a simple heuristic: 
    # Process candidates in order of decreasing |u-v|.
    # For each candidate, if it's compatible (triangle-free) and adding it reduces 
    # the current diameter, add it.
    
    # To avoid expensive diameter checks for every candidate, we can be smarter.
    # But n=24 is small.
    
    current_diam = n - 1  # 23 for path
    for u, v in sorted_cands:
        if time.time() > deadline:
            break
        e = (min(u, v), max(u, v))
        if e in added_set or e in base_set:
            continue
        if not _edge_compatible(n, e, added_set, base_adj, base_set):
            continue
        # Tentatively add
        added_set.add(e)
        added_list.append(e)
        adj = _build_adj(n, added_set, base_adj)
        # Check diameter
        diam = 0
        valid = True
        for src in range(n):
            dist = _bfs_dist(n, adj, src)
            if -1 in dist:
                valid = False
                break
            m = max(dist)
            if m > diam:
                diam = m
        if valid and diam < current_diam:
            current_diam = diam
            if current_diam <= 4:
                break
        else:
            # Didn't help or invalid, remove
            added_set.remove(e)
            added_list.pop()
    
    return added_list, current_diam


def _greedy_remove(n, edges, base_adj, base_set, deadline):
    """Remove edges while maintaining diameter <= 4 and triangle-freeness."""
    edges = list(edges)
    added_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(edges)):
            e = edges[i]
            e_key = (min(e[0], e[1]), max(e[0], e[1]))
            # Try removing
            new_edges = edges[:i] + edges[i+1:]
            new_set = set((min(u, v), max(u, v)) for u, v in new_edges)
            
            # Triangle-freeness is preserved (subset)
            adj = _build_adj(n, new_set, base_adj)
            if _diameter_leq4(n, adj):
                edges = new_edges
                added_set = new_set
                changed = True
                break
    return edges


def _local_search(n, edges, base_adj, base_set, candidates, deadline):
    """Local search with swaps to find smaller solutions."""
    edges = list(edges)
    added_set = set((min(u, v), max(u, v)) for u, v in edges)
    best = len(edges)
    best_edges = list(edges)
    
    # Phase 1: Greedy removal
    edges = _greedy_remove(n, edges, base_adj, base_set, deadline)
    added_set = set((min(u, v), max(u, v)) for u, v in edges)
    if len(edges) < best:
        best = len(edges)
        best_edges = list(edges)
    
    # Phase 2: Swap-based search
    # Try swapping one edge with another
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                break
            e_out = edges[i]
            e_out_key = (min(e_out[0], e_out[1]), max(e_out[0], e_out[1]))
            base_edges_without = edges[:i] + edges[i+1:]
            
            for cand in candidates:
                cand_key = (min(cand[0], cand[1]), max(cand[0], cand[1]))
                if cand_key in added_set or cand_key in base_set:
                    continue
                if cand_key == e_out_key:
                    continue
                
                # Check compatibility of cand with base_edges_without
                # (triangle-freeness of base ∪ base_edges_without ∪ {cand})
                if not _edge_compatible(n, cand, added_set - {e_out_key}, base_adj, base_set):
                    continue
                
                new_edges = base_edges_without + [cand]
                new_set = set((min(u, v), max(u, v)) for u, v in new_edges)
                adj = _build_adj(n, new_set, base_adj)
                if _diameter_leq4(n, adj):
                    # Valid swap. Same size, but might enable further removal.
                    # Try to remove an edge from new_edges
                    for j in range(len(new_edges)):
                        if time.time() > deadline:
                            break
                        e_rem = new_edges[j]
                        e_rem_key = (min(e_rem[0], e_rem[1]), max(e_rem[0], e_rem[1]))
                        test_edges = new_edges[:j] + new_edges[j+1:]
                        test_set = set((min(u, v), max(u, v)) for u, v in test_edges)
                        adj2 = _build_adj(n, test_set, base_adj)
                        if _diameter_leq4(n, adj2):
                            # Found a smaller solution
                            edges = test_edges
                            added_set = test_set
                            best = len(edges)
                            best_edges = list(edges)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    return best_edges


def _random_greedy_init(n, base_adj, base_set, candidates, deadline):
    """Random greedy initialization."""
    added_set = set()
    added_list = []
    cands = candidates[:]
    random.shuffle(cands)
    
    current_diam = n - 1
    for u, v in cands:
        if time.time() > deadline:
            break
        e = (min(u, v), max(u, v))
        if e in added_set or e in base_set:
            continue
        if not _edge_compatible(n, e, added_set, base_adj, base_set):
            continue
        added_set.add(e)
        added_list.append(e)
        adj = _build_adj(n, added_set, base_adj)
        if _diameter_leq4(n, adj):
            return added_list
    return added_list


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n, base_set, base_adj, base_dist, base_between = _precompute()
    
    # Generate candidates
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_set:
                candidates.append((u, v))
    
    # Strategy 1: Structured initialization
    # Use a hub-like structure but optimized
    # Try multiple hubs
    
    best_edges = []
    best_count = float('inf')
    
    # Try hub strategies
    for hub in range(4, 20):  # Central-ish hubs
        if time.time() > deadline:
            break
        # Connect hub to nodes that are far from it
        # In a path, nodes far from hub are at distance > 4
        targets = []
        for v in range(n):
            if v == hub:
                continue
            if abs(v - hub) > 4:
                # Check if edge (hub, v) is compatible
                e = (min(hub, v), max(hub, v))
                if e not in base_set:
                    targets.append(v)
        
        # Sort targets by distance (farthest first)
        targets.sort(key=lambda v: abs(v - hub), reverse=True)
        
        added_set = set()
        added_list = []
        for v in targets:
            e = (min(hub, v), max(hub, v))
            if e in added_set or e in base_set:
                continue
            if _edge_compatible(n, e, added_set, base_adj, base_set):
                added_set.add(e)
                added_list.append(e)
        
        # Check validity
        if added_list:
            adj = _build_adj(n, added_set, base_adj)
            if _diameter_leq4(n, adj):
                # Greedy remove
                removed = _greedy_remove(n, added_list, base_adj, base_set, deadline)
                if len(removed) < best_count:
                    best_count = len(removed)
                    best_edges = removed
    
    # Strategy 2: Greedy construct
    if time.time() < deadline - 5:
        greedy_edges, diam = _greedy_construct(n, base_adj, base_set, candidates, deadline)
        if greedy_edges and diam <= 4:
            removed = _greedy_remove(n, greedy_edges, base_adj, base_set, deadline)
            if len(removed) < best_count:
                best_count = len(removed)
                best_edges = removed
    
    # Strategy 3: Random restarts
    if time.time() < deadline - 5:
        for _ in range(30):
            if time.time() > deadline:
                break
            init = _random_greedy_init(n, base_adj, base_set, candidates, deadline)
            if init:
                adj = _build_adj(n, set((min(u,v), max(u,v)) for u,v in init), base_adj)
                if _diameter_leq4(n, adj):
                    ls_res = _local_search(n, init, base_adj, base_set, candidates, deadline)
                    if ls_res and len(ls_res) < best_count:
                        best_count = len(ls_res)
                        best_edges = ls_res
    
    # Fallback
    if not best_edges:
        # Known valid solution: hub at 11
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        best_edges = [[hub, v] for v in targets]
    
    # Final validation
    added_set = set((min(u, v), max(u, v)) for u, v in best_edges)
    adj = _build_adj(n, added_set, base_adj)
    if not _is_triangle_free(n, added_set, base_adj) or not _diameter_leq4(n, adj):
        hub = 11
        targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
        best_edges = [[hub, v] for v in targets]
    
    return [[int(u), int(v)] for u, v in best_edges]
