import time
from collections import defaultdict
import random

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _try_extend(path, adj, remaining, path_set, side):
    """
    Extend path from the given side ('front' or 'back').
    Returns (success, extended_path, updated_adj, updated_remaining)
    """
    if side == 'front':
        curr = path[0]
    else:
        curr = path[-1]
    
    # Find best next vertex: prefer one with highest remaining degree
    best_next = None
    best_deg = -1
    for w in sorted(adj[curr]):  # sort for determinism
        if w in path_set:
            continue
        deg = len(adj[w])
        if deg > best_deg:
            best_deg = deg
            best_next = w
    
    if best_next is None:
        return False, path, adj, remaining
    
    e = tuple(sorted((curr, best_next)))
    if e not in remaining:
        return False, path, adj, remaining
    
    remaining = remaining - {e}
    adj = {k: set(v) for k, v in adj.items()}
    adj[curr].discard(best_next)
    adj[best_next].discard(curr)
    
    if side == 'front':
        extended = [best_next] + path
    else:
        extended = path + [best_next]
    
    return True, extended, adj, remaining


def _greedy_decompose(edges):
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    paths = []
    
    while remaining:
        # Pick an edge to start with: prefer edge where both endpoints have high degree
        best_edge = None
        best_score = -1
        for u, v in sorted(remaining):
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        u, v = best_edge
        remaining.discard(best_edge)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Try extending from both ends multiple times
        extended = True
        while extended:
            extended = False
            # Try front
            ok, path, adj, remaining = _try_extend(path, adj, remaining, path_set, 'front')
            if ok:
                path_set = set(path)
                extended = True
            # Try back
            ok, path, adj, remaining = _try_extend(path, adj, remaining, path_set, 'back')
            if ok:
                path_set = set(path)
                extended = True
        
        paths.append(path)
    
    return paths


def _try_merge(paths, graph_edges, used_edges):
    """
    Try to merge two paths by finding a common endpoint edge.
    Returns (success, merged_paths, updated_used_edges)
    """
    n = len(paths)
    if n < 2:
        return False, paths, used_edges
    
    # For each pair of paths, check if we can merge them
    # Path i and path j can be merged if there's an edge between an endpoint of i and an endpoint of j
    # such that the merged path is still simple
    
    for i in range(n):
        for j in range(i + 1, n):
            pi = paths[i]
            pj = paths[j]
            set_i = set(pi)
            set_j = set(pj)
            
            # Check all four combinations of endpoints
            endpoints_i = [pi[0], pi[-1]]
            endpoints_j = [pj[0], pj[-1]]
            
            for ei in endpoints_i:
                for ej in endpoints_j:
                    if ei in set_j or ej in set_i:
                        continue
                    
                    e = tuple(sorted((ei, ej)))
                    if e not in graph_edges or e in used_edges:
                        continue
                    
                    # Determine how to merge
                    # We need to figure out which end of pi connects to which end of pj
                    # and which direction each path goes
                    
                    # Case 1: ei is pi[0], ej is pj[0]
                    if ei == pi[0] and ej == pj[0]:
                        # pi: ei ... pi[-1], pj: ej ... pj[-1]
                        # Merged: pi + reversed(pj) -- wait, no
                        # Actually: we connect ei-ej, so new path is [pi] + [pj[1:]] reversed? No.
                        # ei is start of pi, ej is start of pj.
                        # If we go from ej to ei, we can't because ej is start.
                        # We can go: pj[-1], ..., ej, ei, ..., pi[-1]
                        # So: reversed(pj) + pi, but skip duplicate
                        merged = list(reversed(pj)) + pi
                        # Check simplicity
                        if len(set(merged)) == len(merged):
                            new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                            new_used = used_edges | {e}
                            return True, new_paths, new_used
                    
                    # Case 2: ei is pi[0], ej is pj[-1]
                    if ei == pi[0] and ej == pj[-1]:
                        # pj: pj[0] ... ej, pi: ei ... pi[-1]
                        # Merged: pj + pi, skip duplicate
                        merged = pj + pi
                        if len(set(merged)) == len(merged):
                            new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                            new_used = used_edges | {e}
                            return True, new_paths, new_used
                    
                    # Case 3: ei is pi[-1], ej is pj[0]
                    if ei == pi[-1] and ej == pj[0]:
                        # pi: pi[0] ... ei, pj: ej ... pj[-1]
                        # Merged: pi + pj, skip duplicate
                        merged = pi + pj
                        if len(set(merged)) == len(merged):
                            new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                            new_used = used_edges | {e}
                            return True, new_paths, new_used
                    
                    # Case 4: ei is pi[-1], ej is pj[-1]
                    if ei == pi[-1] and ej == pj[-1]:
                        # pi: pi[0] ... ei, pj: pj[0] ... ej
                        # Merged: pi + reversed(pj), skip duplicate
                        merged = pi + list(reversed(pj))
                        if len(set(merged)) == len(merged):
                            new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                            new_used = used_edges | {e}
                            return True, new_paths, new_used
    
    return False, paths, used_edges


def decompose(edges):
    if not edges:
        return []
    
    graph_edges = set(edges)
    paths = _greedy_decompose(edges)
    
    # Compute used edges from paths
    used_edges = set()
    for p in paths:
        for k in range(len(p) - 1):
            used_edges.add(tuple(sorted((p[k], p[k+1]))))
    
    # Iteratively merge paths
    changed = True
    while changed:
        changed, paths, used_edges = _try_merge(paths, graph_edges, used_edges)
        # Recompute used edges after merge
        if changed:
            used_edges = set()
            for p in paths:
                for k in range(len(p) - 1):
                    used_edges.add(tuple(sorted((p[k], p[k+1]))))
    
    return paths


def construct():
    start = time.time()
    deadline = start + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            break
    
    while len(results) < 3:
        results.append([])
    
    return results
