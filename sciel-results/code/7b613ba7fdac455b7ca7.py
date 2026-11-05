import time
from collections import defaultdict

def instance(n, p, seed):
    import random
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _extend_path(path, path_set, adj, remaining):
    """Extend path from both ends as much as possible."""
    extended = True
    while extended:
        extended = False
        for side in ('front', 'back'):
            if side == 'front':
                curr = path[0]
            else:
                curr = path[-1]
            
            best_next = None
            best_deg = -1
            for w in sorted(adj[curr]):
                if w in path_set:
                    continue
                e = tuple(sorted((curr, w)))
                if e not in remaining:
                    continue
                deg = len(adj[w])
                if deg > best_deg:
                    best_deg = deg
                    best_next = w
            
            if best_next is None:
                continue
            
            e = tuple(sorted((curr, best_next)))
            remaining.discard(e)
            adj[curr].discard(best_next)
            adj[best_next].discard(curr)
            
            if side == 'front':
                path.insert(0, best_next)
            else:
                path.append(best_next)
            path_set.add(best_next)
            extended = True


def _greedy_decompose_from_seeds(edges, seed_vertices):
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    paths = []
    used_seeds = set()
    
    for sv in seed_vertices:
        if not remaining or sv in used_seeds:
            continue
        if not adj[sv]:
            continue
        best_e = None
        best_score = -1
        for w in sorted(adj[sv]):
            e = tuple(sorted((sv, w)))
            if e not in remaining:
                continue
            score = len(adj[w])
            if score > best_score:
                best_score = score
                best_e = (sv, w)
        if best_e is None:
            continue
        u, v = best_e
        remaining.discard(tuple(sorted((u, v))))
        adj[u].discard(v)
        adj[v].discard(u)
        path = [u, v]
        path_set = {u, v}
        used_seeds.add(sv)
        _extend_path(path, path_set, adj, remaining)
        paths.append(path)
    
    while remaining:
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
        _extend_path(path, path_set, adj, remaining)
        paths.append(path)
    
    return paths


def _merge_paths(paths, edges_set):
    used = set()
    for p in paths:
        for k in range(len(p) - 1):
            used.add(tuple(sorted((p[k], p[k+1]))))
    
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = paths[i], paths[j]
            set_i, set_j = set(pi), set(pj)
            for ei in [pi[0], pi[-1]]:
                for ej in [pj[0], pj[-1]]:
                    if ei in set_j or ej in set_i:
                        continue
                    e = tuple(sorted((ei, ej)))
                    if e not in edges_set or e in used:
                        continue
                    merged = None
                    if ei == pi[0] and ej == pj[0]:
                        merged = list(reversed(pj)) + pi
                    elif ei == pi[0] and ej == pj[-1]:
                        merged = pj + pi
                    elif ei == pi[-1] and ej == pj[0]:
                        merged = pi + pj
                    else:
                        merged = pi + list(reversed(pj))
                    if merged and len(set(merged)) == len(merged):
                        new_paths = paths[:i] + paths[i+1:j] + [merged] + paths[j+1:]
                        return new_paths
    return paths


def decompose(edges):
    if not edges:
        return []
    
    edges_set = set(edges)
    adj = _adjacency(edges)
    odd = [v for v in adj if len(adj[v]) % 2 == 1]
    
    paths = _greedy_decompose_from_seeds(edges, odd)
    
    changed = True
    while changed:
        new_paths = _merge_paths(paths, edges_set)
        if new_paths != paths:
            paths = new_paths
            changed = True
        else:
            changed = False
    
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
