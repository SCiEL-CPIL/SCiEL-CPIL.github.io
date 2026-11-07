"""Packing trees T1..T20 into K20."""
import time
from collections import defaultdict

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    trees = instance_trees(20, 611)
    tree_edges = []
    for k in range(1, 21):
        if k == 1:
            tree_edges.append([])
        elif k == 2:
            tree_edges.append([(0, 1)])
        else:
            tree_edges.append(trees[k-1])
    
    # Embed smallest trees first
    order = list(range(1, 21))
    
    used_edges = set()
    assignments = [None] * 21
    
    tree_adj = [defaultdict(list) for _ in range(21)]
    tree_bfs = [None] * 21
    tree_root = [0] * 21
    
    for k in range(1, 21):
        if k <= 2:
            tree_bfs[k] = list(range(k))
            tree_root[k] = 0
            continue
        adj = tree_adj[k]
        for (a, b) in tree_edges[k-1]:
            adj[a].append(b)
            adj[b].append(a)
        root = max(range(k), key=lambda v: len(adj[v]))
        visited = {root}
        bfs = [root]
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    bfs.append(v)
                    queue.append(v)
        tree_bfs[k] = bfs
        tree_root[k] = root
    
    def solve(pos):
        if time.time() > deadline:
            return False
        if pos == len(order):
            return True
        
        k = order[pos]
        k_idx = k - 1
        
        if k == 1:
            assignments[1] = []
            return solve(pos + 1)
        
        adj = tree_adj[k]
        bfs = tree_bfs[k]
        root = tree_root[k]
        
        def place_vertex(idx, mapping, used_hosts):
            if time.time() > deadline:
                return False
            if idx == len(bfs):
                for (a, b) in tree_edges[k_idx]:
                    ha, hb = mapping[a], mapping[b]
                    e = (min(ha, hb), max(ha, hb))
                    if e in used_edges:
                        return False
                for (a, b) in tree_edges[k_idx]:
                    ha, hb = mapping[a], mapping[b]
                    used_edges.add((min(ha, hb), max(ha, hb)))
                assignments[k] = [mapping[i] for i in range(k)]
                if solve(pos + 1):
                    return True
                for (a, b) in tree_edges[k_idx]:
                    ha, hb = mapping[a], mapping[b]
                    used_edges.discard((min(ha, hb), max(ha, hb)))
                assignments[k] = None
                return False
            
            v = bfs[idx]
            for hv in range(20):
                if hv in used_hosts:
                    continue
                ok = True
                for u in adj[v]:
                    if u in mapping:
                        hu = mapping[u]
                        e = (min(hv, hu), max(hv, hu))
                        if e in used_edges:
                            ok = False
                            break
                if not ok:
                    continue
                mapping[v] = hv
                used_hosts.add(hv)
                if place_vertex(idx + 1, mapping, used_hosts):
                    return True
                used_hosts.discard(hv)
                del mapping[v]
            return False
        
        if pos == 0:
            candidates = [0]
        else:
            candidates = list(range(20))
        
        for rh in candidates:
            mapping = {root: rh}
            used_hosts = {rh}
            if place_vertex(1, mapping, used_hosts):
                return True
        return False
    
    if not solve(0):
        return [None] * 19 + [list(range(20))]
    
    result = [None] * 20
    for k in range(1, 21):
        if k == 1:
            result[0] = None
        else:
            if assignments[k] is not None:
                result[k-1] = [int(v) for v in assignments[k]]
            else:
                result[k-1] = None
    return result
