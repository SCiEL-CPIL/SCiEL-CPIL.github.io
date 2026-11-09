"""Packing trees T1..T20 into K20 using greedy + backtracking with constraint propagation."""
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    from random import Random
    rng = Random(seed)
    trees = []
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 115.0
    n = 20
    trees = instance_trees(n, 611)
    
    used_edges = set()
    result = [None] * n
    
    # Precompute tree adjacency and degree info
    tree_adj = []
    tree_deg = []
    for edges in trees:
        k = len(edges) + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
    
    # Order: smaller trees first (easier to place, constrains remaining graph)
    order = sorted(range(n), key=lambda i: (len(trees[i]), i))
    
    # Track which host vertices are "free" for each tree
    # For efficiency, we'll track used_edges as a set of frozensets or tuples
    
    for k_idx in order:
        if time.time() > deadline:
            break
        
        k = k_idx + 1
        if k == 1:
            # Single vertex tree: just pick any unused vertex
            for v in range(n):
                result[k_idx] = [v]
                break
            continue
        
        adj = tree_adj[k_idx]
        deg = tree_deg[k_idx]
        
        # Vertices ordered by degree descending (high-degree vertices are more constrained)
        verts = sorted(range(k), key=lambda v: (-deg[v], v))
        
        success = False
        # Try multiple random attempts with different orderings
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            host = [-1] * k
            used_hosts = set()
            
            # Shuffle vertices for diversity in later attempts
            if attempt > 0:
                verts_shuffled = verts[:]
                random.shuffle(verts_shuffled)
                # But keep high-degree vertices first with some randomness
                verts_try = sorted(verts_shuffled, key=lambda v: (-deg[v], random.random()))
            else:
                verts_try = verts
            
            def can_assign(v, h):
                """Check if host h can be assigned to tree vertex v."""
                if h in used_hosts:
                    return False
                for u in adj[v]:
                    if host[u] != -1:
                        e = (min(host[u], h), max(host[u], h))
                        if e in used_edges:
                            return False
                return True
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(verts_try):
                    return True
                
                v = verts_try[idx]
                
                # Get candidates sorted by how constrained they are
                cands = []
                for h in range(n):
                    if can_assign(v, h):
                        # Score: prefer hosts with fewer available neighbors for unassigned neighbors
                        score = 0
                        for u in adj[v]:
                            if host[u] == -1:
                                # Count how many hosts are available for u
                                avail = 0
                                for hu in range(n):
                                    if hu not in used_hosts and hu != h:
                                        e = (min(hu, h), max(hu, h))
                                        if e not in used_edges:
                                            avail += 1
                                score += avail
                        cands.append((score, h))
                
                if not cands:
                    return False
                
                # Sort by score (prefer more constrained placements)
                cands.sort(key=lambda x: (x[0], x[1]))
                
                # For diversity, sometimes shuffle the top candidates
                if attempt > 10 and random.random() < 0.3:
                    top = cands[:min(10, len(cands))]
                    random.shuffle(top)
                    cands = top + cands[min(10, len(cands)):]
                
                for _, h in cands:
                    # Assign
                    host[v] = h
                    used_hosts.add(h)
                    # Add edges to used_edges temporarily
                    added = []
                    for u in adj[v]:
                        if host[u] != -1:
                            e = (min(host[u], h), max(host[u], h))
                            used_edges.add(e)
                            added.append(e)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    for e in added:
                        used_edges.discard(e)
                    host[v] = -1
                    used_hosts.discard(h)
                
                return False
            
            if backtrack(0):
                # Success: add remaining edges to used_edges
                for u, v in trees[k_idx]:
                    a, b = host[u], host[v]
                    used_edges.add((min(a, b), max(a, b)))
                result[k_idx] = [int(x) for x in host]
                success = True
                break
        
        if not success:
            result[k_idx] = None
    
    return result

if __name__ == "__main__":
    res = construct()
    trees = instance_trees(20, 611)
    used = set()
    ok = True
    total = 0
    for k in range(1, 21):
        r = res[k - 1]
        if r is None:
            continue
        if len(r) != k or len(set(r)) != k:
            ok = False
        for v in r:
            if not isinstance(v, int) or v < 0 or v > 19:
                ok = False
        for (u, v) in trees[k - 1]:
            key = (min(r[u], r[v]), max(r[u], r[v]))
            if key in used:
                ok = False
            used.add(key)
        total += len(trees[k - 1])
    print(f"Valid: {ok}, Total edges: {total} / 190")
