"""Rewritten program with improved tree packing algorithm."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    
    # Precompute tree properties
    tree_info = []
    for idx in range(n):
        k = idx + 1
        edges = trees[idx]
        if k == 1:
            tree_info.append((k, 0, edges, []))
        else:
            adj = [[] for _ in range(k)]
            deg = [0] * k
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
            tree_info.append((k, deg, edges, adj))
    
    # Total edges available
    total_possible = sum(k - 1 for k in range(1, n + 1))  # 190
    
    best_result = None
    best_score = -1
    
    # Strategy: embed trees from largest to smallest
    # Use multiple restarts with different random seeds
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(12345 + restart * 7919)
        
        used_edges = set()
        result = [None] * n
        
        # Embed in order: largest first, but with some randomization for variety
        order = list(range(n - 1, -1, -1))
        
        # Occasionally shuffle the middle trees for diversity
        if restart % 3 == 0 and len(order) > 4:
            # Keep largest and smallest fixed, shuffle middle
            mid = order[1:-1]
            rng.shuffle(mid)
            order = [order[0]] + mid + [order[-1]]
        
        success = True
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            
            k, deg, edges, adj = tree_info[idx]
            
            if k == 1:
                # Pick any unused vertex
                candidates = [v for v in range(n)]
                if not candidates:
                    success = False
                    break
                result[idx] = [rng.choice(candidates)]
                continue
            
            if k == 2:
                # Find an unused edge
                candidates = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            candidates.append((u, v))
                if not candidates:
                    success = False
                    break
                u, v = rng.choice(candidates)
                result[idx] = [u, v]
                used_edges.add((u, v))
                continue
            
            # For k >= 3, use improved backtracking
            embedding = _embed_tree_improved(adj, deg, k, used_edges, n, deadline, rng)
            if embedding is None:
                success = False
                break
            result[idx] = embedding
            for (a, b) in edges:
                u, v = embedding[a], embedding[b]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        
        if success:
            total_edges = sum(len(trees[idx]) for idx in range(n))
            if total_edges > best_score:
                best_score = total_edges
                best_result = [list(r) if r is not None else None for r in result]
                if best_score == total_possible:
                    break
    
    if best_result is None:
        best_result = [None] * n
    
    return best_result

def _embed_tree_improved(adj, deg, k, used_edges, n, deadline, rng):
    """
    Improved tree embedding using degree-aware backtracking.
    """
    # Find a leaf to start from (prefer low-degree leaves)
    start_tv = 0
    min_deg = k
    for v in range(k):
        if deg[v] == 1:
            start_tv = v
            break
    
    # BFS order starting from start_tv
    order = [start_tv]
    visited = {start_tv}
    queue = deque([start_tv])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                order.append(v)
                queue.append(v)
    
    # Precompute neighbor indices for each position in order
    neighbor_indices = [[] for _ in range(k)]
    for i in range(k):
        tv = order[i]
        for j in range(i):
            if order[j] in adj[tv]:
                neighbor_indices[i].append(j)
    
    mapping = [-1] * k
    used_host = [False] * n
    
    # Track remaining degree capacity for each host vertex
    # Initially each host can be incident to at most (n-1) edges total
    # But we need to track which edges are already used
    host_remaining = [n - 1] * n  # max degree in K20
    
    # Count used edges incident to each host
    host_used_deg = [0] * n
    for (u, v) in used_edges:
        host_used_deg[u] += 1
        host_used_deg[v] += 1
    host_remaining = [n - 1 - host_used_deg[i] for i in range(n)]
    
    # Sort host vertices by remaining capacity (ascending) for MRV-like heuristic
    host_order = list(range(n))
    host_order.sort(key=lambda v: host_remaining[v])
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == k:
            return True
        
        tv = order[idx]
        needed_deg = deg[tv]
        
        # Collect valid host candidates
        valid_hosts = []
        for hv in host_order:
            if used_host[hv]:
                continue
            # Check if host has enough remaining degree
            if host_remaining[hv] < needed_deg:
                continue
            # Check edges to already-mapped neighbors
            valid = True
            for j in neighbor_indices[idx]:
                nhv = mapping[j]
                u, v = hv, nhv
                if u > v:
                    u, v = v, u
                if (u, v) in used_edges:
                    valid = False
                    break
            if valid:
                valid_hosts.append(hv)
        
        if not valid_hosts:
            return False
        
        # Shuffle for randomization
        rng.shuffle(valid_hosts)
        
        # Try each valid host
        for hv in valid_hosts:
            mapping[idx] = hv
            used_host[hv] = True
            host_remaining[hv] -= needed_deg
            
            if backtrack(idx + 1):
                return True
            
            host_remaining[hv] += needed_deg
            used_host[hv] = False
            mapping[idx] = -1
        
        return False
    
    if backtrack(0):
        final_mapping = [0] * k
        for i, tv in enumerate(order):
            final_mapping[tv] = mapping[i]
        return final_mapping
    
    return None
# EVOLVE-BLOCK-END
