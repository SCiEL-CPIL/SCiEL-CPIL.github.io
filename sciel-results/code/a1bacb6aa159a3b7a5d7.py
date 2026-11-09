import time
import sys
import random
from random import Random
from collections import deque

def _generate_trees(n=20, seed=611):
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
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    trees = _generate_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Process trees in order of decreasing size (most constrained first)
    # Tree index k-1 has k vertices and k-1 edges
    order = list(range(19, -1, -1))  # T20, T19, ..., T1
    
    def get_avail_adj(available_edges):
        adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj
    
    def try_embed_tree(tree_idx, available_edges, rng, time_limit=2.0):
        """Try to embed tree tree_idx into available_edges using improved backtracking.
        Returns embedding (list of k host vertices) or None."""
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for (a, b) in available_edges:
                return [a, b]
            return None
        
        tree_edges = trees[tree_idx]
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (u, v) in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find all leaves
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            leaves = [0]
        
        # Try multiple roots (leaves) with randomization
        rng.shuffle(leaves)
        
        avail_adj = get_avail_adj(available_edges)
        avail_deg = [len(avail_adj[i]) for i in range(n)]
        
        for root in leaves:
            if time.time() > start_time + time_limit:
                continue
            
            # Build parent-child structure rooted at root using BFS for better ordering
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[root] = True
            queue = deque([root])
            order_list = [root]
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        queue.append(v)
                        order_list.append(v)
            
            # Shuffle children for randomization
            for v in range(k):
                rng.shuffle(children[v])
            
            host = [-1] * k
            used_hosts = set()
            # Track remaining available neighbors for each used host
            remaining_neigh = {i: set(avail_adj[i]) - used_hosts for i in range(n)}
            
            def backtrack(idx):
                if time.time() > start_time + time_limit:
                    return None
                
                if idx == len(order_list):
                    # Verify all tree edges are in available_edges
                    for (u, v) in tree_edges:
                        hu, hv = host[u], host[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e not in available_edges:
                            return None
                    return list(host)
                
                v = order_list[idx]
                
                if parent[v] == -1:
                    # Root, any host vertex
                    candidates = list(range(n))
                else:
                    pv = parent[v]
                    if host[pv] == -1:
                        return None
                    candidates = list(avail_adj[host[pv]])
                
                # Filter candidates: must not be used
                valid_candidates = [h for h in candidates if h not in used_hosts]
                
                if not valid_candidates:
                    return None
                    
                # Heuristic: Sort by degree in available graph (most constrained first)
                # Use remaining neighbors count for better pruning
                def score_candidate(h):
                    count = len(remaining_neigh[h])
                    return count
                
                valid_candidates.sort(key=score_candidate)
                
                for h in valid_candidates:
                    host[v] = h
                    used_hosts.add(h)
                    # Update remaining neighbors
                    for nb in avail_adj[h]:
                        if nb in remaining_neigh:
                            remaining_neigh[nb].discard(h)
                    remaining_neigh[h] = set()
                    
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    
                    # Undo
                    host[v] = -1
                    used_hosts.discard(h)
                    remaining_neigh[h] = set(avail_adj[h]) - used_hosts
                    for nb in avail_adj[h]:
                        if nb in remaining_neigh:
                            remaining_neigh[nb].add(h)
                
                return None
            
            result = backtrack(0)
            if result is not None:
                return result
        
        return None
    
    # Main search with randomized restarts
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        rng = Random(restart * 1000 + 42)
        result = [None] * 20
        available = set(all_edges)
        success = True
        
        for t_idx in order:
            if time.time() > deadline - 3:
                success = False
                break
                
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [0]
                continue
            elif k == 2:
                edge_list = list(available)
                if not edge_list:
                    success = False
                    break
                (a, b) = rng.choice(edge_list)
                result[t_idx] = [a, b]
                available.discard((min(a, b), max(a, b)))
                continue
            
            # Adaptive time limit: more time for harder (larger) trees
            time_limit = min(5.0, max(1.0, (deadline - 3 - time.time()) * 0.3 / (20 - t_idx)))
            emb = try_embed_tree(t_idx, available, rng, time_limit=time_limit)
            if emb is None:
                success = False
                break
            
            result[t_idx] = emb
            for (u, v) in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                if e in available:
                    available.discard(e)
                else:
                    success = False
                    break
            if not success:
                break
        
        if success:
            # Verify solution
            used_edges = set()
            valid = True
            for t_idx in range(20):
                if result[t_idx] is None:
                    valid = False
                    break
                emb = result[t_idx]
                if len(emb) != t_idx + 1:
                    valid = False
                    break
                if len(set(emb)) != t_idx + 1:
                    valid = False
                    break
                for x in emb:
                    if not isinstance(x, int) or x < 0 or x >= 20:
                        valid = False
                        break
                if not valid:
                    break
                for (u, v) in trees[t_idx]:
                    hu, hv = emb[u], emb[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
            if valid:
                # Ensure proper types
                for i in range(20):
                    if result[i] is not None:
                        result[i] = [int(x) for x in result[i]]
                return result
    
    # Fallback: try with different ordering (increasing size)
    order_inc = list(range(20))
    rng = Random(77777)
    result = [None] * 20
    available = set(all_edges)
    success = True
    
    for t_idx in order_inc:
        if time.time() > deadline - 2:
            success = False
            break
            
        k = t_idx + 1
        if k == 1:
            result[t_idx] = [0]
            continue
        elif k == 2:
            edge_list = list(available)
            if not edge_list:
                success = False
                break
            (a, b) = rng.choice(edge_list)
            result[t_idx] = [a, b]
            available.discard((min(a, b), max(a, b)))
            continue
        
        time_limit = min(5.0, max(1.0, (deadline - 2 - time.time()) * 0.3))
        emb = try_embed_tree(t_idx, available, rng, time_limit=time_limit)
        if emb is None:
            success = False
            break
        
        result[t_idx] = emb
        for (u, v) in trees[t_idx]:
            hu, hv = emb[u], emb[v]
            e = (min(hu, hv), max(hu, hv))
            if e in available:
                available.discard(e)
            else:
                success = False
                break
        if not success:
            break
    
    if success:
        for i in range(20):
            if result[i] is not None:
                result[i] = [int(x) for x in result[i]]
        return result
    
    # Final fallback: return best partial result
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    return result
