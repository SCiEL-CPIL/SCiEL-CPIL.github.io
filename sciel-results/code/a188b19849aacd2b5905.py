import time
import sys
import random
from random import Random

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
    
    # Total edges in all trees should be 190
    total_tree_edges = sum(len(t) for t in trees)
    # assert total_tree_edges == 190
    
    # Process trees in order of decreasing size (most constrained first)
    # Tree index k-1 has k vertices and k-1 edges
    order = list(range(19, -1, -1))  # T20, T19, ..., T1
    
    # We'll try multiple randomized strategies
    best_result = None
    best_score = -1
    
    # Strategy: For each tree, try multiple embedding approaches
    # with randomized restarts if stuck
    
    def get_avail_adj(available_edges):
        adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj
    
    def try_embed_tree(tree_idx, available_edges, rng, time_limit=2.0):
        """Try to embed tree tree_idx into available_edges.
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
        
        for root in leaves:
            if time.time() > start_time + time_limit:
                continue
            
            # Build parent-child structure rooted at root
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[root] = True
            stack = [root]
            order_list = [root]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        stack.append(v)
                        order_list.append(v)
            
            # Shuffle children for randomization
            for v in range(k):
                rng.shuffle(children[v])
            
            host = [-1] * k
            used_hosts = set()
            
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
                def score_candidate(h):
                    count = 0
                    for nb in avail_adj[h]:
                        if nb not in used_hosts:
                            count += 1
                    return count
                
                valid_candidates.sort(key=score_candidate)
                
                for h in valid_candidates:
                    host[v] = h
                    used_hosts.add(h)
                    
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    
                    host[v] = -1
                    used_hosts.remove(h)
                
                return None
            
            result = backtrack(0)
            if result is not None:
                return result
        
        return None
    
    # Main search with randomized restarts
    max_restarts = 50
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
                found = None
                # Pick a random edge
                edge_list = list(available)
                if not edge_list:
                    success = False
                    break
                (a, b) = rng.choice(edge_list)
                result[t_idx] = [a, b]
                available.discard((min(a, b), max(a, b)))
                continue
            
            emb = try_embed_tree(t_idx, available, rng, time_limit=3.0)
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
    
    # If we get here, try a more aggressive approach with longer time limits
    # on the last few trees
    rng = Random(99999)
    result = [None] * 20
    available = set(all_edges)
    success = True
    
    for t_idx in order:
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
        
        # Give more time for larger trees
        time_limit = min(10.0, (deadline - 2 - time.time()) * 0.5)
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
    
    # Fallback: return best partial result (will score 0 if invalid, but try)
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    return result
