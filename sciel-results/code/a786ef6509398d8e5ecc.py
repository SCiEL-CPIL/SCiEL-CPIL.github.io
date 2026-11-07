import time
from random import Random

def instance_trees(n=20, seed=611):
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
    deadline = start_time + 145.0  # 5s safety margin
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute all edges in K20 for quick lookup? No, just use a set.
    
    def find_embedding(tree_edges, used_edges_global):
        """
        Finds an injective embedding of tree_edges into K20 
        using edges not in used_edges_global.
        Returns a list of length k (number of vertices in tree) 
        where result[i] is the host vertex for tree vertex i, or None.
        """
        k = len(tree_edges) + 1
        if k == 1:
            # Find any host vertex
            # Any vertex is fine, pick 0
            return [0]
        
        # Build adjacency list for the tree
        adj = [[] for _ in range(k)]
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Determine BFS order starting from 0
        # This gives a parent-child relationship
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        
        # Backtracking variables
        host = [-1] * k  # host[i] is host vertex for tree vertex i
        used_hosts = set()
        
        # We assign hosts in the order defined by 'order'
        # order[0] is 0.
        
        def backtrack(idx):
            if idx == k:
                return True
            
            tv = order[idx]
            # Try all possible host vertices 0..n-1
            # Heuristic: try vertices that are "close" to parent's host? 
            # Or just random shuffle to avoid bias?
            # For speed, let's try 0..n-1 but maybe shuffle if we fail often.
            # Let's stick to 0..n-1 for determinism in single run, 
            # but we will run multiple times with different seeds for the "try" logic if we add randomization.
            # However, simple backtracking might get stuck. 
            # Let's add a simple randomization: shuffle candidates.
            candidates = list(range(n))
            # To make it robust, we'll just iterate. If it fails, the outer loop will retry with different global state (if we restart) or different order.
            # But here we are fixing the global state. 
            # If this specific tree fails, we return None.
            
            # Optimization: If tv has a parent, the edge to parent must be free.
            p_tv = parent[tv]
            
            for hv in candidates:
                if hv in used_hosts:
                    continue
                
                # Check edge to parent
                if p_tv != -1:
                    p_hv = host[p_tv]
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges_global:
                        continue
                    # Also check if this edge was used by a previous tree in this embedding? 
                    # No, used_edges_global contains all edges used by PREVIOUS trees.
                    # Edges used by the CURRENT tree are not in used_edges_global yet.
                    # We must ensure edges within the current tree are distinct.
                    # Since the tree is a tree, and we are assigning vertices injectively, 
                    # and we are adding edges one by one, do we need to check for duplicate edges within the tree?
                    # A tree has distinct edges. If the mapping is injective, 
                    # can two different tree edges map to the same host edge?
                    # Yes, e.g., tree (0-1, 0-2) mapped to (5-6, 5-6) is impossible because 1 and 2 map to distinct hosts.
                    # If tree edges are distinct, and vertex mapping is injective, 
                    # then edge mapping is injective?
                    # Edge (u,v) maps to (H_u, H_v). Edge (x,y) maps to (H_x, H_y).
                    # If (H_u, H_v) == (H_x, H_y), then {H_u, H_v} == {H_x, H_y}.
                    # Since H is injective, {u, v} must be {x, y}.
                    # So distinct tree edges map to distinct host edges.
                    # So we only need to check against used_edges_global.
                    
                    # Mark as used for this step? 
                    # We don't need to add to used_hosts for edges, just vertices.
                    # But we need to "reserve" the edge so it's not used by another branch? 
                    # No, backtracking handles this by trying other hv.
                    # We don't need a local set of edges if we rely on the fact that 
                    # injective vertex mapping implies injective edge mapping for a tree.
                    # Wait, is that true?
                    # Tree: 0-1, 1-2. Map: 0->5, 1->6, 2->5? No, injective vertices.
                    # So yes, injective vertices -> injective edges.
                    # So we ONLY need to check if the edge is in used_edges_global.
                    
                # Tentatively assign
                host[tv] = hv
                used_hosts.add(hv)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                used_hosts.remove(hv)
                host[tv] = -1
            
            return False
        
        return backtrack(0)

    # Packing strategy:
    # We want to embed all 20 trees.
    # We process them in an order that is likely to succeed.
    # Larger trees have more edges and are more constrained.
    # Let's sort indices by tree size (number of edges) descending.
    tree_indices = list(range(n))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    results = [None] * n
    used_edges_global = set()
    
    # We might need to restart if we get stuck.
    # Let's try a few random permutations of the processing order?
    # Or just try the sorted order. If it fails, try another seed for randomization in backtracking?
    # The current backtracking is deterministic (0..n-1). 
    # To improve success rate, we can randomize the candidate order in backtracking.
    
    def try_pack(seed):
        rng = Random(seed)
        used = set()
        res = [None] * n
        
        # Shuffle the processing order slightly? 
        # No, keep largest first. 
        # But randomize the "candidates" in the backtracking.
        
        # We need to pass rng to the backtracking function to shuffle candidates.
        # Let's redefine find_embedding to accept rng.
        
        def find_embedding_random(tree_edges, used_edges_global, rng):
            k = len(tree_edges) + 1
            if k == 1:
                return [0]
            
            adj = [[] for _ in range(k)]
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            parent = [-1] * k
            order = [0]
            visited = [False] * k
            visited[0] = True
            queue = [0]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        order.append(v)
                        queue.append(v)
                        
            host = [-1] * k
            used_hosts = set()
            
            def backtrack(idx):
                if idx == k:
                    return True
                
                tv = order[idx]
                candidates = list(range(n))
                rng.shuffle(candidates)
                
                p_tv = parent[tv]
                
                for hv in candidates:
                    if hv in used_hosts:
                        continue
                    
                    if p_tv != -1:
                        p_hv = host[p_tv]
                        e = (min(hv, p_hv), max(hv, p_hv))
                        if e in used_edges_global:
                            continue
                    
                    host[tv] = hv
                    used_hosts.add(hv)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    used_hosts.remove(hv)
                    host[tv] = -1
                
                return False
            
            return backtrack(0)

        for idx in tree_indices:
            if time.time() > deadline:
                return None, 0
                
            tree = trees[idx]
            embedding = find_embedding_random(tree, used, rng)
            
            if embedding is None:
                return None, 0 # Failed to embed this tree
            
            # Record result
            res[idx] = embedding
            
            # Update used edges
            for u, v in tree:
                hu = embedding[u]
                hv = embedding[v]
                e = (min(hu, hv), max(hu, hv))
                used.add(e)
                
        # Calculate score
        score = 0
        for i in range(n):
            if res[i] is not None:
                score += len(trees[i])
        
        return res, score

    best_res = None
    best_score = 0
    
    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
            
        res, score = try_pack(seed)
        
        if res is not None:
            if score > best_score:
                best_score = score
                best_res = res
                
            if best_score == 190:
                break
                
    if best_res is None:
        # If we failed completely, return a valid but low score result?
        # We must return 20 entries. 
        # If we can't embed any, we can't return None for all? 
        # "Entry k-1 is None or a list..."
        # So we can return None for failed trees.
        # But we should try to embed at least some.
        # The loop above returns None if ANY tree fails. 
        # This is too strict. We should try to embed as many as possible.
        
        # Let's modify try_pack to not fail immediately, but just skip the tree?
        # No, if a large tree fails, it blocks smaller ones? No, they are independent except for edges.
        # If a tree fails, we can just leave it as None and continue with others.
        # However, if we skip a tree, we don't use its edges, so others have more freedom.
        # So "greedy" should be: try to embed. If fail, skip (leave None). 
        # But we want to maximize total edges. 
        # Maybe trying a different order or seed helps.
        
        # Let's adjust try_pack to allow skipping.
        
        def try_pack_greedy(seed):
            rng = Random(seed)
            used = set()
            res = [None] * n
            score = 0
            
            for idx in tree_indices:
                if time.time() > deadline:
                    break
                    
                tree = trees[idx]
                embedding = find_embedding_random(tree, used, rng)
                
                if embedding is not None:
                    res[idx] = embedding
                    score += len(tree)
                    for u, v in tree:
                        hu = embedding[u]
                        hv = embedding[v]
                        e = (min(hu, hv), max(hu, hv))
                        used.add(e)
                        
            return res, score

        best_res = None
        best_score = 0
        
        for seed in range(200):
            if time.time() > deadline:
                break
            
            res, score = try_pack_greedy(seed)
            
            if score > best_score:
                best_score = score
                best_res = res
                
            if best_score == 190:
                break
        
        if best_res is None:
            # Fallback: embed only T1 and T2?
            # T1 is 1 vertex. T2 is 1 edge.
            # We can always embed T1 (any vertex) and T2 (any edge).
            fallback = [None] * n
            fallback[0] = [0]
            fallback[1] = [0, 1]
            return fallback

        return best_res

    return best_res

