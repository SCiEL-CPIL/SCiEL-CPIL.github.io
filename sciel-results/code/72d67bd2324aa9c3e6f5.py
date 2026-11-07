import time
import random
from itertools import permutations

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
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Process trees in order: largest first (most constrained)
    # Tree k has k-1 edges, so order is k=20,19,...,2,1
    order = list(range(19, -1, -1))  # indices 19 down to 0
    
    def try_embed_dfs(tree_idx, used_edges, embedding, depth=0):
        """
        Try to embed tree_idx (0-indexed, so tree k = tree_idx+1) 
        using DFS/backtracking on vertex assignments.
        Returns True if successful, updates used_edges and embedding.
        """
        k = tree_idx + 1
        if k == 1:
            embedding[tree_idx] = None
            return True
        
        adj = tree_adj[tree_idx]
        # Root the tree at vertex 0
        # Build a BFS/DFS ordering of the tree vertices
        parent = [-1] * k
        order_list = [0]
        parent[0] = -2
        for v in order_list:
            for nb in adj[v]:
                if parent[nb] == -1:
                    parent[nb] = v
                    order_list.append(nb)
        
        # We'll assign vertices in order_list order
        # mapping: tree_vertex -> host_vertex
        mapping = [None] * k
        mapping[0] = None  # will be set
        
        def dfs(pos, host_choices):
            """
            Try to assign host vertices for tree vertices in order_list[pos:]
            host_choices: list of available host vertices
            """
            if time.time() > deadline:
                return False
            
            if pos == len(order_list):
                # All vertices assigned, verify edges
                # (should already be verified during assignment)
                return True
            
            tv = order_list[pos]
            
            # Try each available host vertex
            for i, hv in enumerate(host_choices):
                # Check if all edges from tv to already-assigned neighbors are free
                ok = True
                for nb in adj[tv]:
                    if parent[nb] != -1 and parent[nb] != -2:
                        # nb is already assigned (parent or earlier in order)
                        if mapping[nb] is not None:
                            edge = (min(hv, mapping[nb]), max(hv, mapping[nb]))
                            if edge in used_edges:
                                ok = False
                                break
                        else:
                            # nb should be assigned if it's before tv in order
                            # Actually, we assign in order_list order, so if nb is before tv, it's assigned
                            # If nb is after tv, it's not assigned yet
                            pass
                
                if not ok:
                    continue
                
                # Also check edges to children that might have constraints
                # For now, just check edges to already-assigned vertices
                
                mapping[tv] = hv
                # Remove hv from available
                remaining = host_choices[:i] + host_choices[i+1:]
                
                if dfs(pos + 1, remaining):
                    return True
                
                mapping[tv] = None
            
            return False
        
        # Try all permutations is too expensive for k up to 20
        # Instead, use random sampling with some structure
        
        return False
    
    def random_attempt(seed_val):
        """Try one random embedding strategy."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed_val)
        used_edges = set()
        embedding = [None] * n
        
        for idx in order:
            k = idx + 1
            if k == 1:
                embedding[idx] = None
                continue
            
            tree_edges = trees[idx]
            adj = tree_adj[idx]
            
            # Build BFS order rooted at 0
            parent = [-1] * k
            order_list = [0]
            parent[0] = -2
            for v in order_list:
                for nb in adj[v]:
                    if parent[nb] == -1:
                        parent[nb] = v
                        order_list.append(nb)
            
            # Try multiple random assignments
            max_tries = 5000 if k <= 12 else 2000
            found = False
            
            for attempt in range(max_tries):
                if time.time() > deadline:
                    break
                
                # Randomly assign host vertices
                host_verts = rng.sample(range(n), k)
                mapping = [None] * k
                for i, tv in enumerate(order_list):
                    mapping[tv] = host_verts[i]
                
                # Check all edges
                ok = True
                for (u, v) in tree_edges:
                    a, b = mapping[u], mapping[v]
                    edge = (min(a, b), max(a, b))
                    if edge in used_edges:
                        ok = False
                        break
                
                if ok:
                    # Commit
                    for (u, v) in tree_edges:
                        a, b = mapping[u], mapping[v]
                        edge = (min(a, b), max(a, b))
                        used_edges.add(edge)
                    embedding[idx] = [mapping[i] for i in range(k)]
                    found = True
                    break
            
            if not found:
                return None
        
        total_edges = len(used_edges)
        return (total_edges, embedding, used_edges)
    
    best_edges = -1
    best_embedding = None
    
    # Try many random restarts
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        result = random_attempt(611 + restart)
        if result is not None:
            total_edges, embedding, used_edges = result
            if total_edges > best_edges:
                best_edges = total_edges
                best_embedding = [list(e) if e is not None else None for e in embedding]
        
        # Also try a greedy approach with different orderings
        if restart % 10 == 0:
            # Try with slightly different random seed variations
            for extra in range(5):
                if time.time() > deadline - 5:
                    break
                result = random_attempt(100000 + restart * 1000 + extra)
                if result is not None:
                    total_edges, embedding, used_edges = result
                    if total_edges > best_edges:
                        best_edges = total_edges
                        best_embedding = [list(e) if e is not None else None for e in embedding]
    
    if best_embedding is not None and best_edges > 0:
        return best_embedding
    
    # Fallback: embed T20 only
    return [None] * 19 + [list(range(20))]
