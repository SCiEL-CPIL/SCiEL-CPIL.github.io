from random import Random

def instance_trees(n=20, seed=611):
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
    trees = instance_trees(20, 611)
    n = 20
    
    # Track which K20 edges are used
    used_edges = set()
    
    # Result: list of 20 entries, entry k-1 is None or list of k host vertices
    result = [None] * 20
    
    # Process trees in order of decreasing size (harder ones first)
    tree_indices = list(range(20))
    tree_indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    for idx in tree_indices:
        k = idx + 1  # Tree T_k has k vertices
        if k == 1:
            result[idx] = None
            continue
        if k == 2:
            # Find any available edge
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in used_edges:
                        used_edges.add((u, v))
                        result[idx] = [u, v]
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For larger trees, use backtracking to find an embedding
        tree_edges = trees[idx]
        
        # Build adjacency for the tree
        adj = [[] for _ in range(k)]
        for (a, b) in tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Determine a BFS order for the tree starting from vertex 0
        # This ensures we assign parents before children
        order = []
        visited = [False] * k
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        # assigned[i] will be the host vertex for tree vertex order[i]
        # We will fill this array via backtracking
        assigned = [-1] * k
        
        def backtrack(pos):
            if pos == k:
                return True
            
            tree_v = order[pos]
            # If this is not the root, it must be connected to its parent in the tree
            # Find the parent in the BFS tree
            # Actually, in the BFS order, the parent is the first neighbor that was visited before it?
            # We can precompute the parent in the BFS tree.
            
            # For the root (pos=0), we try all hosts.
            # For others, we try hosts adjacent to the parent's host.
            
            if pos == 0:
                candidates = list(range(n))
            else:
                # Find parent of tree_v in the BFS tree
                # The parent is the neighbor in adj[tree_v] that appears earlier in 'order'
                parent_v = -1
                for neighbor in adj[tree_v]:
                    if visited[neighbor]: # This check is always true for neighbors in a tree?
                         # We need to know which neighbor is the parent in the BFS tree.
                         # Let's precompute parent array.
                         pass
                
                # This logic is getting complex inside the closure. Let's precompute parents.
                pass
            
            return False

        # Precompute parent and order
        parent = [-1] * k
        order = []
        visited = [False] * k
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        
        assigned = [-1] * k
        
        def dfs(pos):
            if pos == k:
                return True
            
            tree_v = order[pos]
            
            if pos == 0:
                # Root: try all available hosts
                for h in range(n):
                    assigned[tree_v] = h
                    if dfs(pos + 1):
                        return True
                    assigned[tree_v] = -1
            else:
                # Non-root: must be adjacent to parent's host
                p_tree_v = parent[tree_v]
                p_host = assigned[p_tree_v]
                
                # Find available neighbors of p_host in K20
                # A neighbor h is valid if:
                # 1. h != p_host
                # 2. (p_host, h) is not in used_edges (global)
                # 3. (p_host, h) is not already used by the current partial embedding
                #    We can check if any other assigned vertex is h? No, vertices are distinct.
                #    We need to check if the edge is already taken by another edge in the current tree.
                #    Since it's a tree, each vertex (except root) has exactly one parent edge.
                #    So we just need to ensure the edge (p_host, h) is not used by *another* edge in the current tree.
                #    However, since we are building the tree, the only edges added are (parent, child).
                #    Could (p_host, h) be the edge for a different child of p_host?
                #    Yes. If p_host has two children in the tree, say c1 and c2.
                #    If we assign h to c1, then for c2, we cannot use h again (vertex distinct).
                #    But could we use the *edge* (p_host, h) for c2? No, because h is already used by c1.
                #    So vertex distinctness implies edge distinctness for the star at p_host.
                #    What if the edge is used by a non-parent edge?
                #    In a tree, there are no cycles. So no other edge connects p_host to h.
                #    So we just need to check:
                #    - h is not in assigned (vertex distinct)
                #    - (p_host, h) not in used_edges (global)
                
                for h in range(n):
                    if h == p_host:
                        continue
                    # Check if h is already assigned to another vertex in the current tree
                    if h in assigned:
                        continue
                    
                    # Check edge availability
                    a, b = p_host, h
                    if a > b: a, b = b, a
                    if (a, b) in used_edges:
                        continue
                    
                    # Additional check: Is this edge already used by the current partial embedding?
                    # As reasoned above, if h is not in assigned, the edge (p_host, h) cannot be used
                    # by any other edge in the current tree, because that would require h to be assigned
                    # to the other endpoint.
                    
                    assigned[tree_v] = h
                    if dfs(pos + 1):
                        return True
                    assigned[tree_v] = -1
            
            return False
        
        success = False
        # We can optimize by trying roots that have high degree in K20? 
        # Or just try all.
        for root_host in range(n):
            assigned[0] = root_host
            # We need to clear assigned for others? 
            # The dfs sets assigned[tree_v] = -1 on backtrack.
            # But initially, all are -1 except assigned[0].
            # We should reset assigned to all -1 before each root attempt?
            # Yes, because the previous attempt might have left some values if it failed?
            # The dfs cleans up on failure. But to be safe, reset.
            assigned = [-1] * k
            assigned[0] = root_host
            
            if dfs(1): # Start from pos 1, since 0 is assigned
                # Success
                # Commit edges to used_edges
                for (a, b) in tree_edges:
                    ha, hb = assigned[a], assigned[b]
                    if ha > hb:
                        ha, hb = hb, ha
                    used_edges.add((ha, hb))
                
                # Map back to original vertex indices
                # result[idx] should be a list of length k, where result[idx][i] is the host for tree vertex i
                res_map = [0] * k
                for i, tree_v in enumerate(order):
                    res_map[tree_v] = assigned[tree_v]
                
                result[idx] = res_map
                success = True
                break
        
        if not success:
            result[idx] = None
            
    return result
