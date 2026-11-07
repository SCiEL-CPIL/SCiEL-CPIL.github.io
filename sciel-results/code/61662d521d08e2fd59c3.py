import time
from random import Random
from collections import defaultdict

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
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    trees = instance_trees(n, 611)
    
    # Precompute tree properties
    tree_adj = []
    tree_deg = []
    tree_size = []
    for t in trees:
        k = len(t) + 1 if t else 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
        tree_size.append(k)
        
    # Sort trees by size descending
    order = list(range(n))
    order.sort(key=lambda x: tree_size[x], reverse=True)
    
    # Precompute host graph edges
    edge_idx = {}
    edge_verts = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx
            edge_verts.append((u, v))
            idx += 1
            
    # Adjacency list of edge indices for each vertex
    adj_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_verts):
        adj_edges[u].append(i)
        adj_edges[v].append(i)
        
    # Bipartite matching helper using Hopcroft-Karp
    def hopcroft_karp(left_nodes, right_nodes, edges):
        """
        left_nodes: list of left node indices
        right_nodes: list of right node indices  
        edges: dict mapping left_node -> list of right_nodes it connects to
        Returns: matching dict left_node -> right_node for matched pairs
        """
        if not left_nodes:
            return {}
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Build adjacency
        adj = [[] for _ in range(n_left)]
        for li, ln in enumerate(left_nodes):
            for rn in edges.get(ln, []):
                if rn in right_nodes:
                    adj[li].append(rn)
        
        # BFS to find augmenting paths
        def bfs():
            dist = [-1] * n_left
            queue = []
            for i in range(n_left):
                if matching_l[i] == -1:
                    dist[i] = 0
                    queue.append(i)
                else:
                    dist[i] = -1
            found = False
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    ri = right_to_idx.get(v, -1)
                    if ri == -1:
                        continue
                    if matching_r[ri] == -1:
                        found = True
                    elif dist[matching_r[ri]] == -1:
                        dist[matching_r[ri]] = dist[u] + 1
                        queue.append(matching_r[ri])
            return found
        
        # DFS to find augmenting path
        def dfs(u):
            for v in adj[u]:
                ri = right_to_idx.get(v, -1)
                if ri == -1:
                    continue
                if matching_r[ri] == -1 or (dist[matching_r[ri]] == dist[u] + 1 and dfs(matching_r[ri])):
                    matching_l[u] = ri
                    matching_r[ri] = u
                    return True
            dist[u] = -1
            return False
        
        # Initialize matching
        matching_l = [-1] * n_left
        matching_r = [-1] * n_right
        right_to_idx = {rn: ri for ri, rn in enumerate(right_nodes)}
        
        result = 0
        while bfs():
            for i in range(n_left):
                if matching_l[i] == -1 and dfs(i):
                    result += 1
        
        # Convert back to original node indices
        matching = {}
        for i in range(n_left):
            if matching_l[i] != -1:
                matching[left_nodes[i]] = right_nodes[matching_l[i]]
        
        return matching
    
    # Greedy with backtracking for each tree
    def try_embed_tree(t_idx, avail_mask, used_mask, rng):
        """Try to embed tree t_idx using greedy + limited backtracking"""
        k = t_idx + 1
        if k == 1:
            # Single vertex
            candidates = [v for v in range(n) if not (used_mask & (1 << v))]
            if not candidates:
                return None, avail_mask, used_mask, False
            v = rng.choice(candidates)
            used_mask |= (1 << v)
            return [v], avail_mask, used_mask, True
        
        adj = tree_adj[t_idx]
        deg = tree_deg[t_idx]
        edges = trees[t_idx]
        
        # Root at highest degree vertex
        max_deg = max(deg)
        high_deg_nodes = [i for i in range(k) if deg[i] == max_deg]
        
        # Try different roots
        roots_to_try = high_deg_nodes[:]
        rng.shuffle(roots_to_try)
        
        for root in roots_to_try:
            # Use bipartite matching to assign vertices
            # Left side: tree vertices (except root)
            # Right side: available host vertices
            # Edge exists if tree edge (u,v) can map to host edge (gu,gv)
            
            # Build constraint graph
            # For each tree vertex v (except root), find which host vertices it can map to
            # Given parent mapping, v must be adjacent to parent's host vertex via an available edge
            
            # We'll do a layered assignment: BFS from root
            # Level 0: root
            # Level 1: neighbors of root
            # etc.
            
            # First, get BFS levels from root
            parent_in_tree = [-1] * k
            level = [-1] * k
            queue = [root]
            level[root] = 0
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if level[v] == -1:
                        level[v] = level[u] + 1
                        parent_in_tree[v] = u
                        queue.append(v)
            
            # Get vertices by level
            max_level = max(level) if level else 0
            by_level = defaultdict(list)
            for v in range(k):
                by_level[level[v]].append(v)
            
            # Assign root first
            root_candidates = [v for v in range(n) if not (used_mask & (1 << v))]
            if not root_candidates:
                continue
            
            # Score root candidates by available degree
            def score_host(v, mask):
                count = 0
                for e_idx in adj_edges[v]:
                    if mask & (1 << e_idx):
                        count += 1
                return count
            
            root_candidates.sort(key=lambda v: score_host(v, avail_mask), reverse=True)
            # Add some randomness
            for i in range(min(len(root_candidates), 5)):
                j = rng.randint(0, i)
                root_candidates[i], root_candidates[j] = root_candidates[j], root_candidates[i]
            
            for root_host in root_candidates[:10]:  # Try top 10
                mapping = [-1] * k
                mapping[root] = root_host
                local_used = {root_host}
                local_avail = avail_mask
                local_used_mask = used_mask | (1 << root_host)
                
                success = True
                # Process levels 1, 2, ...
                for lv in range(1, max_level + 1):
                    if time.time() > deadline:
                        return None, avail_mask, used_mask, False
                    
                    verts_at_level = by_level[lv]
                    if not verts_at_level:
                        continue
                    
                    # For each vertex at this level, determine valid host candidates
                    # A vertex v at level lv has parent p = parent_in_tree[v]
                    # v can map to host h if:
                    # 1. h is not in local_used
                    # 2. h is not in local_used_mask
                    # 3. There's an available edge between mapping[p] and h
                    
                    # Build bipartite matching:
                    # Left: verts_at_level
                    # Right: available host vertices
                    # Edge (v, h) exists if h is adjacent to mapping[parent_in_tree[v]] via available edge
                    
                    left_nodes = verts_at_level
                    right_candidates = set()
                    for v in verts_at_level:
                        p = parent_in_tree[v]
                        ph = mapping[p]
                        for e_idx in adj_edges[ph]:
                            if local_avail & (1 << e_idx):
                                u, w = edge_verts[e_idx]
                                h = w if u == ph else u
                                if h not in local_used and not (local_used_mask & (1 << h)):
                                    right_candidates.add(h)
                    
                    if not right_candidates:
                        success = False
                        break
                    
                    right_nodes = list(right_candidates)
                    
                    # Build edges for matching
                    match_edges = defaultdict(list)
                    for v in left_nodes:
                        p = parent_in_tree[v]
                        ph = mapping[p]
                        for e_idx in adj_edges[ph]:
                            if local_avail & (1 << e_idx):
                                u, w = edge_verts[e_idx]
                                h = w if u == ph else u
                                if h in right_candidates:
                                    match_edges[v].append(h)
                    
                    # Run Hopcroft-Karp
                    matching = hopcroft_karp(left_nodes, right_nodes, match_edges)
                    
                    if len(matching) != len(left_nodes):
                        success = False
                        break
                    
                    # Apply matching
                    for v, h in matching.items():
                        mapping[v] = h
                        local_used.add(h)
                        local_used_mask |= (1 << h)
                        # Remove edges used by this vertex to its parent
                        p = parent_in_tree[v]
                        ph = mapping[p]
                        e_idx = edge_idx[(ph, h)]
                        local_avail &= ~(1 << e_idx)
                
                if success:
                    # Verify all edges are properly mapped and update avail_mask
                    for u, v in edges:
                        gu, gv = mapping[u], mapping[v]
                        e_idx = edge_idx[(gu, gv)]
                        if not (avail_mask & (1 << e_idx)):
                            success = False
                            break
                        avail_mask &= ~(1 << e_idx)
                    
                    if success:
                        return mapping, avail_mask, local_used_mask, True
        
        return None, avail_mask, used_mask, False
    
    # Main search loop
    best_solution = None
    best_value = 0
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 12345 + 678)
        
        avail_mask = (1 << 190) - 1
        used_mask = 0
        current_solution = [None] * n
        success = True
        packed_edges = 0
        
        for t_idx in order:
            k = t_idx + 1
            if k == 1:
                candidates = [v for v in range(n) if not (used_mask & (1 << v))]
                if not candidates:
                    success = False
                    break
                v = rng.choice(candidates)
                used_mask |= (1 << v)
                current_solution[t_idx] = [v]
                continue
            
            mapping, avail_mask, used_mask, ok = try_embed_tree(t_idx, avail_mask, used_mask, rng)
            if not ok:
                success = False
                break
            
            current_solution[t_idx] = mapping
            packed_edges += len(trees[t_idx])
        
        if success:
            value = sum(len(t) for t in trees if current_solution[trees.index(t)] is not None)
            if value > best_value:
                best_value = value
                best_solution = current_solution[:]
            if value == 190:
                break
    
    # Fallback if no complete solution found
    if best_solution is None:
        best_solution = [None] * n
        best_solution[0] = [0]
        best_solution[1] = [0, 1]
    
    return best_solution
