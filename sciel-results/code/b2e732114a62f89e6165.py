"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    import random
    from collections import deque, defaultdict
    
    deadline = time.time() + 85
    n = P['n']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Compute k-core decomposition
    def compute_k_cores():
        """Return list of cores where cores[k] = set of vertices in k-core"""
        deg = [len(adj[i]) for i in range(n)]
        removed = [False] * n
        core_num = [0] * n
        max_k = max(deg) if n > 0 else 0
        
        # Standard k-core peeling
        # Use bucket-based approach
        buckets = defaultdict(list)
        for v in range(n):
            buckets[deg[v]].append(v)
        
        # Process vertices in increasing degree order
        for k in range(max_k + 1):
            # Process all vertices with current degree == k
            queue = list(buckets[k])
            while queue:
                v = queue.pop(0)
                if removed[v]:
                    continue
                if deg[v] < k:
                    continue
                removed[v] = True
                core_num[v] = k
                for u in adj[v]:
                    if not removed[u] and deg[u] >= k:
                        deg[u] -= 1
                        if deg[u] == k:
                            buckets[k].append(u)
        
        # Reconstruct: vertices not removed are in the highest core
        # Actually, let's do a cleaner version
        return core_num
    
    # Better k-core computation
    def get_k_core(k):
        """Return set of vertices in k-core"""
        deg = [len(adj[i]) for i in range(n)]
        removed = [False] * n
        in_core = [True] * n
        
        # Remove vertices with degree < k iteratively
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if not in_core[v]:
                    continue
                actual_deg = sum(1 for u in adj[v] if in_core[u])
                if actual_deg < k:
                    in_core[v] = False
                    changed = True
        
        return set(v for v in range(n) if in_core[v])
    
    # Find the maximum k such that k-core has at least k vertices (clique bound)
    # The max clique is at most the size of any k-core where k >= clique_size
    # But more usefully: a clique of size c must be contained in the c-core
    
    # Strategy: Find max clique using branch-and-bound but seeded with k-core insights
    # First, find a good initial clique from the k-core structure
    
    best_clique = []
    
    # Try to find cliques within dense regions using k-core guided search
    # The maximum clique must be in the max_k core
    
    # Compute core numbers properly
    def compute_core_numbers():
        """Compute core number for each vertex using Nagamachi's algorithm"""
        core_num = [0] * n
        deg = [len(adj[i]) for i in range(n)]
        
        # Sort vertices by degree
        vertices = list(range(n))
        vertices.sort(key=lambda v: -deg[v])
        
        # Process vertices in order of decreasing degree
        for i, v in enumerate(vertices):
            # Core number of v is min(deg[v], max core number of already processed neighbors)
            # Actually: core_num[v] = min(deg[v], max(core_num[u] for u in adj[v] if u processed))
            cn = deg[v]
            for u in adj[v]:
                if u in vertices[:i]:  # already processed
                    if core_num[u] < cn:
                        cn = core_num[u]
            core_num[v] = cn
            # Update degrees of unprocessed neighbors
            for u in adj[v]:
                if u in vertices[i+1:]:
                    deg[u] -= 1
        
        return core_num
    
    core_nums = compute_core_numbers()
    max_core = max(core_nums) if n > 0 else 0
    
    # Maximum clique size <= max_core + 1 (actually <= max_core for the core number definition)
    # A vertex with core number k is in the k-core, so clique size <= max_core
    
    # Use branch and bound with core-number based ordering
    # Order vertices by core number descending, then degree descending
    
    def greedy_color(candidates, adj_local):
        """Greedy coloring to get upper bound on clique size in candidates"""
        if not candidates:
            return 0
        colors = {}
        # Order candidates by degree in the induced subgraph, descending
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: -len(adj[v] & set(cand_list)))
        
        for v in cand_list:
            used = set()
            for u in cand_list:
                if u == v:
                    break
                if u in colors and u in adj[v]:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return max(colors.values()) + 1 if colors else 0
    
    def expand_clique(candidates, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        
        # Upper bound via greedy coloring
        ub = greedy_color(candidates, adj)
        if len(current) + ub <= len(best_clique):
            return
        
        # Try vertices in order of core number descending
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: (-core_nums[v], -len(adj[v])))
        
        for v in cand_list:
            if time.time() > deadline:
                return
            if v not in candidates:
                continue
            new_candidates = candidates & adj[v]
            expand_clique(new_candidates, current | {v})
            candidates = candidates - {v}
            # Update bound
            ub = greedy_color(candidates, adj)
            if len(current) + ub <= len(best_clique):
                return
    
    # Initial candidate set: all vertices
    expand_clique(set(range(n)), set())
    
    branches = sorted(best_clique) if len(best_clique) >= 2 else [P['edges'][0][0], P['edges'][0][1]]
    k = len(branches)
    
    # Now verify and route: for each pair of branches, find a path avoiding all other branches
    # But we also need all path interiors to be pairwise disjoint!
    # This is the hard part. Let's use a flow-based or iterative BFS approach.
    
    branch_set = set(branches)
    
    # For clique subdivisions, we need vertex-disjoint paths (except at endpoints)
    # Use max-flow with vertex splitting
    
    def find_disjoint_paths(branch_list):
        """Find vertex-disjoint paths between all pairs of branches in branch_list"""
        k = len(branch_list)
        if k < 2:
            return None
        
        # Vertex splitting: each vertex v becomes v_in -> v_out with capacity 1
        # Branch vertices have capacity k (or infinity) since they're endpoints
        # Non-branch vertices have capacity 1
        
        # Build flow network
        # Nodes: for each vertex v, create v_in and v_out
        # Edges:
        #   v_in -> v_out: capacity 1 for non-branch, capacity k for branch
        #   u_out -> v_in: capacity 1 for each edge (u,v) in graph
        #   Source: super source S connected to each branch_in
        #   Sink: super sink T connected from each branch_out
        
        # Actually, for pairwise disjoint paths, we need a different approach.
        # We need |C(k,2)| paths, each connecting a specific pair.
        
        # Alternative: Use iterative BFS with conflict resolution
        # Or use a multi-commodity flow approach
        
        # For now, try simple BFS for each pair and check for conflicts
        # If conflicts exist, try to resolve by re-routing
        
        paths = {}
        used_interiors = set()
        
        # Order pairs by difficulty (shortest available path first)
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branch_list[i], branch_list[j]))
        
        # Try to find paths greedily
        for (u, v) in pairs:
            forbidden = branch_set - {u, v}
            # BFS avoiding forbidden and used_interiors
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue:
                curr = queue.popleft()
                if curr == v:
                    found = True
                    break
                for w in sorted(adj[curr]):
                    if w in visited:
                        continue
                    if w in forbidden:
                        continue
                    if w in used_interiors and w != u and w != v:
                        continue
                    visited.add(w)
                    parent[w] = curr
                    queue.append(w)
            
            if not found:
                return None
            
            # Reconstruct path
            path = []
            cur = v
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            paths[(u, v)] = path
            
            # Mark interior vertices as used
            for node in path[1:-1]:
                used_interiors.add(node)
        
        return paths
    
    paths = find_disjoint_paths(branches)
    
    if paths is None:
        # Fallback: try smaller clique
        # Remove vertices one at a time and retry
        for remove_v in sorted(branches, key=lambda x: -core_nums[x]):
            if len(branches) <= 2:
                break
            new_branches = [b for b in branches if b != remove_v]
            paths = find_disjoint_paths(new_branches)
            if paths is not None:
                branches = new_branches
                break
    
    if paths is None:
        # Last resort: just use 2 branches
        branches = [P['edges'][0][0], P['edges'][0][1]]
        branch_set = set(branches)
        paths = {}
        u, v = branches[0], branches[1]
        if v in adj[u]:
            paths[(u, v)] = [u, v]
        else:
            # Find any path
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if curr == v:
                    break
                for w in adj[curr]:
                    if w not in visited:
                        visited.add(w)
                        parent[w] = curr
                        queue.append(w)
            path = []
            cur = v
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            paths[(u, v)] = path
    
    # Build output
    path_list = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            path_list.append(list(paths[(u, v)]))
    
    return {'branches': list(branches), 'paths': path_list}
# EVOLVE-BLOCK-END
