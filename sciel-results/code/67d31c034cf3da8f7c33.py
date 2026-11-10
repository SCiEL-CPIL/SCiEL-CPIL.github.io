"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    from collections import deque, defaultdict
    import sys
    
    deadline = time.time() + 85
    n = P['n']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Compute core numbers for ordering
    def compute_core_numbers():
        core_num = [0] * n
        deg = [len(adj[i]) for i in range(n)]
        vertices = list(range(n))
        vertices.sort(key=lambda v: -deg[v])
        
        processed = set()
        for i, v in enumerate(vertices):
            cn = deg[v]
            for u in adj[v]:
                if u in processed:
                    if core_num[u] < cn:
                        cn = core_num[u]
            core_num[v] = cn
            processed.add(v)
            for u in adj[v]:
                if u not in processed:
                    deg[u] -= 1
        return core_num
    
    core_nums = compute_core_numbers()
    
    # Flow-based feasibility check for a set of branch vertices
    # Uses vertex-splitting and max-flow to check if C(k,2) vertex-disjoint paths exist
    def check_feasible_and_route(branch_list):
        """
        Check if branch_list admits vertex-disjoint paths between all pairs.
        Returns dict of paths if feasible, None otherwise.
        Uses iterative BFS with backtracking on conflicts.
        """
        k = len(branch_list)
        if k < 2:
            return None
        
        branch_set = set(branch_list)
        
        # Generate all pairs
        pairs = []
        for i in range(k):
            for j in range(i+1, k):
                pairs.append((branch_list[i], branch_list[j]))
        
        # Try multiple orderings of pairs to find a valid assignment
        import itertools
        
        # Sort pairs by "difficulty" - estimate path length
        def pair_difficulty(u, v):
            # BFS to find shortest path length
            if v in adj[u]:
                return 1
            visited = {u}
            queue = deque([(u, 0)])
            while queue:
                curr, dist = queue.popleft()
                for w in adj[curr]:
                    if w == v:
                        return dist + 1
                    if w not in visited and w not in branch_set:
                        visited.add(w)
                        queue.append((w, dist + 1))
            return float('inf')
        
        # Try several orderings
        base_order = sorted(pairs, key=lambda p: pair_difficulty(p[0], p[1]))
        
        # Also try reverse and other heuristics
        orderings = [
            base_order,
            list(reversed(base_order)),
            sorted(pairs, key=lambda p: (-pair_difficulty(p[0], p[1]), p[0], p[1])),
        ]
        
        for ordering in orderings:
            paths = {}
            used_interiors = set()
            success = True
            
            for (u, v) in ordering:
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
                        if w in used_interiors:
                            continue
                        visited.add(w)
                        parent[w] = curr
                        queue.append(w)
                
                if not found:
                    success = False
                    break
                
                # Reconstruct path
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                paths[(u, v)] = path
                
                # Mark interior vertices
                for node in path[1:-1]:
                    used_interiors.add(node)
            
            if success:
                return paths
        
        return None
    
    # Strategy: Find maximum clique, then check feasibility.
    # If not feasible, try removing vertices (in order of least "connectivity")
    # Also try adding non-clique vertices that are well-connected.
    
    # First, find maximum clique via branch-and-bound
    best_clique = []
    
    def greedy_color(candidates):
        if not candidates:
            return 0
        colors = {}
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: -len(adj[v] & candidates))
        
        for v in cand_list:
            used = set()
            for u in cand_list:
                if u != v and u in colors and u in adj[v]:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return max(colors.values()) + 1 if colors else 0
    
    def expand_clique(candidates, current, depth=0):
        nonlocal best_clique
        if time.time() > deadline - 5:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        
        ub = greedy_color(candidates)
        if len(current) + ub <= len(best_clique):
            return
        
        cand_list = list(candidates)
        cand_list.sort(key=lambda v: (-core_nums[v], -len(adj[v])))
        
        for v in cand_list:
            if time.time() > deadline - 5:
                return
            if v not in candidates:
                continue
            new_candidates = candidates & adj[v]
            expand_clique(new_candidates, current | {v}, depth + 1)
            candidates = candidates - {v}
            ub = greedy_color(candidates)
            if len(current) + ub <= len(best_clique):
                return
    
    expand_clique(set(range(n)), set())
    
    if len(best_clique) < 2:
        best_clique = [P['edges'][0][0], P['edges'][0][1]]
    
    branches = sorted(best_clique)
    
    # Check feasibility
    paths = check_feasible_and_route(branches)
    
    if paths is None:
        # Try removing vertices one at a time
        # Remove the vertex with fewest "alternative routes" (low degree in non-branch subgraph)
        removed_order = sorted(branches, key=lambda v: len(adj[v] - set(branches)))
        
        for remove_v in removed_order:
            if len(branches) <= 2:
                break
            new_branches = [b for b in branches if b != remove_v]
            paths = check_feasible_and_route(new_branches)
            if paths is not None:
                branches = new_branches
                break
    
    if paths is None:
        # Try a different approach: greedy construction
        # Start with 2 branches, try to add more
        branches = [P['edges'][0][0], P['edges'][0][1]]
        paths = check_feasible_and_route(branches)
        
        if paths is not None:
            # Try to add vertices
            candidates = [v for v in range(n) if v not in branches]
            candidates.sort(key=lambda v: (-core_nums[v], -len(adj[v])))
            
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                new_branches = branches + [v]
                new_paths = check_feasible_and_route(new_branches)
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
    
    if paths is None:
        # Last resort: 2 branches with direct edge
        for u, v in P['edges']:
            if v in adj[u]:
                branches = [u, v]
                paths = {(u, v): [u, v]}
                break
    
    # Build output
    path_list = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            path_list.append(list(paths[(u, v)]))
    
    return {'branches': [int(b) for b in branches], 'paths': [list(p) for p in path_list]}
# EVOLVE-BLOCK-END
