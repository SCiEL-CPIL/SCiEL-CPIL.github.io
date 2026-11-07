"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute adjacency sets for faster lookup
    adj_set = {u: set(adj[u]) for u in range(n)}
    
    def check_bicolored_cycle_incremental(colors, c1, c2, changed_vert=None):
        """Check if subgraph induced by colors c1,c2 has a cycle using union-find."""
        verts = [i for i in range(n) if colors[i] in (c1, c2)]
        if len(verts) < 3:
            return False
        parent = {v: v for v in verts}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        vert_set = set(verts)
        for u in verts:
            for v in adj[u]:
                if v in vert_set and u < v:
                    if not union(u, v):
                        return True
        return False
    
    def is_valid_full(colors):
        """Check if coloring is fully valid: proper + no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored cycles for all pairs of colors used
        used_colors = set(colors)
        color_list = sorted(used_colors)
        k = len(color_list)
        for i in range(k):
            for j in range(i+1, k):
                if check_bicolored_cycle_incremental(colors, color_list[i], color_list[j]):
                    return False
        return True
    
    def count_violations(colors):
        """Count total violations: improper edges + bicolored cycles."""
        cnt = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                cnt += 1
        used_colors = set(colors)
        color_list = sorted(used_colors)
        k = len(color_list)
        for i in range(k):
            for j in range(i+1, k):
                if check_bicolored_cycle_incremental(colors, color_list[i], color_list[j]):
                    cnt += 1
        return cnt
    
    # Backtracking search with forward checking
    # Try to find a valid coloring with k colors, starting from small k
    
    import random
    rng = random.Random(42)
    
    # Vertex ordering: use degree-based heuristic (high degree first)
    degree = [len(adj[u]) for u in range(n)]
    vertex_order = sorted(range(n), key=lambda u: -degree[u])
    
    best_colors = list(range(n))  # Fallback: all distinct
    best_k = n
    
    def backtrack_search(k, max_time=60.0):
        """Backtracking search for valid k-coloring."""
        colors = [-1] * n
        start_time = time.time()
        nodes_explored = 0
        
        # For each color pair, track union-find state incrementally
        # Key: (c1, c2) where c1 < c2, value: parent dict for union-find
        uf_states = {}
        
        def get_uf_state(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            key = (c1, c2)
            if key not in uf_states:
                uf_states[key] = {}
            return uf_states[key]
        
        def uf_find(parent_dict, x):
            while parent_dict[x] != x:
                parent_dict[x] = parent_dict[parent_dict[x]]
                x = parent_dict[x]
            return x
        
        def uf_union(parent_dict, x, y):
            rx, ry = uf_find(parent_dict, x), uf_find(parent_dict, y)
            if rx == ry:
                return False
            parent_dict[rx] = ry
            return True
        
        def would_create_cycle(c1, c2, u, v):
            """Check if adding edge (u,v) to bicolored subgraph (c1,c2) creates a cycle."""
            if c1 == c2:
                return False
            if c1 > c2:
                c1, c2 = c2, c1
            key = (c1, c2)
            state = uf_states.get(key)
            if state is None:
                return False
            # u and v must both have colors in {c1, c2}
            if colors[u] not in (c1, c2) or colors[v] not in (c1, c2):
                return False
            if colors[u] == colors[v]:
                return False  # Same color, no bicolored edge
            # Check if u and v are already connected in the bicolored subgraph
            if uf_find(state, u) == uf_find(state, v):
                return True
            return False
        
        def assign_color(v, c):
            """Assign color c to vertex v and update union-find states."""
            old_c = colors[v]
            colors[v] = c
            
            # Update union-find for all color pairs involving c
            # For each neighbor u of v:
            for u in adj[v]:
                uc = colors[u]
                if uc == c:
                    continue  # Same color, no bicolored edge
                
                # Edge (u,v) connects colors uc and c
                c1, c2 = min(uc, c), max(uc, c)
                key = (c1, c2)
                
                if key not in uf_states:
                    uf_states[key] = {}
                state = uf_states[key]
                
                # Add u and v to this state if not already
                if u not in state:
                    state[u] = u
                if v not in state:
                    state[v] = v
                
                # Union u and v
                uf_union(state, u, v)
            
            # Also need to handle: when we change v's color, we might need to
            # remove old edges and add new ones. This is complex.
            # Instead, let's use a simpler approach: full recheck for small k.
        
        # Simpler approach: use full validation at each step but with pruning
        def can_assign(v, c, colors_so_far):
            """Check if assigning color c to vertex v is consistent."""
            # Check proper coloring: no neighbor has color c
            for u in adj[v]:
                if colors_so_far[u] == c:
                    return False
            
            # Check bicolored cycle: for each color d != c used by neighbors,
            # check if adding (v, neighbor_with_color_d) would create a cycle
            # in the bicolored subgraph (c, d)
            for u in adj[v]:
                d = colors_so_far[u]
                if d == -1 or d == c:
                    continue
                # Check if v and u are connected in the bicolored subgraph (c,d)
                # excluding the edge (u,v) itself
                # We need to check if there's a path from v to u using only
                # vertices with colors in {c, d}
                
                # BFS from v to u avoiding direct edge (u,v)
                visited = set()
                queue = [v]
                visited.add(v)
                found = False
                while queue:
                    cur = queue.pop(0)
                    if cur == u:
                        found = True
                        break
                    for w in adj[cur]:
                        if w in visited:
                            continue
                        if colors_so_far[w] not in (c, d):
                            continue
                        # Avoid the direct edge (u,v)
                        if (cur == v and w == u) or (cur == u and w == v):
                            continue
                        visited.add(w)
                        queue.append(w)
                if found:
                    return False
            return True
        
        def search(idx):
            nonlocal nodes_explored
            if time.time() - start_time > max_time:
                return None
            
            if idx == n:
                return colors[:]
            
            v = vertex_order[idx]
            
            # Try colors 0 to k-1
            for c in range(k):
                nodes_explored += 1
                if can_assign(v, c, colors):
                    colors[v] = c
                    result = search(idx + 1)
                    if result is not None:
                        return result
                    colors[v] = -1
            
            return None
        
        result = search(0)
        return result
    
    # Try increasing k until we find a valid coloring
    for k in range(2, min(n, 10)):
        if time.time() > deadline - 5:
            break
        result = backtrack_search(k, max_time=min(30.0, deadline - time.time() - 2))
        if result is not None:
            # Verify the result
            if is_valid_full(result):
                best_colors = result
                best_k = k
                break
    
    # If backtracking didn't find a solution, try local search from a good start
    if best_k == n:
        # Try local search to improve
        import random
        rng2 = random.Random(123)
        
        # Start with a random k-coloring and try to fix
        for k in range(3, 8):
            if time.time() > deadline - 5:
                break
            colors = [rng2.randint(0, k-1) for _ in range(n)]
            
            # Local search: minimize violations
            for iteration in range(20000):
                if time.time() > deadline - 3:
                    break
                
                v = rng2.randint(0, n-1)
                old_c = colors[v]
                
                # Try all colors
                best_c = old_c
                best_violations = count_violations(colors)
                
                for c in range(k):
                    if c == old_c:
                        continue
                    colors[v] = c
                    v_new = count_violations(colors)
                    if v_new < best_violations:
                        best_violations = v_new
                        best_c = c
                    colors[v] = old_c
                
                if best_c != old_c:
                    colors[v] = best_c
            
            if count_violations(colors) == 0:
                distinct = len(set(colors))
                if distinct < best_k:
                    best_colors = colors
                    best_k = distinct
    
    # Final verification and normalization
    if is_valid_full(best_colors):
        # Normalize colors to 0..k-1
        color_map = {}
        norm_colors = []
        for c in best_colors:
            if c not in color_map:
                color_map[c] = len(color_map)
            norm_colors.append(color_map[c])
        return [int(c) for c in norm_colors]
    else:
        # Fallback: all distinct
        return list(range(n))
# EVOLVE-BLOCK-END
