import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n = 16

    # Build Grötzsch graph (Mycielski of C5) on vertices 0..10
    def build_grotzsch():
        c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        edges = list(c5_edges)
        for u, v in c5_edges:
            edges.append((u, 5 + v))
            edges.append((v, 5 + u))
        for i in range(5):
            edges.append((10, 5 + i))
        return edges

    core_edges = build_grotzsch()
    core_n = 11
    new_vertices = list(range(core_n, n))
    num_new = 5

    # Build core adjacency
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)

    # Enumerate all independent sets in Grötzsch
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind and mask != 0:
            independent_sets.append(mask)

    independent_sets.sort(key=lambda x: -bin(x).count("1"))

    num_sets = len(independent_sets)
    disjoint = [[False] * num_sets for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(i, num_sets):
            d = (independent_sets[i] & independent_sets[j]) == 0
            disjoint[i][j] = d
            disjoint[j][i] = d

    set_sizes = [bin(s).count("1") for s in independent_sets]

    def is_3_colorable(adj_full):
        degrees = [bin(adj_full[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = [False] * 3
            for u in range(n):
                if color[u] != -1 and (adj_full[v] & (1 << u)):
                    used[color[u]] = True
            
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def count_edges(adj_full):
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_full[i] & (1 << j):
                    cnt += 1
        return cnt

    def is_triangle_free(adj_full):
        for u in range(n):
            for v in range(u + 1, n):
                if adj_full[u] & (1 << v):
                    if adj_full[u] & adj_full[v]:
                        return False
        return True

    # Strategy 1: Branch and bound over independent set assignments
    def evaluate_assignment(assignment):
        adj_full = [0] * n
        for u, v in core_edges:
            adj_full[u] |= (1 << v)
            adj_full[v] |= (1 << u)
        
        for i in range(num_new):
            v_new = new_vertices[i]
            s_mask = independent_sets[assignment[i]]
            for u in range(core_n):
                if s_mask & (1 << u):
                    adj_full[v_new] |= (1 << u)
                    adj_full[u] |= (1 << v_new)
        
        possible_new_edges = []
        for i in range(num_new):
            for j in range(i + 1, num_new):
                if disjoint[assignment[i]][assignment[j]]:
                    possible_new_edges.append((i, j))
        
        m = len(possible_new_edges)
        best_mask = 0
        max_cnt = 0
        for mask in range(1 << m):
            adj_new = [0] * num_new
            for idx in range(m):
                if mask & (1 << idx):
                    u, v = possible_new_edges[idx]
                    adj_new[u] |= (1 << v)
                    adj_new[v] |= (1 << u)
            tf = True
            for u in range(num_new):
                for v in range(u + 1, num_new):
                    if adj_new[u] & (1 << v):
                        if adj_new[u] & adj_new[v]:
                            tf = False
                            break
                if not tf:
                    break
            if tf:
                cnt = bin(mask).count("1")
                if cnt > max_cnt:
                    max_cnt = cnt
                    best_mask = mask
        
        for idx in range(m):
            if best_mask & (1 << idx):
                u, v = possible_new_edges[idx]
                adj_full[new_vertices[u]] |= (1 << new_vertices[v])
                adj_full[new_vertices[v]] |= (1 << new_vertices[u])
        
        if is_3_colorable(adj_full):
            return -1, adj_full
        
        return count_edges(adj_full), adj_full

    K = min(30, num_sets)
    top_sets = list(range(K))
    
    best_score = -1
    best_assignment = None

    def dfs(k, assignment_prefix, current_core_edges):
        nonlocal best_score, best_assignment
        if time.time() > deadline:
            return
        
        if k == num_new:
            score, adj_full = evaluate_assignment(assignment_prefix)
            if score > best_score:
                best_score = score
                best_assignment = list(assignment_prefix)
            return
        
        max_remaining_core = 0
        for r in range(k, num_new):
            max_remaining_core += max(set_sizes[s] for s in top_sets)
        max_possible = current_core_edges + max_remaining_core + 10
        
        if max_possible <= best_score:
            return
        
        for s_idx in top_sets:
            new_core_edges = current_core_edges + set_sizes[s_idx]
            dfs(k + 1, assignment_prefix + [s_idx], new_core_edges)

    dfs(0, [], 0)

    # Reconstruct from best_assignment if found
    best_edges = None
    if best_assignment is not None and best_score > 0:
        final_edges = list(core_edges)
        for i in range(num_new):
            v_new = new_vertices[i]
            s_mask = independent_sets[best_assignment[i]]
            for u in range(core_n):
                if s_mask & (1 << u):
                    final_edges.append([v_new, u])

        possible_new_edges = []
        for i in range(num_new):
            for j in range(i + 1, num_new):
                if disjoint[best_assignment[i]][best_assignment[j]]:
                    possible_new_edges.append((i, j))

        m = len(possible_new_edges)
        max_new_edges = 0
        best_mask = 0
        for mask in range(1 << m):
            adj_subset = [0] * num_new
            for idx in range(m):
                if mask & (1 << idx):
                    u, v = possible_new_edges[idx]
                    adj_subset[u] |= (1 << v)
                    adj_subset[v] |= (1 << u)

            tf = True
            for u in range(num_new):
                for v in range(u + 1, num_new):
                    if adj_subset[u] & (1 << v):
                        if adj_subset[u] & adj_subset[v]:
                            tf = False
                            break
            if not tf:
                break

            if tf:
                cnt = bin(mask).count("1")
                if cnt > max_new_edges:
                    max_new_edges = cnt
                    best_mask = mask

        for idx in range(m):
            if best_mask & (1 << idx):
                u, v = possible_new_edges[idx]
                final_edges.append([new_vertices[u], new_vertices[v]])

        best_edges = final_edges
        best_score = len(best_edges)

    # Strategy 2: Randomized greedy edge completion with restarts
    # Start from Grötzsch + random edges, greedily add edges while maintaining
    # triangle-freeness and 4-chromaticity
    
    def greedy_complete(adj_full, max_time=5.0):
        """Greedily add edges to maximize edge count while keeping triangle-free and not 3-colorable."""
        t_start = time.time()
        adj = list(adj_full)
        
        # All potential edges not yet present
        all_potential = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    all_potential.append((i, j))
        
        random.shuffle(all_potential)
        
        improved = True
        while improved and time.time() - t_start < max_time:
            improved = False
            # Try adding each potential edge in random order
            random.shuffle(all_potential)
            for u, v in all_potential:
                if time.time() - t_start > max_time:
                    break
                if adj[u] & (1 << v):
                    continue
                # Check if adding (u,v) creates a triangle
                common = adj[u] & adj[v]
                if common:
                    continue
                # Try adding edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                # Check 3-colorability
                if not is_3_colorable(adj):
                    improved = True
                    # Remove from potential list
                    all_potential = [(a, b) for a, b in all_potential if not ((a == u and b == v) or (a == v and b == u))]
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return adj

    # Multiple restarts with different initial configurations
    best_greedy_edges = None
    best_greedy_score = best_score if best_score > 0 else 0
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create initial adjacency: Grötzsch core + random connections to new vertices
        adj_init = [0] * n
        for u, v in core_edges:
            adj_init[u] |= (1 << v)
            adj_init[v] |= (1 << u)
        
        # Randomly connect new vertices to core (each new vertex gets random independent set)
        for i in range(num_new):
            v_new = new_vertices[i]
            # Pick a random independent set
            s_idx = random.randint(0, num_sets - 1)
            s_mask = independent_sets[s_idx]
            for u in range(core_n):
                if s_mask & (1 << u):
                    adj_init[v_new] |= (1 << u)
                    adj_init[u] |= (1 << v_new)
        
        # Greedily complete
        adj_result = greedy_complete(adj_init, max_time=2.0)
        score = count_edges(adj_result)
        
        if score > best_greedy_score:
            best_greedy_score = score
            # Convert to edge list
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_result[i] & (1 << j):
                        edge_list.append([i, j])
            best_greedy_edges = edge_list
    
    # Choose the best result
    if best_greedy_edges is not None and best_greedy_score > (best_score if best_score > 0 else 0):
        return best_greedy_edges
    elif best_edges is not None:
        return best_edges
    else:
        return core_edges
