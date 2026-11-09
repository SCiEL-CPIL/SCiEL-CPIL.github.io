import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # Fallback: Andrásfai graph And(9) on 26 vertices
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_independent_set_10(adj, n):
        all_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            if mask.bit_count() < 10 - size:
                return False
            
            # Try to find a vertex with fewest non-neighbors in mask
            min_count = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            new_mask_include = non_neighbors[best_v] & mask
            if dfs(new_mask_include, size + 1):
                return True
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                return True
            return False
            
        return dfs(all_mask, 0)

    def count_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    count += (adj[i] & adj[j] & ((1 << n) - 1) & ~((1 << (i+1)) - 1)).bit_count()
        return count

    def try_cayley(n, max_attempts):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            S = set()
            candidates = list(range(1, n // 2 + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if x == n - x:
                    pair = {x}
                else:
                    pair = {x, n - x}
                
                valid = True
                if len(pair) == 2:
                    p1, p2 = list(pair)
                    if (p1 + p2) % n in pair:
                        valid = False
                    if (p1 + p1) % n in pair:
                        valid = False
                    if (p2 + p2) % n in pair:
                        valid = False
                
                if valid:
                    for s in S:
                        for p in pair:
                            sum_val = (s + p) % n
                            if sum_val in S or sum_val in pair:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    S.update(pair)
            
            if not S:
                continue
                
            adj = [0] * n
            for u in range(n):
                for v in S:
                    neighbor = (u + v) % n
                    adj[u] |= (1 << neighbor)
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    def try_random_greedy(n, max_attempts):
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not has_independent_set_10(adj, n):
                return adj
        return None

    def try_dense_prune(n, max_attempts):
        """
        Start with a dense graph (complement of a random regular graph),
        then iteratively remove edges to destroy triangles while monitoring
        independence number.
        """
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Start with a random graph with high edge density
            # Use complement of a random d-regular graph
            # For n vertices, try to make a random graph with ~n/2 edges per vertex
            adj = [0] * n
            
            # Generate a random graph by including each edge with probability p
            # Choose p to get roughly n/2 degree
            p = 0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Now iteratively remove edges to destroy triangles
            # While triangles exist, find one and remove an edge from it
            # But be careful: removing edges can increase independence number
            # So we remove edges that are in many triangles (high "triangle centrality")
            
            max_iterations = 500
            for _ in range(max_iterations):
                if time.time() > deadline:
                    return None
                
                # Find all triangles
                triangles = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            common = adj[i] & adj[j] & ((1 << n) - 1)
                            # Only consider k > j to avoid duplicates
                            common &= ~((1 << (j + 1)) - 1)
                            while common:
                                lsb = common & -common
                                k = lsb.bit_length() - 1
                                triangles.append((i, j, k))
                                common ^= lsb
                
                if not triangles:
                    break
                
                # Count how many triangles each edge is in
                edge_triangle_count = {}
                for (i, j, k) in triangles:
                    for (a, b) in [(i, j), (i, k), (j, k)]:
                        key = (min(a, b), max(a, b))
                        edge_triangle_count[key] = edge_triangle_count.get(key, 0) + 1
                
                # Remove the edge that is in the most triangles
                if edge_triangle_count:
                    best_edge = max(edge_triangle_count, key=edge_triangle_count.get)
                    u, v = best_edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    break
            
            # After removing triangles, check if we have a valid graph
            if not has_independent_set_10(adj, n):
                return adj
        return None

    # Main search loop
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # Allocate attempts based on n
        if n >= 39:
            cayley_attempts = 300
            greedy_attempts = 300
            dense_attempts = 200
        elif n >= 37:
            cayley_attempts = 150
            greedy_attempts = 150
            dense_attempts = 100
        else:
            cayley_attempts = 50
            greedy_attempts = 50
            dense_attempts = 50
            
        # Strategy 1: Cayley Graphs
        adj = try_cayley(n, cayley_attempts)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
        # Strategy 2: Random Greedy
        adj = try_random_greedy(n, greedy_attempts)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
        # Strategy 3: Dense Start, Prune Triangles
        adj = try_dense_prune(n, dense_attempts)
        if adj is not None:
            edges = edges_from_adj(adj, n)
            return {'n': n, 'edges': edges}
            
    return best_result
