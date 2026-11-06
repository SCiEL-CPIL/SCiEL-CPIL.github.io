import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_triangle_free(n, adj):
        for u in range(n):
            neighbors_u = adj[u]
            temp = neighbors_u
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
        return True

    def has_independent_set_10(n, adj):
        if n < 10:
            return False
        
        # Greedy check first
        remaining = (1 << n) - 1
        greedy_size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = popcount(remaining & adj[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            greedy_size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            if greedy_size >= 10:
                return True
        
        # Exact search with Tomita-style coloring bound
        # We are looking for a clique of size 10 in the complement graph.
        # Complement adjacency: u~v in barG iff NOT (u~v in G)
        
        def get_color_bound(candidates):
            # Greedy coloring of barG induced by candidates
            # Color class in barG is an independent set in barG, which is a Clique in G.
            # So we partition candidates into cliques of G.
            color_bound = 0
            remaining = candidates
            while remaining:
                color_bound += 1
                clique = 0
                temp = remaining
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= ~(1 << v)
                    # Check if v is adjacent in G to all vertices in clique
                    # i.e., adj[v] & clique == clique
                    if (adj[v] & clique) == clique:
                        clique |= (1 << v)
                remaining &= ~clique
            return color_bound

        def backtrack(candidates, size):
            if size >= 10:
                return True
            
            if not candidates:
                return False
            
            # Bound 1: Size of candidates
            if size + popcount(candidates) < 10:
                return False
            
            # Bound 2: Color bound
            cb = get_color_bound(candidates)
            if size + cb < 10:
                return False
            
            # Pick vertex with max degree in candidates (in G)
            # High degree in G means low degree in barG?
            # For Clique in barG, we usually pick high degree in barG.
            # Degree in barG = (n-1) - degree in G.
            # So pick min degree in G.
            best_v = -1
            best_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = popcount(candidates & adj[v])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Option 1: Include best_v in the independent set (Clique in barG)
            # In G, this means best_v is in the set.
            # Other vertices must NOT be adjacent to best_v in G.
            # So candidates become candidates & ~adj[best_v] & ~(1 << best_v)
            new_candidates = candidates & ~(1 << best_v) & ~adj[best_v]
            if backtrack(new_candidates, size + 1):
                return True
            
            # Option 2: Exclude best_v
            candidates_without_v = candidates & ~(1 << best_v)
            if backtrack(candidates_without_v, size):
                return True
            
            return False

        return backtrack((1 << n) - 1, 0)

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                if v > u:
                    edges.append([u, v])
        return edges

    best_n = 0
    best_edges = []
    
    # Baseline: C_19
    baseline_n = 19
    baseline_edges = [[i, (i + 1) % baseline_n] for i in range(baseline_n)]
    best_n = baseline_n
    best_edges = baseline_edges

    # Try to find larger n
    for n in range(40, 19, -1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts depends on n
        # Larger n is harder, but we have a fast checker now.
        max_attempts = 5000 if n > 35 else 10000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            adj = [0] * n
            perm = list(range(n))
            random.shuffle(perm)
            
            for u in perm:
                for v in perm:
                    if v >= u:
                        continue
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if is_triangle_free(n, adj):
                if not has_independent_set_10(n, adj):
                    best_n = n
                    best_edges = edges_from_adj(n, adj)
                    break
        
        if best_n == n:
            break

    return {'n': best_n, 'edges': best_edges}
