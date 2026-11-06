import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def is_triangle_free(n, adj):
        for u in range(n):
            for v in range(n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def greedy_alpha_lower_bound(n, adj):
        """Return size of a greedy independent set (lower bound on alpha)."""
        remaining = (1 << n) - 1
        size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(remaining & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return size

    def has_independent_set_10(n, adj):
        """Exact branch and bound for independent set of size 10."""
        # Quick greedy check
        if greedy_alpha_lower_bound(n, adj) >= 10:
            return True

        def backtrack(candidates, size):
            if size == 10:
                return True
            if size + bin(candidates).count('1') < 10:
                return False

            # Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(candidates & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            candidates &= ~(1 << best_v)

            # Option 1: Include best_v
            new_candidates = candidates & ~adj[best_v]
            if backtrack(new_candidates, size + 1):
                return True

            # Option 2: Exclude best_v
            if backtrack(candidates, size):
                return True

            return False

        return backtrack((1 << n) - 1, 0)

    def validate(n, adj):
        if not is_triangle_free(n, adj):
            return False
        if has_independent_set_10(n, adj):
            return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Fallback: known valid graph on 26 vertices
    best_n = 26
    best_adj = [0] * 26
    for i in range(26):
        for j in range(i + 1, 26):
            if (j - i) % 26 % 3 == 1:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    best_edges = edges_from_adj(26, best_adj)

    def is_sum_free_circulant(n, S):
        """Check if circulant graph C_n(S) is triangle-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a != b:
                    if (b - a) % n in S_set:
                        return False
        return True

    def build_circulant(n, S):
        """Build adjacency list for circulant graph C_n(S)."""
        adj = [0] * n
        S_set = set(S)
        for i in range(n):
            for s in S:
                adj[i] |= (1 << ((i + s) % n))
                adj[i] |= (1 << ((i - s) % n))
            adj[i] &= ~(1 << i)
        return adj

    def search_circulants(n):
        """Search for a valid circulant graph on n vertices."""
        if time.time() > deadline - 5:
            return None

        max_s = n // 2
        candidates = list(range(1, max_s + 1))
        
        # Try different sizes of S
        # For n=40, |S| ~ 6-8 is good density
        # We want dense graphs to minimize alpha
        for size in range(max_s // 2, 2, -1):
            if time.time() > deadline - 2:
                return None
            
            # To avoid combinatorial explosion, we sample if combinations are too many
            # C(20, 8) is 125970, which is fine. C(20, 10) is 184756, also fine.
            # But we should prioritize larger S first for density.
            
            # Generate combinations
            # If n is large and size is large, this might be slow.
            # For n=40, max_s=20.
            # Let's limit attempts if it takes too long per n.
            start_combo_time = time.time()
            count = 0
            for S_comb in itertools.combinations(candidates, size):
                count += 1
                if count % 1000 == 0 and time.time() > deadline - 2:
                    return None
                if time.time() - start_combo_time > 10: # 10s per size
                    break
                
                S = set(S_comb)
                # Make symmetric
                S_sym = set()
                for s in S:
                    S_sym.add(s)
                    S_sym.add(n - s)
                
                # Check triangle-free
                if not is_sum_free_circulant(n, S_sym):
                    continue
                
                # Quick density check
                edge_count = n * len(S_sym) // 2
                if edge_count < n * n // 10:
                    continue
                
                # Build and validate
                adj = build_circulant(n, S_sym)
                if greedy_alpha_lower_bound(n, adj) >= 10:
                    continue
                
                if validate(n, adj):
                    edges = edges_from_adj(n, adj)
                    return {'n': n, 'edges': edges}
        
        # Try random symmetric sets if systematic failed or was too slow
        for attempt in range(200):
            if time.time() > deadline - 2:
                return None
            
            S = set()
            for s in range(1, n // 2 + 1):
                if random.random() < 0.3:
                    S.add(s)
                    S.add(n - s)
            
            if len(S) < 4:
                continue
            
            if not is_sum_free_circulant(n, S):
                continue
            
            edge_count = n * len(S) // 2
            if edge_count < n * n // 10:
                continue
            
            adj = build_circulant(n, S)
            if greedy_alpha_lower_bound(n, adj) >= 10:
                continue
            
            if validate(n, adj):
                edges = edges_from_adj(n, adj)
                return {'n': n, 'edges': edges}
        
        return None

    def search_random(n):
        """Random graph generation with local search repair."""
        if time.time() > deadline - 5:
            return None

        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                return None

            # Generate random maximal triangle-free graph
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Quick density check
            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count < n * n // 10:
                continue

            # Try local search repair
            current_adj = adj[:]
            
            # Limit repair steps
            max_repair_steps = 50
            for _ in range(max_repair_steps):
                if time.time() > deadline - 2:
                    return None
                
                # Find independent set of size 10
                # We use has_independent_set_10 which returns True/False
                # We need the set to repair. 
                # Let's use a helper that returns the set if found.
                
                def find_iso_set(n, adj):
                    def backtrack(candidates, size, current_set):
                        if size == 10:
                            return current_set
                        if size + bin(candidates).count('1') < 10:
                            return None

                        best_v = -1
                        best_deg = -1
                        temp = candidates
                        while temp:
                            v = (temp & -temp).bit_length() - 1
                            temp &= ~(1 << v)
                            deg = bin(candidates & adj[v]).count('1')
                            if deg > best_deg:
                                best_deg = deg
                                best_v = v

                        if best_v == -1:
                            return None

                        candidates &= ~(1 << best_v)

                        new_candidates = candidates & ~adj[best_v]
                        result = backtrack(new_candidates, size + 1, current_set | (1 << best_v))
                        if result is not None:
                            return result

                        result = backtrack(candidates, size, current_set)
                        if result is not None:
                            return result

                        return None

                    return backtrack((1 << n) - 1, 0, 0)

                iso_set = find_iso_set(n, current_adj)
                if iso_set is None:
                    # Success!
                    edges = edges_from_adj(n, current_adj)
                    return {'n': n, 'edges': edges}
                
                # Try to add an edge within the independent set
                verts = []
                temp = iso_set
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= ~(1 << v)
                    verts.append(v)
                
                edge_added = False
                random.shuffle(verts)
                
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u = verts[i]
                        v = verts[j]
                        if (current_adj[u] & (1 << v)) == 0:
                            if (current_adj[u] & current_adj[v]) == 0:
                                current_adj[u] |= (1 << v)
                                current_adj[v] |= (1 << u)
                                edge_added = True
                                break
                    if edge_added:
                        break
                
                if not edge_added:
                    break
            
        return None

    # Main search loop
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break
        
        # Try Circulants first (structured, likely better alpha)
        result = search_circulants(n)
        if result:
            return result
        
        # Try Random + Repair
        result = search_random(n)
        if result:
            return result

    return {'n': best_n, 'edges': best_edges}
