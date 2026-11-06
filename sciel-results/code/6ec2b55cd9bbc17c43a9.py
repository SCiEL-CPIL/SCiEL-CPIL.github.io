import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Greedy lower bound: try to find a large independent set quickly
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
            
        if size >= 10:
            return True
            
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates (removes most from neighbors)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            # Branch 2: exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def make_andrasfai(k):
        """Andrasfai graph And(k): 3k vertices, edge iff diff mod 3k in {1, 3k-1}."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return n, adj

    # First, try known Andrasfai graphs for small k (valid when k >= 2, as A_1 is K3)
    for k in range(2, 10):
        n, adj = make_andrasfai(k)
        if is_triangle_free(adj, n) and not has_independent_set_10(adj, n):
            return {'n': n, 'edges': build_edge_list(adj, n)}

    # Now search for larger n using random triangle-free graphs
    # Start from larger n and work down
    best_result = {'n': 1, 'edges': []}
    
    for target_n in range(40, 20, -1):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random triangle-free graph
            adj = [0] * target_n
            edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(edges)
            
            for u, v in edges:
                common = adj[u] & adj[v]
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if valid
            if is_triangle_free(adj, target_n) and not has_independent_set_10(adj, target_n):
                result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                return result
    
    # Fallback to smallest known valid graph (Andrasfai k=2 is n=6, C6-like structure, actually A2 is 6-cycle? No, A2 is 6 vertices. Let's just use a simple valid small graph if needed, but A2 is triangle free)
    n, adj = make_andrasfai(2)
    if is_triangle_free(adj, n) and not has_independent_set_10(adj, n):
        return {'n': n, 'edges': build_edge_list(adj, n)}
    
    # Absolute fallback: Single vertex is always valid
    return {'n': 1, 'edges': []}
