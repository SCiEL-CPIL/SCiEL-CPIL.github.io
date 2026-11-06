import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

        # Exact branch and bound
        sys.setrecursionlimit(10000)

        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False

            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False

            # Pick vertex with max degree in candidates (removes most)
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

    def random_triangle_free(n):
        """Generate a random triangle-free graph on n vertices."""
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            # Check if adding (u, v) creates a triangle
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, n, deadline):
        """Try to reduce independence number below 10 by adding edges."""
        current_adj = adj[:]
        
        while time.time() < deadline:
            # Check if we already have alpha < 10
            if not has_independent_set_10(current_adj, n):
                return current_adj
            
            # Find a large independent set to target for edge addition
            candidates = (1 << n) - 1
            indep_set = []
            while candidates:
                best_v = -1
                min_deg = n + 1
                temp = candidates
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    deg = (current_adj[v] & candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v == -1:
                    break
                indep_set.append(best_v)
                candidates &= ~current_adj[best_v] & ~(1 << best_v)
                
            if len(indep_set) < 10:
                return current_adj

            # Try to add an edge between two vertices in indep_set
            added_edge = False
            random.shuffle(indep_set)
            
            for i in range(len(indep_set)):
                for j in range(i + 1, len(indep_set)):
                    u = indep_set[i]
                    v = indep_set[j]
                    
                    # Check if adding (u, v) creates a triangle
                    if (current_adj[u] & current_adj[v]) == 0:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        added_edge = True
                        break
                if added_edge:
                    break
                    
            if not added_edge:
                break
                
        return current_adj

    # Main search loop
    for n in range(40, 38, -1):
        if time.time() > deadline:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            adj = random_triangle_free(n)
            
            if not has_independent_set_10(adj, n):
                return {'n': n, 'edges': build_edge_list(adj, n)}
                
            improved_adj = local_search(adj, n, deadline - 1.0)
            
            if not has_independent_set_10(improved_adj, n):
                return {'n': n, 'edges': build_edge_list(improved_adj, n)}
                
    # Fallback to And(9)
    def make_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n

    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        return {'n': n9, 'edges': build_edge_list(adj9, n9)}

    return {'n': 1, 'edges': []}
