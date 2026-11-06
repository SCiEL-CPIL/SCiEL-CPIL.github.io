import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def compute_alpha(adj_mask, n):
        """Compute maximum independent set size using Branch and Bound."""
        best = [0]
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            # Upper bound
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Pick a vertex to branch on: highest degree in candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def has_independent_set(adj_mask, n, target=11):
        """Check if there is an independent set of size >= target."""
        sys.setrecursionlimit(10000)
        
        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False
                
            # Pick vertex with max degree in candidate subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                return True
                
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)

    def can_add_edge(adj_mask, u, v, n):
        """Check if adding edge (u,v) preserves C4-free property."""
        nu = adj_mask[u]
        nv = adj_mask[v]
        
        # Check if any neighbor of u is adjacent to any neighbor of v
        # We iterate over the smaller neighborhood
        if nu.bit_count() < nv.bit_count():
            c = nu
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nv:
                    return False
        else:
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nu:
                    return False
        
        return True

    def generate_greedy_c4_free(n, seed):
        """Generate a random C4-free graph using greedy edge addition."""
        random.seed(seed)
        adj_mask = [0] * n
        
        # Generate all possible edges and shuffle
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj_mask, u, v, n):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        return adj_mask

    def local_search_reduce_alpha(adj_mask, n, time_limit):
        """
        Local search to add edges to reduce independence number.
        Strategy: Try adding non-edges that preserve C4-free and check if alpha decreases.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Generate list of non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj_mask[u] & (1 << v)):
                        non_edges.append((u, v))
            
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                    
                # Check if we can add this edge
                if not can_add_edge(adj_mask, u, v, n):
                    continue
                
                # Save state
                old_u = adj_mask[u]
                old_v = adj_mask[v]
                
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
                # Check if alpha < 11
                if not has_independent_set(adj_mask, n, 11):
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[u] = old_u
                    adj_mask[v] = old_v
        
        return adj_mask

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses random greedy construction followed by local search.
        """
        if time.time() > deadline:
            return None
            
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > deadline:
                return None
                
            adj_mask = generate_greedy_c4_free(n, seed)
            
            # Check if alpha < 11 already
            if not has_independent_set(adj_mask, n, 11):
                return adj_mask
            
            # Try local search to reduce alpha
            remaining_time = deadline - time.time()
            if remaining_time > 2:
                local_search_reduce_alpha(adj_mask, n, time.time() + min(5, remaining_time))
                if not has_independent_set(adj_mask, n, 11):
                    return adj_mask
        
        return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Start from high n and go down
    for n in range(43, 30, -1):
        if time.time() > deadline - 5:
            break
            
        adj_mask = try_find_graph(n, 10)
        if adj_mask:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_mask[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
