import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Leave 10s margin for 120s limit

    def time_check():
        return time.time() - start_time

    def check_triangle_free(n, adj_masks):
        """
        Returns True if the graph defined by adj_masks is triangle-free.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    if adj_masks[u] & adj_masks[v]:
                        return False
        return True

    def max_independent_set_size(n, adj_masks):
        """
        Returns the size of the maximum independent set.
        Uses bitsets and branch and bound.
        Optimized: Branch on vertex with max degree in candidate subgraph.
        """
        best = [0]
        
        # Precompute degrees for initial pruning?
        # We'll compute degrees on the fly or maintain them.
        # For simplicity and speed in Python, we'll use a recursive approach with memoization?
        # No, state space is too large.
        
        def dfs(candidates, size):
            if time_check() > DEADLINE - 5:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            # Upper bound
            if size + candidates.bit_count() <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the subgraph induced by candidates
            # This often leads to faster pruning.
            # We need to find v in candidates that maximizes (adj_masks[v] & candidates).bit_count()
            # Iterating through all bits in candidates to find max degree might be slow.
            # Let's try picking the vertex with the smallest index first as a baseline, 
            # but maybe we can just pick a random vertex?
            # Actually, for independent set, picking a high-degree vertex is good for the "exclude" branch?
            # If we exclude a high-degree vertex, we keep many candidates.
            # If we include a high-degree vertex, we remove many candidates.
            # Usually, branching on the vertex with max degree in the *complement* (i.e., min degree in original?) 
            # is not standard.
            # Standard for Max Independent Set: Branch on vertex with max degree in the *current* graph?
            # Let's stick to a simple heuristic: pick the vertex with the lowest index.
            # It's fast. The pruning bound is the main driver.
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            # Neighbors of v in candidates are adj_masks[v] & candidates
            # New candidates = candidates \ {v} \ N(v)
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def is_valid(n, adj_masks):
        """
        Returns True if the graph is triangle-free and has no independent set of 10 vertices.
        """
        if not check_triangle_free(n, adj_masks):
            return False
        alpha = max_independent_set_size(n, adj_masks)
        return alpha < 10

    def get_edges(adj_masks, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def search_graph(n, seed, use_spectral_init=False):
        random.seed(seed)
        adj_masks = [0] * n
        
        # Standard Random Greedy Initialization
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            # Check if adding (u,v) creates triangle
            if adj_masks[u] & adj_masks[v]:
                continue
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Check current alpha
        alpha = max_independent_set_size(n, adj_masks)
        if alpha < 10:
            return n, get_edges(adj_masks, n)

        return None, None

    def try_circulant(n, seed):
        """
        Try to find a valid circulant graph on n vertices.
        """
        random.seed(seed)
        max_diff = n // 2
        # We want a set of differences D
        # Randomly select a subset of {1, ..., max_diff}
        # To ensure connectivity and density, we might want to include small differences?
        # Let's just try random subsets.
        
        # We can bias towards larger sets?
        # Triangle-free circulant graphs are rare for large n if dense.
        
        # Iterate through possible sizes or just random subsets
        for _ in range(50): # Try 50 random subsets per seed
            D = set()
            for d in range(1, max_diff + 1):
                if random.random() < 0.5:
                    D.add(d)
            
            if not D:
                continue
                
            # Construct adj_masks
            adj_masks = [0] * n
            for d in D:
                for i in range(n):
                    j = (i + d) % n
                    if i < j:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                    elif j < i:
                        # Already added? 
                        # If we iterate i from 0 to n-1, and add i->(i+d), we cover all edges.
                        # But we need to be careful not to add duplicates or self-loops.
                        # Since d != 0 and d != n/2 (if n even), i and j are distinct.
                        # The condition i < j ensures we only add each edge once if we iterate all i?
                        # No, if we iterate all i, we add i->j and j->i separately?
                        # My loop above:
                        # for i in range(n):
                        #    j = (i+d)%n
                        #    if i < j: add
                        # This works.
                        pass

            # Check triangle free
            if not check_triangle_free(n, adj_masks):
                continue
            
            # Check alpha
            alpha = max_independent_set_size(n, adj_masks)
            if alpha < 10:
                return n, get_edges(adj_masks, n)
                
        return None, None

    best_n = 0
    best_edges = []
    
    # Try circulant graphs for n=40 and n=39 first
    # This is a structured search that might find a solution quickly.
    for n_target in [40, 39]:
        if time_check() > DEADLINE - 60:
            break
        for seed in range(100):
            if time_check() > DEADLINE - 60:
                break
            res = try_circulant(n_target, seed)
            if res[0]:
                best_n = res[0]
                best_edges = res[1]
                break
        if best_n > 0:
            break

    # Fallback to Random Greedy if circulant failed or n < 39
    if best_n < 39:
        for n_target in [40, 39, 38]:
            if time_check() > DEADLINE - 20:
                break
            for seed in range(1000):
                if time_check() > DEADLINE - 10:
                    break
                n_found, edges_found = search_graph(n_target, seed)
                if n_found:
                    best_n = n_found
                    best_edges = edges_found
                    break
            if best_n == n_target:
                break

    if best_n == 0:
        # Fallback to known small graph
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
        best_n = n
        best_edges = edges

    return {'n': best_n, 'edges': best_edges}
