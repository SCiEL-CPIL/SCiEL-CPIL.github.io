import random
import time
import sys

def construct():
    n = 18
    best_score = n
    best_edges = None
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Precompute combinations of 4 vertices for K4 check
    # There are 3060 K4s.
    # We can represent the graph as a list of sets or bitsets.
    
    def is_k4_free(adj):
        # adj is a list of bitmasks
        # Check all 4-tuples
        # This is slow if done naively for every graph.
        # Better: check during generation or use a fast check.
        # For n=18, 3060 checks is fast.
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1: continue
                for k in range(j+1, n):
                    if not (adj[i] >> k) & 1: continue
                    if not (adj[j] >> k) & 1: continue
                    for l in range(k+1, n):
                        if not (adj[i] >> l) & 1: continue
                        if not (adj[j] >> l) & 1: continue
                        if not (adj[k] >> l) & 1: continue
                        return False
        return True

    def max_triangle_free_induced(adj):
        # Find max subset of vertices with no triangle.
        # Use branch and bound.
        # State: current set of vertices (bitset), current size, best size.
        # Optimization: Precompute for each vertex the set of vertices that form a triangle with it?
        # Or just recursive search.
        
        best = [0]
        
        def search(mask, size, start_idx):
            if size + (n - start_idx) <= best[0]:
                return
            if mask == 0:
                if size > best[0]:
                    best[0] = size
                return
            
            # Try to add vertices
            # We can use a greedy lower bound or just iterate.
            # To speed up, we can pick a vertex that is "hard" to include.
            # For simplicity, iterate from start_idx.
            
            # Pruning: if adding all remaining vertices doesn't beat best, stop.
            
            # Find next candidate
            # We can maintain a list of available vertices.
            
            # Simple recursion:
            # Include v: if v doesn't create a triangle with existing vertices.
            # Exclude v.
            
            # This naive recursion is $2^n$. $2^{18}$ is fine.
            # But we need to check "triangle-free" condition.
            # A set is triangle-free if no 3 vertices form a triangle.
            
            # Let's use a different approach:
            # Iterate through all subsets? No.
            # Backtracking:
            # Maintain a bitset `current` of included vertices.
            # When adding vertex `v`, we must ensure that `v` does not form a triangle with any pair in `current`.
            # This means for all `u, w` in `current`, if `u-w` is an edge, then `u-v` and `w-v` cannot both be edges?
            # No, triangle is `u-v-w`. So if `u-w` is edge, we cannot add `v` if both `u-v` and `w-v` are edges.
            # Actually, we just need to check if `v` forms a triangle with any two vertices in `current`.
            # This is equivalent to: `v` is not in the "triangle-closure" of `current`.
            
            # Let's precompute `triangles_with[v]`? No.
            # Let's just check on the fly.
            
            pass # Placeholder for actual implementation below

        # Implementation of search
        def search_rec(mask, size, start):
            if size + (n - start) <= best[0]:
                return
            
            if start == n:
                if size > best[0]:
                    best[0] = size
                return

            # Option 1: Exclude vertex 'start'
            search_rec(mask, size, start + 1)
            
            # Option 2: Include vertex 'start'
            v = start
            # Check if v can be added to 'mask' without creating a triangle
            # A triangle is formed if there are u, w in mask such that u-w is edge, u-v is edge, w-v is edge.
            # This is equivalent to checking if v has two neighbors in mask that are connected to each other.
            
            # Let's find neighbors of v in mask
            neigh_in_mask = mask & adj[v]
            
            # Check if any edge exists within neigh_in_mask
            # If yes, adding v creates a triangle.
            has_triangle = False
            # Iterate through bits in neigh_in_mask
            nm = neigh_in_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                nm ^= lsb
                # Check if u is connected to any other vertex in neigh_in_mask
                # adj[u] & neigh_in_mask & ~lsb
                if adj[u] & (neigh_in_mask & ~lsb):
                    has_triangle = True
                    break
            
            if not has_triangle:
                search_rec(mask | (1 << v), size + 1, start + 1)
            
        search_rec(0, 0, 0)
        return best[0]

    # Main loop
    # Try different probabilities
    for p in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for seed in range(1000):
            if time.time() - start_time > timeout:
                break
            
            random.seed(seed)
            adj = [0] * n
            edges = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append([i, j])
            
            # Check K4-free
            if not is_k4_free(adj):
                continue
            
            # Evaluate score
            score = max_triangle_free_induced(adj)
            
            if score < best_score:
                best_score = score
                best_edges = edges
                # Update timeout if we find a very good score early? No, keep searching.
                # Maybe break if score is very low?
                if best_score <= 7: # Theoretical lower bound?
                    # We can try to optimize further, but 7 is very good.
                    pass

            if time.time() - start_time > timeout:
                break
        if time.time() - start_time > timeout:
            break

    if best_edges is None:
        # Fallback to seed program
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return best_edges
