import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    def is_triangle_free_adj(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def is_3_colorable(adj):
        # Fast backtracking with bitmasks
        # adj[i] is a bitmask of neighbors
        # colors: 0, 1, 2
        # We will use a recursive function with memoization or just fast backtracking.
        # Since n=16 is small, fast backtracking should be very quick.
        
        # Order vertices by degree (descending) to prune faster?
        # But we need to map back to original indices.
        # For n=16, simple order might be fine if bitmasks are used.
        
        color = [-1] * n
        # avail[i] is a 3-bit mask of available colors
        avail = [7] * n # 0b111
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = idx
            # Try each available color
            mask = avail[u]
            while mask:
                # Pick the lowest set bit
                c_bit = mask & (-mask)
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                color[u] = c
                # Update neighbors
                neighbors = adj[u]
                # We need to check if any neighbor becomes unavailable
                # Actually, we just need to check if we can proceed.
                # A better way: check if any neighbor has no available colors.
                # But we are doing forward checking.
                
                # Let's just do simple backtracking with bitmasks for available colors
                # We don't need to update all neighbors' avail masks permanently if we backtrack.
                # We can just check consistency.
                
                # Simpler: Just try the color and recurse.
                # The "avail" check is done by the recursion.
                # To speed up, we can check if any neighbor v of u has (avail[v] & ~c_bit) == 0?
                # No, that's not right. avail[v] is reduced by colors of colored neighbors.
                
                # Let's use a standard efficient backtracking:
                # 1. Pick a vertex (heuristic: most constrained, or just order).
                # 2. Try colors.
                # 3. Check consistency with neighbors.
                
                # For n=16, the overhead of heuristics might not be worth it if the bitmask check is fast.
                # Let's just recurse.
                
                # Check if coloring u with c is consistent with already colored neighbors
                valid = True
                neighbors_mask = adj[u]
                # Iterate through neighbors
                v = 0
                m = neighbors_mask
                while m:
                    v_bit = m & (-m)
                    v = v_bit.bit_length() - 1
                    m ^= v_bit
                    if color[v] == c:
                        valid = False
                        break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                color[u] = -1
                
            return False
            
        return backtrack(0)

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    def greedy_add_edges(adj):
        added = 0
        possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        random.shuffle(possible)
        
        for u, v in possible:
            if is_triangle_free_adj(adj, u, v):
                # Check if adding edge keeps it not 3-colorable?
                # Adding edges preserves non-3-colorability if the graph was not 3-colorable?
                # No! Adding edges can make a graph 3-colorable?
                # Wait. If G is not 3-colorable, is G+e not 3-colorable?
                # Yes. If G+e is 3-colorable, then G (subgraph) is 3-colorable.
                # So adding edges NEVER makes a non-3-colorable graph 3-colorable.
                # It can only make it *more* constrained.
                # So we don't need to check 3-colorability when adding edges!
                # We only need to check triangle-freeness.
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def make_grotzsch():
        adj = [0] * n
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u, v in cycle:
            adj[u] |= (1 << (5+v))
            adj[5+v] |= (1 << u)
            adj[v] |= (1 << (5+u))
            adj[5+u] |= (1 << v)
            
        for i in range(5):
            adj[10] |= (1 << (5+i))
            adj[5+i] |= (1 << 10)
            
        return adj

    def make_random_tf():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    if is_triangle_free_adj(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    while time.time() < deadline:
        # We need a starting graph that is NOT 3-colorable.
        # Grötzsch is not 3-colorable.
        # Random TF graphs are likely 3-colorable (most are bipartite or close).
        # So we should mostly start with Grötzsch or derivatives.
        
        if random.random() < 0.8:
            adj = make_grotzsch()
        else:
            adj = make_random_tf()
        
        # Verify it's not 3-colorable
        if not is_3_colorable(adj):
            greedy_add_edges(adj)
            edges = get_edges_from_adj(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
        else:
            # If random TF is 3-colorable, we can try to add edges to make it not?
            # No, adding edges doesn't help with 3-colorability if it's already 3-colorable?
            # Wait. If G is 3-colorable, G+e might be not 3-colorable.
            # Example: C5 is not 3-colorable? No, C5 is not 3-colorable?
            # C5 has odd cycle. Chromatic number 3.
            # C3 is not 3-colorable? No, C3 is 3-colorable.
            # K4 is not 3-colorable.
            # If we have a 3-colorable graph, we can add edges to make it not 3-colorable?
            # Yes, if we add enough edges to force a 4-chromatic structure.
            # But we are constrained by triangle-freeness.
            # So we can't add K4.
            # We can add edges to create a Grötzsch-like structure?
            # This is hard to do greedily.
            pass

    return best_edges
