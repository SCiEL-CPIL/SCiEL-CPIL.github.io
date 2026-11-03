import time
import random
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using a greedy randomized approach
    with bitset-based girth verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound approximation: 1 + k * sum(k^i for i in range((g-1)//2))
    # For odd g=2r+1, bound is 1 + k * (k^r - 1)/(k - 1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum([k**i for i in range(r)])
    
    # Start from a lower bound, maybe slightly above Moore bound
    # The prompt mentions lower bound 428 for k=5, g=9.
    # Let's start trying from max(moore_bound, 100)
    
    best_n = None
    best_edges = None
    
    # We will try increasing n until we find a valid graph
    # Or we can try random restarts for a fixed n.
    # Given the difficulty, let's try to build for a specific n.
    # If we fail, increase n.
    
    # Heuristic: The actual cage number might be significantly larger than Moore bound.
    # Let's start at Moore bound and increase by small steps or jumps.
    
    # For k=5, g=9, Moore bound is 781.
    # Let's try n from 800 upwards.
    
    # To save time, we can parallelize mentally by just running sequentially with time checks.
    
    def check_girth_constraint(u, v, adj_bits, max_depth):
        """
        Returns True if adding edge (u,v) would create a cycle of length < g.
        This is equivalent to checking if there is a path of length <= g-2 between u and v.
        """
        if u == v:
            return True
        if adj_bits[u] & (1 << v):
            return True # Already connected (cycle of length 2? No, simple graph, so this shouldn't happen if we check before adding)
        
        # BFS using bitsets
        seen = 1 << u
        frontier = 1 << u
        
        for _ in range(max_depth):
            next_frontier = 0
            # Iterate over set bits in frontier
            f = frontier
            while f:
                lsb = f & -f
                idx = lsb.bit_length() - 1
                next_frontier |= adj_bits[idx]
                f -= lsb
            
            next_frontier &= ~seen
            if not next_frontier:
                break
            seen |= next_frontier
            if seen & (1 << v):
                return True
            frontier = next_frontier
            
        return False

    def try_build(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        adj_bits = [0] * n
        
        # We will add vertices 0..n-1
        # Vertex 0 has no edges yet.
        # We need to ensure every vertex ends up with degree k.
        # Greedy: For vertex u, connect to k existing vertices v < u.
        # This requires that the "deficit" of previous vertices is sufficient.
        
        # A better way:
        # 1. Initialize all degrees to 0.
        # 2. For u in range(n):
        #    - Find k vertices v < u with degree < k.
        #    - Check girth constraint.
        #    - If found, add edge.
        #    - If not, fail.
        
        # This simple greedy often fails because it doesn't plan for the future.
        # We can use a "random" order of processing or random selection of neighbors.
        
        for u in range(n):
            if u == 0:
                continue
                
            # Find candidates: v < u such that len(adj[v]) < k
            candidates = [v for v in range(u) if len(adj[v]) < k]
            random.shuffle(candidates)
            
            edges_added = 0
            for v in candidates:
                if edges_added == k:
                    break
                if len(adj[u]) == k:
                    break
                    
                # Check if v is already connected to u
                if v in adj[u]:
                    continue
                
                # Check girth constraint
                # Path length < g-1
                if check_girth_constraint(u, v, adj_bits, g - 2):
                    continue
                
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges_added += 1
                
            if len(adj[u]) < k:
                return False, None, None
        
        # Verify all degrees are k
        for u in range(n):
            if len(adj[u]) != k:
                return False, None, None
        
        # Construct edge list
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return True, n, edges

    # Main loop
    # Try different n values
    # Start from a reasonable estimate
    # For k=5, g=9, let's start at 800.
    # If that fails, try 900, 1000, etc.
    
    # To be safe, we can also try smaller n if we think Moore bound is tight?
    # But Moore bound is a lower bound.
    
    current_n = max(moore_bound, 100)
    # Step size
    step = 50
    attempts = 0
    
    while time.time() < deadline:
        # Try to build graph of size current_n
        # Use a few random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            success, n_found, edges_found = try_build(current_n, seed + attempts * 1000)
            if success:
                if best_n is None or n_found < best_n:
                    best_n = n_found
                    best_edges = edges_found
                # If we found one, we can break inner loop and try smaller n?
                # No, we want the minimum n.
                # If we found a valid graph of size N, we should try smaller N.
                # But we are increasing N.
                # So if we find a valid graph, we should record it and then try to find a smaller one?
                # The loop structure is increasing N.
                # So if we find a valid graph at N=1000, we should stop increasing and try to find a valid graph at N=950, 900, etc.
                # Or simply, if we find a valid graph, we can return it if we are confident it's near optimal?
                # No, we should try to minimize n.
                
                # Strategy:
                # 1. Find a valid graph with some N.
                # 2. Try to find a valid graph with N - step, N - 2*step, etc.
                
                # Let's restructure:
                # We don't know the optimal N.
                # Let's just try increasing N until we find one.
                # Then we can try to reduce it.
                
                # For now, if we find one, we store it.
                # We break the seed loop and try a smaller N?
                # No, if we found one at N, it's possible a smaller one exists.
                # But searching smaller N is harder (more constrained).
                # So it's better to find a large one first, then binary search down?
                
                # Let's just try increasing N. If we find one, we can try to improve it by reducing N.
                # But time is limited.
                
                # Let's assume the first valid N we find is a good candidate.
                # We will break the outer loop if we find a valid graph?
                # No, we should try a few more seeds for the same N to see if we can get a different structure?
                # No, N is fixed.
                
                # Let's break and try to reduce N.
                # Actually, let's just return the first valid graph we find to ensure we return *something*.
                # But we want to minimize n.
                
                # Let's try a range of N.
                # If we find a valid graph at N, we set best_n = N.
                # Then we try N - step.
                
                break
        
        if best_n is not None:
            # We found a valid graph. Let's try to find a smaller one.
            # Try current_n - step
            target_n = current_n - step
            if target_n < moore_bound:
                break
            current_n = target_n
            # Continue loop to try smaller N
        else:
            # Failed to build for current_n. Increase N.
            current_n += step
            attempts += 1
            
            # If current_n becomes too large, we might run out of time.
            # But we need to find *a* graph.
            
    if best_n is not None:
        return best_n, best_edges
    else:
        # If we didn't find any, we must return something valid.
        # The LUW construction is a fallback.
        # Let's implement a simple LUW or just return a small invalid graph?
        # No, must be valid.
        # Let's use the original LUW code as a fallback if the greedy fails.
        
        # Re-implement LUW briefly for fallback
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

# EVOLVE-BLOCK-END
