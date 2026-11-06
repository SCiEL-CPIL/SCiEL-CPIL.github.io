import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145

    # Known valid 20-vertex graph
    known_20_edges = [
        [0, 1], [0, 2], [0, 5], [0, 6], [0, 9], [0, 10], [0, 13], [0, 14],
        [1, 3], [1, 4], [1, 7], [1, 8], [1, 11], [1, 12], [1, 15], [1, 16],
        [2, 3], [2, 6], [2, 8], [2, 10], [2, 13], [2, 16], [2, 18],
        [3, 5], [3, 9], [3, 12], [3, 14], [3, 17],
        [4, 6], [4, 7], [4, 10], [4, 15], [4, 18],
        [5, 7], [5, 11], [5, 14], [5, 17],
        [6, 8], [6, 12], [6, 15],
        [7, 9], [7, 13], [7, 16],
        [8, 11], [8, 14], [8, 19],
        [9, 12], [9, 15], [9, 18],
        [10, 13], [10, 16], [10, 19],
        [11, 14], [11, 17],
        [12, 15], [12, 18],
        [13, 16], [13, 19],
        [14, 17],
        [15, 18],
        [16, 19],
    ]

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def find_violation(n, adj_masks):
        """Returns a violating subset (list of vertices) or None."""
        # Check sizes from 6 to n
        # Optimization: Pre-compute masks? No, too much memory.
        # Optimization: Early exit.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if degrees are all equal
                d0 = (adj_masks[subset[0]] & mask).bit_count()
                is_regular = True
                for v in subset[1:]:
                    if (adj_masks[v] & mask).bit_count() != d0:
                        is_regular = False
                        break
                if is_regular:
                    return list(subset)
        return None

    def local_search(n, adj_masks, time_budget):
        """
        Perform local search to find a valid graph.
        Returns valid adj_masks or None.
        """
        deadline = time.time() + time_budget
        current_adj = list(adj_masks)
        
        while time.time() < deadline:
            violation = find_violation(n, current_adj)
            if violation is None:
                return current_adj
            
            # Pick a random pair in the violation to flip
            # Flipping an edge within the violation changes degrees of 2 vertices
            # This is likely to break regularity
            if len(violation) >= 2:
                u, v = random.sample(violation, 2)
                # Flip edge (u, v)
                if (current_adj[u] >> v) & 1:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            else:
                # Should not happen for k>=6
                break
                
        return None

    # Step 1: Verify n=20
    adj_20 = build_adj_masks(20, known_20_edges)
    # We trust the known graph, but let's be safe. 
    # If it fails, we return a small valid graph.
    if find_violation(20, adj_20) is not None:
        # Fallback: Empty graph on 5 vertices is valid (no subset of size 6)
        return {'n': 5, 'edges': []}

    # Step 2: Try to find n=21
    # Strategy A: Extend n=20
    # Strategy B: Random n=21
    
    # Try Strategy A first
    # Add vertex 20
    adj_21 = adj_20 + [0]
    # Connect vertex 20 to random subset of 0..19
    # Let's try a few random configurations for vertex 20
    for attempt in range(10):
        if time.time() > time_limit - 10: break
        
        # Reset vertex 20 connections
        adj_21 = adj_20 + [0]
        # Random degree for vertex 20? 
        # Let's pick a random mask for vertex 20
        mask_20 = random.getrandbits(20)
        adj_21[20] = mask_20
        for i in range(20):
            if (mask_20 >> i) & 1:
                adj_21[i] |= (1 << 20)
        
        # Run local search
        result = local_search(21, adj_21, time_budget=5.0)
        if result is not None:
            # Final verification
            if find_violation(21, result) is None:
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}

    # Try Strategy B: Random n=21
    for attempt in range(5):
        if time.time() > time_limit - 5: break
        
        adj_21 = [0] * 21
        for i in range(21):
            for j in range(i + 1, 21):
                if random.random() < 0.5:
                    adj_21[i] |= (1 << j)
                    adj_21[j] |= (1 << i)
                    
        result = local_search(21, adj_21, time_budget=5.0)
        if result is not None:
            if find_violation(21, result) is None:
                edges = get_edges_from_masks(21, result)
                return {'n': 21, 'edges': edges}

    # Step 3: Return n=20
    return {'n': 20, 'edges': known_20_edges}
