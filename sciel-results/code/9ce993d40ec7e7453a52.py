import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit

    def is_regular(adj, subset, n):
        """
        Check if the induced subgraph on 'subset' (list of vertices) is regular.
        """
        if not subset:
            return False
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Calculate degree of each vertex in subset
        first_deg = -1
        for v in subset:
            deg = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
        return True

    def try_build(n):
        """
        Attempt to build a valid graph on n vertices using greedy incremental construction.
        Returns adj list if successful, None otherwise.
        """
        adj = [0] * n
        # We build vertex by vertex: 0, 1, ..., n-1
        # For vertex v, we decide edges to 0..v-1.
        
        # To make it faster, we can use a simple heuristic: 
        # Try random edge sets for the new vertex. If a conflict arises, try another.
        # If too many conflicts, fail this attempt.
        
        max_retries_per_vertex = 50
        
        for v in range(n):
            found_valid = False
            for _ in range(max_retries_per_vertex):
                # Generate random edges from v to 0..v-1
                new_adj_v = 0
                for u in range(v):
                    if random.random() < 0.5:
                        new_adj_v |= (1 << u)
                
                # Tentatively assign
                # Check all subsets S of {0..v} that contain v and have size >= 6
                # If such a subset is regular, this edge choice is invalid.
                
                valid = True
                # We need to check subsets of size k from 6 to v+1
                # The subset must contain v.
                # So we choose k-1 vertices from {0..v-1}.
                
                # Optimization: Only check if v+1 >= 6
                if v + 1 >= 6:
                    # Iterate over all subsets of {0..v-1} of size k-1
                    # This is expensive if we do it naively for every k.
                    # But v is small (max 20).
                    # Number of subsets of {0..v-1} of size k-1 is C(v, k-1).
                    # Total checks for one v is sum_{k=6}^{v+1} C(v, k-1) = sum_{j=5}^{v} C(v, j).
                    # For v=20, sum_{j=5}^{20} C(20, j) is approx 2^20 / 2 = 500k.
                    # This is too slow if we do it for every retry (50 retries * 500k = 25M ops per vertex).
                    # 20 vertices * 25M = 500M ops. Too slow for Python.
                    
                    # We need a faster check or fewer retries.
                    # Let's reduce retries and optimize the check.
                    pass
                
                # Faster Check Strategy:
                # Instead of checking all subsets, we can check specific "dangerous" ones?
                # Or we can just check size 6 first, as it's the most common constraint?
                # No, size 7+ are also forbidden.
                
                # Let's try a different approach: 
                # Check only size 6 subsets containing v. 
                # If a size 7+ regular subgraph exists, does it imply a size 6 regular subgraph?
                # No.
                # However, regular subgraphs of size > 6 are rarer.
                # Let's just check size 6 for speed, and verify the final graph for all sizes.
                # If the greedy pass with size-6 check produces a graph, we verify it fully.
                # If it fails full verification, we discard.
                
                # So, inner loop: Check only size 6 subsets containing v.
                valid = True
                if v + 1 >= 6:
                    # Choose 5 vertices from 0..v-1
                    # Use itertools.combinations for clarity, or bit masks.
                    # C(20, 5) = 15504. This is fast enough.
                    import itertools
                    for combo in itertools.combinations(range(v), 5):
                        subset = list(combo) + [v]
                        # Check if subset is regular
                        mask = 0
                        for x in subset:
                            mask |= (1 << x)
                        # Check degrees
                        first_deg = -1
                        is_reg = True
                        for x in subset:
                            # Degree of x in subset
                            # adj[x] contains edges to all previous vertices (0..x-1) and potentially v (if x < v)
                            # But we haven't updated adj yet. 
                            # We need to simulate the degree.
                            # For x in combo (x < v):
                            # deg_x = (adj[x] & mask).bit_count() + (1 if (new_adj_v & (1<<x)) else 0)
                            # For v:
                            # deg_v = (new_adj_v & mask).bit_count()
                            
                            if x == v:
                                deg = (new_adj_v & mask).bit_count()
                            else:
                                deg = (adj[x] & mask).bit_count()
                                if new_adj_v & (1 << x):
                                    deg += 1
                            
                            if first_deg == -1:
                                first_deg = deg
                            elif deg != first_deg:
                                is_reg = False
                                break
                        if is_reg:
                            valid = False
                            break
                
                if valid:
                    # Commit to this edge configuration
                    adj[v] = new_adj_v
                    for u in range(v):
                        if new_adj_v & (1 << u):
                            adj[u] |= (1 << v)
                    found_valid = True
                    break
            
            if not found_valid:
                return None
        
        return adj

    def verify_full(n, adj):
        """
        Verify that no subset of size >= 6 is regular.
        """
        import itertools
        # Check all subsets of size k from 6 to n
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in subset:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Try to build for n=21, then 20, etc.
    for target_n in [21, 20, 19]:
        if time.time() > deadline:
            break
        
        # Multiple attempts for each n
        attempts = 20 if target_n == 21 else 50
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            adj = try_build(target_n)
            if adj is not None:
                # Verify full validity
                if verify_full(target_n, adj):
                    return {'n': target_n, 'edges': get_edges(adj, target_n)}
    
    # Fallback: Return a small valid graph (e.g., n=5)
    # A graph with n=5 has no subsets of size 6, so it's always valid.
    adj = [0] * 5
    return {'n': 5, 'edges': []}
