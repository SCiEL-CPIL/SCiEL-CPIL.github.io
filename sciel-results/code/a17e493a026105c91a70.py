import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds buffer
    n = 31
    target_edges = 91
    base_edges = 90
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_edges = None
    
    # We will run random greedy constructions
    # To speed up, we use bitmasks for adjacency? n=31 fits in 64-bit int.
    # But Python ints are arbitrary precision, so we can use them as bitsets.
    
    while time.time() < deadline:
        random.shuffle(all_pairs)
        adj = [0] * n  # adj[u] is a bitmask of neighbors of u
        edge_list = []
        
        # Greedily add edges to avoid C4
        for u, v in all_pairs:
            if len(edge_list) >= base_edges:
                break
            
            # Check if adding (u, v) creates a C4
            # A C4 is u-a-b-v-u.
            # This exists if there is an edge (a, b) where a in N(u) and b in N(v).
            # Let's iterate through neighbors of u
            # Neighbors of u are bits in adj[u]
            # Neighbors of v are bits in adj[v]
            
            # We need to check if there exists a, b such that:
            # bit a is set in adj[u]
            # bit b is set in adj[v]
            # bit b is set in adj[a] (or bit a in adj[b])
            
            # Efficient check:
            # For each a in N(u), check if N(a) intersects N(v).
            # N(a) is adj[a]. N(v) is adj[v].
            # Intersection is adj[a] & adj[v].
            # If any intersection is non-zero, we have a C4.
            
            creates_c4 = False
            # Get neighbors of u
            # Iterate through set bits in adj[u]
            au = adj[u]
            while au:
                # Get lowest set bit
                lsb = au & (-au)
                a_idx = lsb.bit_length() - 1
                # Check intersection of adj[a_idx] and adj[v]
                if adj[a_idx] & adj[v]:
                    creates_c4 = True
                    break
                au ^= lsb
            
            if not creates_c4:
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_list.append((u, v))
        
        if len(edge_list) < base_edges:
            continue
            
        # Now we have a C4-free graph with 90 edges.
        # Find the best non-edge to add to minimize C4s.
        # The cost of adding (u, v) is the number of C4s created,
        # which is the number of pairs (a, b) with a in N(u), b in N(v), and (a, b) in E.
        # This is sum_{a in N(u)} popcount(adj[a] & adj[v]).
        
        min_cost = float('inf')
        best_edge_to_add = None
        
        # Check all non-edges
        # We can iterate through all pairs and check if they are in edge_list
        # To speed up, we can just check all pairs and skip if in adj
        for u, v in all_pairs:
            if adj[u] & (1 << v):
                continue # Edge already exists
            
            # Calculate cost
            cost = 0
            au = adj[u]
            while au:
                lsb = au & (-au)
                a_idx = lsb.bit_length() - 1
                # Count bits in adj[a_idx] & adj[v]
                cost += (adj[a_idx] & adj[v]).bit_count()
                au ^= lsb
            
            if cost < min_cost:
                min_cost = cost
                best_edge_to_add = (u, v)
                
                # If cost is 0, we can't do better, break early?
                # Yes, 0 is the minimum possible.
                if min_cost == 0:
                    break
        
        if best_edge_to_add is not None:
            total_c4s = min_cost
            if total_c4s < best_score:
                best_score = total_c4s
                best_edges = edge_list + [list(best_edge_to_add)]
                
                # If we found 0, we can stop early?
                # It's the theoretical minimum.
                if best_score == 0:
                    break

    # If no graph was found (should not happen), return a default
    if best_edges is None:
        # Fallback: Just take first 91 pairs
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:target_edges]
        
    return best_edges
