import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 31
    target_edges = 91
    
    # Greedy construction of C4-free graph
    # We need 90 edges first.
    
    # Use a list of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We will try multiple random restarts if the first one fails or takes too long
    # But n=31 is small, greedy should be fast.
    
    best_edges = None
    best_c4_count = float('inf')
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        edges = []
        adj = [set() for _ in range(n)]
        
        # Add 90 edges greedily
        for u, v in all_edges:
            if len(edges) >= 90:
                break
            # Check common neighbors
            # Common neighbors are in adj[u] & adj[v]
            # If len(common) >= 2, adding (u,v) creates a C4
            common = len(adj[u] & adj[v])
            if common < 2:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        if len(edges) < 90:
            # Failed to find 90 edges, try again
            continue
            
        # Now we have a C4-free graph with 90 edges.
        # Find a non-edge with minimum common neighbors to add.
        # We want to minimize the number of C4s created.
        # C4s created = number of common neighbors of the added edge.
        
        # Build set of existing edges for fast lookup
        edge_set = set(edges)
        
        min_common = float('inf')
        best_new_edge = None
        
        for u, v in all_edges:
            if (u, v) in edge_set:
                continue
            common = len(adj[u] & adj[v])
            if common < min_common:
                min_common = common
                best_new_edge = (u, v)
                
        if best_new_edge is None:
            continue
            
        # Add the best edge
        final_edges = edges + [best_new_edge]
        
        # Verify C4 count (should be min_common, which is likely 1)
        # But let's just return the edges. The evaluator counts them.
        # We assume min_common is 1. If it is 0, we have a C4-free 91-edge graph, which is impossible.
        # So it must be >= 1.
        
        # To be safe, we can calculate the exact C4 count for the final graph
        # and keep the best one across restarts.
        
        # Count C4s in final_edges
        adj_final = [set() for _ in range(n)]
        for u, v in final_edges:
            adj_final[u].add(v)
            adj_final[v].add(u)
            
        c4_count = 0
        # Count C4s by iterating over pairs of vertices
        # A C4 is u-v-w-x-u.
        # We can count by summing over pairs (u, w) the number of common neighbors.
        # Each C4 is counted twice (once for each pair of opposite vertices).
        # So C4s = 1/2 * sum_{u < w} (|N(u) & N(w)| choose 2)
        
        for u in range(n):
            for w in range(u+1, n):
                common = len(adj_final[u] & adj_final[w])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
                    
        c4_count //= 2
        
        if c4_count < best_c4_count:
            best_c4_count = c4_count
            best_edges = final_edges
            
        if best_c4_count == 1:
            # Optimal found
            break
            
    if best_edges is None:
        # Fallback: just return first 91 edges
        return [[i, j] for i in range(n) for j in range(i+1, n)][:91]
        
    return [list(e) for e in best_edges]
