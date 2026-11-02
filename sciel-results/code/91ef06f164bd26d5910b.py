import time
import random
import sys

def construct_luw(k, g):
    # ... (copy from prompt) ...
    pass

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # If k is not 4, or g is not 10, fallback to LUW
    # The task specifies k=4, g=10, but interface is generic.
    if k != 4 or g != 10:
        return construct_luw(k, g)
        
    # Try to find a better graph using local search
    # Target n: We know 320 is possible.
    # We want to minimize n.
    # Let's try a range of n values.
    # Moore bound is 242.
    # Let's try n from 242 to 320.
    # But search is stochastic.
    # Strategy:
    # 1. Try n=320 for 30s.
    # 2. If success, return it.
    # 3. If fail, try n=300 for 30s.
    # 4. If fail, try n=280 for 30s.
    # 5. If fail, return LUW.
    
    # Actually, finding a valid graph for n=242 is very hard if it doesn't exist.
    # It's better to target a known feasible n like 320 and try to optimize down?
    # No, we can't "optimize down" vertices easily.
    # We must construct for a specific n.
    
    # Let's try n=320 first as it's the known best.
    # If we find a valid graph with n=320, score is 2048/320 = 6.4.
    # If we can find n=300, score is 6.8.
    
    # We will attempt to construct a valid graph for n=320.
    # If we succeed, we check if we can try smaller n in remaining time.
    
    best_result = None
    best_n = float('inf')
    
    # List of target sizes to try, sorted by n (smaller is better)
    # But we need to balance probability of success.
    # Let's just try 320, then 300, then 280, then 260, then 242.
    targets = [320, 300, 280, 260, 242]
    
    for n in targets:
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph for this n
        success = False
        for attempt in range(20): # Max 20 random restarts per n
            if time.time() > deadline - 10:
                break
                
            edges = random_regular_graph(n, 4)
            if not edges:
                continue
                
            # Optimize girth
            adj = build_adjacency(n, edges)
            girth = get_girth(adj, n, g)
            
            if girth >= g:
                best_result = (n, edges)
                best_n = n
                success = True
                break
            
            # Local search to improve girth
            girth_improved = local_search(adj, edges, n, g, deadline - 10)
            if girth_improved:
                # Re-calculate girth to be sure
                new_girth = get_girth(adj, n, g)
                if new_girth >= g:
                    # Reconstruct edges from adj? 
                    # local_search modifies adj and edges?
                    # Let's make local_search return new edges
                    pass

    if best_result:
        return best_result
    else:
        return construct_luw(k, g)
