import time

def construct():
    n = 20
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    # Greedy addition of edges
    # To maximize edges, we should try to add edges in a way that leaves room for more.
    # Randomization or specific ordering can help.
    # A simple ordering is (u, v) with u < v.
    
    # To get a denser graph, we can try multiple random orderings and keep the best.
    # Given the small N, we can afford a few restarts.
    
    best_edges = []
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle edge order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        import random
        random.seed(attempt) # Deterministic for reproducibility in eval? 
        # Actually, random.seed is good for consistency, but different seeds might find different sizes.
        # We want the max, so let's just use a fixed good heuristic or multiple randoms.
        
        random.shuffle(all_edges)
        
        current_edges = []
        # Reset adj
        adj = [[False] * n for _ in range(n)]
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
                
            # Check if adding (u, v) creates a 4-cycle
            creates_c4 = False
            # A 4-cycle u-v-w-x-u is created if there exist w, x
            # such that (v,w) and (x,u) and (w,x) are in the graph.
            # Note: w and x must be distinct and not u or v.
            
            # Iterate over neighbors of v
            for w in range(n):
                if w == u or w == v:
                    continue
                if not adj[v][w]:
                    continue
                # Iterate over neighbors of u
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if w == x:
                        continue
                    if not adj[u][x]:
                        continue
                    if adj[w][x]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u][v] = True
                adj[v][u] = True
                current_edges.append([u, v, 0])
                
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            
    # Return the best found
    return best_edges
