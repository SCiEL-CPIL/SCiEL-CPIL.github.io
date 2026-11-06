import random
import time

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    max_count = 0
    start_time = time.time()
    deadline = start_time + 110 # Safety margin for 120s limit
    
    # We can try multiple random restarts to find a denser C4-free graph
    # The greedy algorithm is order-dependent.
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        # Shuffle edges to randomize the greedy selection
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        adj = [set() for _ in range(n)]
        current_edges = []
        
        # Helper to check if adding (u,v) creates a C4
        def creates_c4(u, v):
            # A C4 is formed if there exists w, x such that
            # (u,w), (w,x), (x,v) are in the graph
            # and u,w,x,v are distinct.
            # As analyzed, distinctness is largely handled by the graph structure
            # and the fact that (u,v) is not yet in the graph.
            for w in adj[u]:
                for x in adj[w]:
                    if x == v:
                        continue
                    if x in adj[v]:
                        return True
            return False

        for u, v in shuffled_edges:
            if not creates_c4(u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v, 0])
                
        if len(current_edges) > max_count:
            max_count = len(current_edges)
            best_edges = current_edges[:]

    return best_edges
