import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 20
    # Adjacency matrix
    A = [[0] * n for _ in range(n)]
    edges = []
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Shuffle to avoid bias? Or just iterate.
    # Greedy in fixed order might be suboptimal.
    # We can try multiple random orderings.
    
    best_edges = []
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        A = [[0] * n for _ in range(n)]
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a 4-cycle
            # A 4-cycle u-w-x-v-u exists if there is a path of length 3 from u to v
            # i.e., exists w, x such that A[u][w]=1, A[w][x]=1, A[x][v]=1
            # and u,w,x,v are distinct.
            
            creates_c4 = False
            for w in range(n):
                if A[u][w]:
                    if w == v: continue # simple graph, no self loops, but u!=v
                    for x in range(n):
                        if A[w][x]:
                            if x == u: continue
                            if x == v: continue
                            if w == v: continue # redundant check
                            if A[x][v]:
                                # Check distinctness: u, w, x, v
                                # u != w (since A[u][w] and simple graph implies u!=w? No, A is 0 on diag)
                                # w != x (simple graph)
                                # x != v (checked above)
                                # u != v (given)
                                # Need u != x and w != v
                                if u != x and w != v:
                                    creates_c4 = True
                                    break
                            if creates_c4: break
                    if creates_c4: break
            if creates_c4:
                continue
            
            # Add edge
            A[u][v] = 1
            A[v][u] = 1
            current_edges.append((u, v))
            
        if len(current_edges) > len(best_edges):
            best_edges = current_edges[:]
            
    # Assign colours
    # Since graph is C4-free, any colouring works.
    # We use colours 0..19.
    res = []
    for i, (u, v) in enumerate(best_edges):
        c = i % 20
        res.append([u, v, c])
        
    return res
