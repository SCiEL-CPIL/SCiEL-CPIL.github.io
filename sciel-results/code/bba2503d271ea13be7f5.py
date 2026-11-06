import time
import math
import random

def construct():
    n = 16
    edges_needed = 65
    
    # Initial construction: K_{6,10} + 5 disjoint edges in the size-10 part
    # Part A: 0-5 (size 6)
    # Part B: 6-15 (size 10)
    # Edges: All (A, B) pairs = 6*10 = 60 edges.
    # Need 5 more edges. Add matching in B.
    # B vertices: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    # Matching: (6,7), (8,9), (10,11), (12,13), (14,15)
    
    adj = [0] * n
    deg = [0] * n
    
    def add_edge(u, v):
        if u == v: return
        if (adj[u] >> v) & 1: return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    def remove_edge(u, v):
        if u == v: return
        if not ((adj[u] >> v) & 1): return
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

    # Build K_{6,10}
    for u in range(6):
        for v in range(6, 16):
            add_edge(u, v)
            
    # Add matching in B
    b_vertices = list(range(6, 16))
    random.shuffle(b_vertices)
    for i in range(0, 10, 2):
        add_edge(b_vertices[i], b_vertices[i+1])
        
    # Verify edge count
    current_edges = sum(deg) // 2
    if current_edges != 65:
        # Fallback to seed if something went wrong
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}

    def get_score():
        max_sum = 0
        # Iterate over all triples u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    for w in range(v + 1, n):
                        if (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = get_score()
    best_score = current_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    T = 1.0
    cooling_rate = 0.99999
    
    # Precompute list of edges and non-edges for faster swap selection?
    # Or just pick random pairs.
    # Random pairs: 16*15/2 = 120 possible edges.
    # Picking random u, v and checking if edge is fast.
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v and ((adj[u] >> v) & 1):
                break
        
        # Pick a random non-edge to add
        while True:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x < y and not ((adj[x] >> y) & 1):
                break
        
        # Perform swap
        remove_edge(u, v)
        add_edge(x, y)
        
        new_score = get_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            # Check acceptance
            delta = new_score - current_score
            if delta > 0:
                accept_prob = math.exp(-delta / T)
                if random.random() > accept_prob:
                    # Reject
                    remove_edge(x, y)
                    add_edge(u, v)
                    new_score = current_score # Restore score variable consistency if needed, but we just revert state
            # If delta == 0, accept (or reject, doesn't matter much, usually accept to explore)
            # We accepted if new_score < current_score or (delta==0 and we didn't reject).
            # Wait, the logic above:
            # If new_score < current_score: accept and update.
            # Else: 
            #   If delta > 0 and random > prob: reject (revert).
            #   Else: accept (keep state).
        
        T *= cooling_rate
    
    # Convert best_adj to edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
