import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    needed_in_B = target_edges - (a_size * b_size)
    # 122
    
    # Deterministic greedy for B-B edges
    B_edges = []
    degree = {v: 0 for v in B}
    adj = {v: set() for v in B}
    
    pairs = list(itertools.combinations(B, 2))
    candidates = set(pairs)
    
    # Try to fill with max degree 8
    while len(B_edges) < needed_in_B:
        if time.time() > deadline:
            break
        best_pair = None
        best_score = None
        for u, v in candidates:
            if adj[u] & adj[v]:
                continue
            if degree[u] >= 8 or degree[v] >= 8:
                continue
            score = degree[u] + degree[v]
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (u, v)
        if best_pair is None:
            break
        u, v = best_pair
        B_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        candidates.remove((u, v))
        
    if len(B_edges) < needed_in_B:
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        candidates = set(pairs)
        while len(B_edges) < needed_in_B:
            if time.time() > deadline:
                break
            best_pair = None
            best_score = None
            for u, v in candidates:
                if adj[u] & adj[v]:
                    continue
                if degree[u] >= 9 or degree[v] >= 9:
                    continue
                score = degree[u] + degree[v]
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (u, v)
            if best_pair is None:
                break
            u, v = best_pair
            B_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            adj[u].add(v)
            adj[v].add(u)
            candidates.remove((u, v))
            
    for u, v in B_edges:
        edges.add((u, v))
        
    # Local Search
    def get_max_book(edge_set):
        adj_local = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj_local[u].add(v)
            adj_local[v].add(u)
        max_book = 0
        for u, v in edge_set:
            common = len(adj_local[u] & adj_local[v])
            if common > max_book:
                max_book = common
        return max_book

    current_edges = set(edges)
    current_max_book = get_max_book(current_edges)
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    while time.time() < deadline:
        if not current_edges:
            break
        edge_to_remove = random.choice(list(current_edges))
        
        # Find a valid non-edge
        # To speed up, we can pick random pairs until we find one not in current_edges
        # And not equal to edge_to_remove
        found_swap = False
        for _ in range(5):
            u, v = random.choice(all_pairs)
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            if (u, v) == edge_to_remove or (v, u) == edge_to_remove:
                continue
            
            new_edges = set(current_edges)
            new_edges.remove(edge_to_remove)
            new_edges.add((u, v))
            
            new_max_book = get_max_book(new_edges)
            
            if new_max_book < current_max_book:
                current_edges = new_edges
                current_max_book = new_max_book
                found_swap = True
            break # Only try one candidate per iteration of main loop for simplicity? 
                  # Or loop until found? 
                  # If we break immediately, we only try 1 candidate per step.
                  # If we want to try more, we can loop here.
                  # But get_max_book is the bottleneck. 
                  # Let's try 1 candidate per step. It's fast enough.
        
    result = [[u, v] for u, v in current_edges]
    return result
