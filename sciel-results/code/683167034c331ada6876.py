import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 40
    m = 401
    
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    def init_random_graph():
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i in range(m):
            edges.add(all_pairs[i])
        return edges

    def compute_max_book(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        max_book = 0
        for u, v in edges:
            common = bin(adj[u] & adj[v]).count('1')
            if common > max_book:
                max_book = common
        return max_book

    best_score = float('inf')
    best_edges = None
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
            
        edges = init_random_graph()
        current_max = compute_max_book(edges)
        
        # Local Search
        T = 5.0 # Temperature
        T_min = 0.1
        cooling = 0.99
        stuck_count = 0
        
        while time.time() < deadline - 5.0:
            # Pick random edge to remove
            if not edges:
                break
            old_edge = random.choice(list(edges))
            u1, v1 = old_edge
            
            # Pick random non-edge to add
            new_edge = None
            for _ in range(100):
                cand = random.choice(all_pairs)
                if cand not in edges:
                    new_edge = cand
                    break
            if new_edge is None:
                continue
            
            u2, v2 = new_edge
            
            # Perform Swap
            edges.remove(old_edge)
            edges.add(new_edge)
            
            # Recalc
            new_max = compute_max_book(edges)
            
            # Acceptance criterion
            if new_max < current_max:
                current_max = new_max
                stuck_count = 0
            else:
                # Revert
                edges.add(old_edge)
                edges.remove(new_edge)
                stuck_count += 1
                
                # If stuck for too long, restart or cool down faster
                if stuck_count > 100:
                    stuck_count = 0
                    # Maybe break and restart?
                    # For now, just continue with lower T
                    T = max(T * 0.9, T_min)
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        if current_max < best_score:
            best_score = current_max
            best_edges = list(edges)
            
    if best_edges is None:
        # Fallback
        edges = []
        for u in range(20):
            for v in range(20, 40):
                if u == 0 and 20 <= v <= 28:
                    continue
                edges.append([u, v])
        for i in range(1, 11):
            edges.append([0, i])
        best_edges = edges
        
    return best_edges
