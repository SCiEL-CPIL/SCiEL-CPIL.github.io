import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 20
    
    def count_edges(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    def is_valid_addition(adj_mask, u, v):
        # Check if adding (u, v) violates K2,3
        # 1. Common neighbors of u and v must be <= 2
        if (adj_mask[u] & adj_mask[v]).bit_count() > 2:
            return False
            
        # 2. For all w != u, v:
        #    If (v, w) in E, then v becomes common neighbor of u and w.
        #    Old common neighbors of u, w: adj_mask[u] & adj_mask[w]
        #    New count = old + 1. Must be <= 2 => old <= 1.
        #    If (u, w) in E, then u becomes common neighbor of v and w.
        #    Old common neighbors of v, w: adj_mask[v] & adj_mask[w]
        #    New count = old + 1. Must be <= 2 => old <= 1.
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mask[v] & (1 << w):
                if (adj_mask[u] & adj_mask[w]).bit_count() > 1:
                    return False
            if adj_mask[u] & (1 << w):
                if (adj_mask[v] & adj_mask[w]).bit_count() > 1:
                    return False
        return True

    def greedy_construct():
        adj_mask = [0] * n
        edges = list(combinations(range(n), 2))
        random.shuffle(edges)
        for u, v in edges:
            if not (adj_mask[u] & (1 << v)):
                if is_valid_addition(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def local_search(adj_mask, time_limit):
        current_edges = count_edges(adj_mask)
        best_edges = current_edges
        best_adj = adj_mask[:]
        
        # List of all edges
        all_edges = list(combinations(range(n), 2))
        
        # Precompute missing edges? No, check on the fly.
        
        steps = 0
        while time.time() < time_limit:
            # Try to add a random missing edge
            # To make it faster, we can keep a list of missing edges, but updating it is costly.
            # Just pick random pairs.
            
            # Pick a random edge
            u, v = random.choice(all_edges)
            
            if not (adj_mask[u] & (1 << v)):
                # Missing edge
                if is_valid_addition(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj_mask[:]
            else:
                # Existing edge
                # With some probability, remove it to explore
                if random.random() < 0.1:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    current_edges -= 1
                    
            steps += 1
            # Optional: periodic check if we are stuck
            # If no improvement for X steps, maybe restart local search?
            # For now, just run.
            
        return best_adj, best_edges

    best_edges = 0
    best_adj_mask = [0] * n
    
    # Run multiple restarts
    while time.time() < deadline - 5.0:
        # 1. Greedy construction
        adj_mask = greedy_construct()
        current_edges = count_edges(adj_mask)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_mask = adj_mask[:]
            
        # 2. Local search
        time_left = deadline - time.time() - 2.0
        if time_left > 1.0:
            improved_adj, improved_edges = local_search(adj_mask, time.time() + time_left)
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_adj_mask = improved_adj[:]
                
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj_mask[u] & (1 << v):
                edges.append([u, v])
                
    return edges
