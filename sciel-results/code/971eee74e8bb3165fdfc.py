import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    N = 16
    
    # Initialize Grötzsch graph
    # Vertices 0-10
    adj = [0] * N
    
    def add_edge(u, v):
        if u == v: return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Grötzsch Edges
    # Cycle 0-4
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(2, 3)
    add_edge(3, 4)
    add_edge(4, 0)
    
    # Mycielski
    add_edge(0, 6)
    add_edge(1, 5)
    add_edge(1, 7)
    add_edge(2, 6)
    add_edge(2, 8)
    add_edge(3, 7)
    add_edge(3, 9)
    add_edge(4, 8)
    add_edge(4, 5)
    add_edge(0, 9)
    
    # Center
    for i in range(5, 10):
        add_edge(i, 10)
        
    # Vertices 11-15 isolated
    
    # All possible edges
    all_edges = []
    for i in range(N):
        for j in range(i+1, N):
            all_edges.append((i,j))
            
    def is_addable(u, v, adj_state):
        # Check if u and v share a neighbor
        return (adj_state[u] & adj_state[v]) == 0

    def greedy_expand(adj_state):
        # Returns new adj_state with max edges added greedily
        # We need to track which edges are present to avoid re-adding
        # Or just check if bit is set.
        
        # Copy state
        new_adj = adj_state[:]
        
        # To make it deterministic but varied, we shuffle edges
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check if already present
            if (new_adj[u] >> v) & 1:
                continue
            # Check triangle-free
            if is_addable(u, v, new_adj):
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
        return new_adj

    def count_edges(adj_state):
        count = 0
        for i in range(N):
            count += bin(adj_state[i]).count('1')
        return count // 2

    best_adj = adj[:]
    best_count = count_edges(best_adj)
    
    # Local Search
    # 1. Greedy expansion from best
    # 2. Perturbation + Greedy
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try greedy from best
        candidate = greedy_expand(best_adj)
        c_count = count_edges(candidate)
        
        if c_count > best_count:
            best_count = c_count
            best_adj = candidate
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Perturbation: Remove random edges from best, then greedy
        if no_improve_count > 10 or random.random() < 0.2:
            perturbed = best_adj[:]
            # Remove 5-10 random edges
            num_remove = random.randint(5, 10)
            for _ in range(num_remove):
                # Pick a random edge present in perturbed
                edges_present = []
                for i in range(N):
                    for j in range(i+1, N):
                        if (perturbed[i] >> j) & 1:
                            edges_present.append((i,j))
                if not edges_present: break
                u, v = random.choice(edges_present)
                perturbed[u] &= ~(1 << v)
                perturbed[v] &= ~(1 << u)
                
            candidate = greedy_expand(perturbed)
            c_count = count_edges(candidate)
            
            if c_count > best_count:
                best_count = c_count
                best_adj = candidate
                no_improve_count = 0

    # Convert best_adj to list of edges
    result = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    return result
