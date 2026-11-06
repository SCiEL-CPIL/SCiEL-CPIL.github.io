import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115

    def solve_single_seed(seed):
        random.seed(seed)
        
        # Initialize with random graph
        edges_set = set()
        while len(edges_set) < target_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                u, v = min(u, v), max(u, v)
                edges_set.add((u, v))
        
        edge_list = list(edges_set)
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        books = [0] * len(edge_list)
        for i, (u, v) in enumerate(edge_list):
            books[i] = (adj[u] & adj[v]).bit_count()
            
        current_max = max(books)
        best_max = current_max
        best_edges = edges_set.copy()
        
        T = 100.0
        cooling = 0.9995
        min_T = 0.01
        
        # Run SA
        while time.time() < deadline - 5 and T > min_T:
            # Pick edge to remove
            idx_rem = random.randint(0, len(edge_list) - 1)
            e_rem = edge_list[idx_rem]
            u1, v1 = e_rem
            
            # Pick edge to add (non-edge)
            e_add = None
            for _ in range(30):
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2: continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) not in edges_set:
                    e_add = (u2, v2)
                    break
            
            if e_add is None:
                continue
                
            u2, v2 = e_add
            
            # Simulate Move
            adj_new = adj[:]
            adj_new[u1] &= ~(1 << v1)
            adj_new[v1] &= ~(1 << u1)
            adj_new[u2] |= (1 << v2)
            adj_new[v2] |= (1 << u2)
            
            # Update edge list (swap)
            new_edge_list = edge_list[:]
            new_edge_list[idx_rem] = e_add
            
            # Recalculate books
            new_books = [0] * len(new_edge_list)
            for i, (u, v) in enumerate(new_edge_list):
                new_books[i] = (adj_new[u] & adj_new[v]).bit_count()
                
            new_max = max(new_books)
            
            # Accept/Reject
            delta = new_max - current_max
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                accept = random.random() < prob
            
            if accept:
                adj = adj_new
                edge_list = new_edge_list
                edges_set.discard(e_rem)
                edges_set.add(e_add)
                
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_edges = edges_set.copy()
                    
            T *= cooling
            
        return best_max, best_edges

    global_best_max = 999
    global_best_edges = None
    
    # Run multiple seeds
    for seed in range(50):
        if time.time() > deadline - 2:
            break
            
        max_b, edges = solve_single_seed(seed)
        
        if max_b < global_best_max:
            global_best_max = max_b
            global_best_edges = edges
            
        if global_best_max <= 7:
            break

    if global_best_edges is None:
        # Fallback: K20,20 + 1 edge
        edges = set()
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        edges.add((0, 1))
        global_best_edges = edges
        global_best_max = 20

    return [[u, v] for (u, v) in sorted(global_best_edges)]
