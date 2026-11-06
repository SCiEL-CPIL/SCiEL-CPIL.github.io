import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Use bitmask representation for adjacency
    # Each vertex has a 40-bit mask of its neighbors
    def create_bitmask_adj(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def count_common_neighbors(adj, u, v):
        # Count common neighbors using bit operations
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    def get_max_book(edge_set, adj=None):
        if not edge_set:
            return 0
        if adj is None:
            adj = create_bitmask_adj(edge_set)
        
        max_book = 0
        for u, v in edge_set:
            count = count_common_neighbors(adj, u, v)
            if count > max_book:
                max_book = count
        return max_book
    
    def get_book_sizes(edge_set, adj=None):
        if adj is None:
            adj = create_bitmask_adj(edge_set)
        
        books = {}
        for u, v in edge_set:
            count = count_common_neighbors(adj, u, v)
            books[(u, v)] = count
        return books, adj
    
    # Initial structure: K_{20,20} + 1 internal edge
    # Part A: 0..19, Part B: 20..39
    # All edges between A and B (400 edges) + one edge within A
    def create_initial_graph(seed_val):
        random.seed(seed_val)
        edges = set()
        
        # Add all bipartite edges
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        
        # Add one internal edge to make 401 edges
        # Randomly choose an internal edge
        a1 = random.randint(0, 19)
        a2 = random.randint(0, 19)
        while a2 == a1:
            a2 = random.randint(0, 19)
        edges.add((min(a1, a2), max(a1, a2)))
        
        return edges
    
    def optimize(edges_set, time_limit):
        current_edges = set(edges_set)
        books, adj = get_book_sizes(current_edges)
        current_max = max(books.values()) if books else 0
        
        best_edges = set(current_edges)
        best_max = current_max
        
        # Simulated annealing with bitmask acceleration
        temp = 3.0
        cooling = 0.9998
        sa_start = time.time()
        
        # Precompute all possible edges for faster sampling
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        while time.time() < time_limit:
            if not current_edges:
                break
            
            # Find edge with max book
            max_b = -1
            worst_edge = None
            for e, b in books.items():
                if b > max_b:
                    max_b = b
                    worst_edge = e
            
            if worst_edge is None:
                break
            
            u, v = worst_edge
            
            # Generate a candidate replacement edge
            found_new_edge = False
            for _ in range(100):
                a, b = random.choice(all_edges)
                new_edge = (a, b)
                if new_edge not in current_edges and new_edge != worst_edge:
                    found_new_edge = True
                    break
            
            if not found_new_edge:
                break
            
            # Perform swap
            current_edges.remove(worst_edge)
            current_edges.add(new_edge)
            
            # Recalculate books efficiently
            new_adj = create_bitmask_adj(current_edges)
            new_books = {}
            new_max = 0
            for eu, ev in current_edges:
                count = count_common_neighbors(new_adj, eu, ev)
                new_books[(eu, ev)] = count
                if count > new_max:
                    new_max = count
            
            delta = new_max - current_max
            
            if delta <= 0 or random.random() < pow(2.718, -delta/temp):
                current_max = new_max
                books = new_books
                adj = new_adj
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(current_edges)
            else:
                # Revert
                current_edges.remove(new_edge)
                current_edges.add(worst_edge)
            
            temp *= cooling
        
        return best_edges, best_max
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts with different seeds
    seed = 42
    while time.time() < deadline - 10.0:
        initial_edges = create_initial_graph(seed)
        optimized_edges, score = optimize(initial_edges, deadline - 5.0)
        
        if score < best_score:
            best_score = score
            best_edges = optimized_edges
            # If we reach theoretical lower bound (7), stop
            if best_score <= 7:
                break
        
        seed += 1
    
    # Final polish pass: if we have time, do more targeted optimization
    if best_edges is not None and time.time() < deadline - 3.0:
        polish_edges, polish_score = optimize(best_edges, deadline - 1.0)
        if polish_score < best_score:
            best_score = polish_score
            best_edges = polish_edges

    if best_edges is None:
        # Fallback: K20,20 + 1 edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    # Convert to list of lists
    return [[u, v] for u, v in best_edges]
