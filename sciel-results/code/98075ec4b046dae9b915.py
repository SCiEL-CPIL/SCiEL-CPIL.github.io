"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 40
    E = 401
    
    def build_adj_matrix(edges):
        """Build adjacency matrix from edge set."""
        adj = np.zeros((N, N), dtype=np.uint8)
        for (u, v) in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def compute_books_np(adj):
        """Compute book sizes using numpy. Book of edge (u,v) = number of common neighbors."""
        books = np.zeros((N, N), dtype=np.int32)
        # For each pair (u,v) that is an edge, book = sum of adj[u] * adj[v] over all w
        # This is the (u,v) entry of adj @ adj
        common_neighbors = adj @ adj
        # Only consider actual edges
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v] == 1:
                    books[u, v] = common_neighbors[u, v]
                    books[v, u] = common_neighbors[u, v]
        return books
    
    def get_max_book(books):
        """Get the maximum book size."""
        max_val = 0
        for u in range(N):
            for v in range(u+1, N):
                if books[u, v] > max_val:
                    max_val = books[u, v]
        return max_val
    
    def get_high_book_edges(books, threshold):
        """Get edges with book size >= threshold."""
        high_edges = []
        for u in range(N):
            for v in range(u+1, N):
                if books[u, v] >= threshold:
                    high_edges.append((u, v))
        return high_edges
    
    def get_non_edges(adj):
        """Get list of non-edges."""
        non_edges = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v] == 0:
                    non_edges.append((u, v))
        return non_edges
    
    def simulated_annealing(edges, max_iter=5000, start_temp=10.0, end_temp=0.1):
        """Simulated annealing to minimize max book size."""
        adj = build_adj_matrix(edges)
        books = compute_books_np(adj)
        current_max = get_max_book(books)
        best_max = current_max
        best_edges = edges.copy()
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max_iter)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            temp *= cooling
            
            # Get edges with high book size
            high_edges = get_high_book_edges(books, current_max)
            if not high_edges:
                # Try any edge
                all_edges = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u, v] == 1]
                high_edges = all_edges
            
            if not high_edges:
                break
            
            # Pick a random high-book edge to remove
            u, v = random.choice(high_edges)
            
            # Pick a random non-edge to add
            non_edges = get_non_edges(adj)
            if not non_edges:
                break
            a, b = random.choice(non_edges)
            
            # Perform swap
            edges.remove((u, v))
            adj[u, v] = 0
            adj[v, u] = 0
            edges.add((a, b))
            adj[a, b] = 1
            adj[b, a] = 1
            
            # Recompute books
            books = compute_books_np(adj)
            new_max = get_max_book(books)
            
            # Accept or reject
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges = edges.copy()
            else:
                # Reject with probability based on temperature
                delta = new_max - current_max
                if delta == 0:
                    # Accept neutral moves with some probability
                    if random.random() < 0.1:
                        current_max = new_max
                    else:
                        # Revert
                        edges.remove((a, b))
                        adj[a, b] = 0
                        adj[b, a] = 0
                        edges.add((u, v))
                        adj[u, v] = 1
                        adj[v, u] = 1
                        books = compute_books_np(adj)
                else:
                    # Reject worsening moves (unless very hot)
                    if temp > 1.0 and random.random() < np.exp(-delta / temp):
                        current_max = new_max
                    else:
                        # Revert
                        edges.remove((a, b))
                        adj[a, b] = 0
                        adj[b, a] = 0
                        edges.add((u, v))
                        adj[u, v] = 1
                        adj[v, u] = 1
                        books = compute_books_np(adj)
        
        return best_edges, best_max
    
    def greedy_local_search(edges, max_iter=2000):
        """Greedy local search: only accept improvements."""
        adj = build_adj_matrix(edges)
        books = compute_books_np(adj)
        current_max = get_max_book(books)
        best_max = current_max
        best_edges = edges.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Get edges with maximum book size
            high_edges = get_high_book_edges(books, current_max)
            if not high_edges:
                break
            
            improved = False
            random.shuffle(high_edges)
            
            for (u, v) in high_edges:
                # Try to find a better swap
                non_edges = get_non_edges(adj)
                if not non_edges:
                    break
                
                random.shuffle(non_edges)
                for (a, b) in non_edges[:50]:  # Try up to 50 candidates
                    # Compute new book for (a,b)
                    common_ab = int(adj[a] @ adj[b])
                    if common_ab >= current_max:
                        continue
                    
                    # Perform swap
                    edges.remove((u, v))
                    adj[u, v] = 0
                    adj[v, u] = 0
                    edges.add((a, b))
                    adj[a, b] = 1
                    adj[b, a] = 1
                    
                    books = compute_books_np(adj)
                    new_max = get_max_book(books)
                    
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                        if new_max < best_max:
                            best_max = new_max
                            best_edges = edges.copy()
                        break
                    else:
                        # Revert
                        edges.remove((a, b))
                        adj[a, b] = 0
                        adj[b, a] = 0
                        edges.add((u, v))
                        adj[u, v] = 1
                        adj[v, u] = 1
                        books = compute_books_np(adj)
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_edges, best_max
    
    def initial_graph():
        """Generate initial graph using balanced bipartite + perturbation."""
        # Start with K20,20 (400 edges) plus one more edge
        edges = set()
        for i in range(20):
            for j in range(20, 40):
                edges.add((i, j))
        # Add one more edge within one partition
        edges.add((0, 1))
        return edges
    
    # Try multiple strategies
    best_edges = None
    best_max = 999
    
    # Strategy 1: Start from balanced bipartite + perturbation
    if time.time() < deadline - 30:
        edges = initial_graph()
        edges, cur_max = greedy_local_search(edges, max_iter=5000)
        if cur_max < best_max:
            best_max = cur_max
            best_edges = edges.copy()
    
    # Strategy 2: Simulated annealing from random starts
    num_sa_restarts = 5
    for r in range(num_sa_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initial graph
        edges = set()
        all_possible = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_possible)
        edges = set(all_possible[:E])
        
        edges, cur_max = simulated_annealing(edges, max_iter=3000, start_temp=5.0, end_temp=0.5)
        if cur_max < best_max:
            best_max = cur_max
            best_edges = edges.copy()
    
    # Strategy 3: More aggressive local search on best
    if best_edges is not None and time.time() < deadline - 10:
        best_edges, best_max = greedy_local_search(best_edges, max_iter=10000)
    
    # Final refinement with simulated annealing
    if best_edges is not None and time.time() < deadline - 5:
        best_edges, best_max = simulated_annealing(best_edges, max_iter=5000, start_temp=2.0, end_temp=0.1)
    
    if best_edges is None:
        best_edges = initial_graph()
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for (u, v) in best_edges]
    return result

# EVOLVE-BLOCK-END
