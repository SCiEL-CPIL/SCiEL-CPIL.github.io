<<<<<<< SEARCH
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
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
            
    needed_in_B = target_edges - (a_size * b_size) # 122
    
    def try_construct_B(max_deg):
        B_edges = []
        degree = {v: 0 for v in B}
        adj = {v: set() for v in B}
        
        # Generate pairs and shuffle
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        
        for u, v in pairs:
            if len(B_edges) == needed_in_B:
                break
            
            # Check triangle constraint (common neighbor in B)
            if adj[u] & adj[v]:
                continue
                
            # Check degree constraint
            if degree[u] < max_deg and degree[v] < max_deg:
                B_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
        if len(B_edges) == needed_in_B:
            return B_edges
        return None

    B_edges = None
    # Try to find a solution with max degree 9 (allows triangle-free)
    # max_deg 8 is too tight for 122 edges in 31 vertices (max 124, but greedy might fail)
    # max_deg 9 is safe (max 139)
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 200:
        B_edges = try_construct_B(9)
        if B_edges:
            break
        attempts += 1
        
    # Fallback to 10 if 9 fails (should not happen)
    if not B_edges:
        B_edges = try_construct_B(10)
        
    if not B_edges:
        # Last resort: random edges, likely high score
        pairs = list(itertools.combinations(B, 2))
        random.shuffle(pairs)
        B_edges = pairs[:needed_in_B]

    for u, v in B_edges:
        edges.add((u, v))
        
    result = [[u, v] for u, v in edges]
    
    return result
=======
import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    a_size = 9
    b_size = 31
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    # Build initial adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Add all edges between A and B
    for a in A:
        for b in B:
            adj[a, b] = 1
            adj[b, a] = 1
    
    # Count current edges
    current_edges = sum(adj) // 2
    needed_in_B = target_edges - current_edges
    
    # Helper to compute book sizes
    def compute_book_sizes(adj_matrix):
        # Book size for edge (u,v) = number of common neighbors
        # = (adj_matrix[u] & adj_matrix[v]).sum()
        # Compute all pairwise common neighbors
        # Using matrix multiplication: (A @ A)[u,v] gives common neighbors for u,v
        A_sq = adj_matrix @ adj_matrix
        # Book size for edge (u,v) is A_sq[u,v] when adj[u,v]=1
        # We want the max over all edges
        # Create mask of existing edges
        edge_mask = adj_matrix > 0
        # Get book sizes only for existing edges
        book_sizes = np.where(edge_mask, A_sq, 0)
        return book_sizes
    
    # Initial B edges: use a balanced approach
    B_edges = []
    degree = {v: 0 for v in B}
    
    # Generate all possible B pairs
    all_B_pairs = list(itertools.combinations(B, 2))
    random.shuffle(all_B_pairs)
    
    # Greedy: add edges that minimize max book size
    for u, v in all_B_pairs:
        if len(B_edges) == needed_in_B:
            break
        if degree[u] < 10 and degree[v] < 10:
            B_edges.append((u, v))
            degree[u] += 1
            degree[v] += 1
            adj[u, v] = 1
            adj[v, u] = 1
    
    # If we don't have enough, add more without degree constraint
    if len(B_edges) < needed_in_B:
        for u, v in all_B_pairs:
            if len(B_edges) == needed_in_B:
                break
            if adj[u, v] == 0:
                B_edges.append((u, v))
                adj[u, v] = 1
                adj[v, u] = 1
    
    # Now optimize using edge swaps
    # Compute initial book sizes
    book_sizes = compute_book_sizes(adj)
    max_book = book_sizes.max()
    
    # List of all possible edges (u,v) where u < v
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Set of current edges as a set for fast lookup
    current_edge_set = set()
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v] == 1:
                current_edge_set.add((u, v))
    
    # Simulated annealing / local search
    best_max_book = max_book
    best_adj = adj.copy()
    best_book_sizes = book_sizes.copy()
    
    temperature = 100.0
    cooling_rate = 0.999
    min_temp = 0.1
    
    # Precompute adjacency lists for faster common neighbor computation
    adj_lists = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u, v] == 1 and u != v:
                adj_lists[u].add(v)
    
    # Function to compute book size for a specific edge
    def get_book_size(edge_u, edge_v):
        return len(adj_lists[edge_u] & adj_lists[edge_v])
    
    # Local search: try swapping edges to reduce max book
    # Focus on edges with high book sizes
    high_book_edges = [(u, v) for u in range(n) for v in range(u+1, n) 
                       if adj[u, v] == 1 and book_sizes[u, v] > 6]
    
    # Iterate through high-book edges and try to reduce their book size
    for _ in range(5000):  # Limit iterations
        if time.time() > deadline - 15.0:
            break
        
        # Find the edge with maximum book size
        max_u, max_v = np.unravel_index(book_sizes.argmax(), book_sizes.shape)
        if max_u > max_v:
            max_u, max_v = max_v, max_u
        current_max = book_sizes[max_u, max_v]
        
        if current_max <= 6:
            break
        
        # Try to remove this edge and add a different one
        # Find a non-edge that would have low book size
        removed_u, removed_v = max_u, max_v
        
        # Try adding a random non-edge
        for _attempt in range(20):
            # Pick a random non-edge
            u1 = random.randint(0, n-1)
            u2 = random.randint(0, n-1)
            if u1 == u2 or adj[u1, u2] == 1:
                continue
            if (u1, u2) == (removed_u, removed_v) or (u2, u1) == (removed_u, removed_v):
                continue
            
            # Check if this swap is valid: removing (removed_u, removed_v) and adding (u1, u2)
            # must keep total edges at 401
            
            # Compute new book sizes after swap
            # Remove (removed_u, removed_v): book sizes of edges incident to removed_u or removed_v may decrease
            # Add (u1, u2): book sizes of edges incident to u1 or u2 may increase
            
            # Quick estimate: what's the book size of the new edge?
            new_edge_book = len(adj_lists[u1] & adj_lists[u2])
            
            # If the new edge's book size is less than current max, accept
            if new_edge_book < current_max:
                # Perform the swap
                adj[removed_u, removed_v] = 0
                adj[removed_v, removed_u] = 0
                adj[u1, u2] = 1
                adj[u2, u1] = 1
                
                adj_lists[removed_u].discard(removed_v)
                adj_lists[removed_v].discard(removed_u)
                adj_lists[u1].add(u2)
                adj_lists[u2].add(u1)
                
                # Recompute book sizes
                book_sizes = compute_book_sizes(adj)
                max_book = book_sizes.max()
                
                if max_book < best_max_book:
                    best_max_book = max_book
                    best_adj = adj.copy()
                    best_book_sizes = book_sizes.copy()
                
                break
            elif new_edge_book == current_max and random.random() < temperature / 100.0:
                # Accept with some probability to escape local minima
                adj[removed_u, removed_v] = 0
                adj[removed_v, removed_u] = 0
                adj[u1, u2] = 1
                adj[u2, u1] = 1
                
                adj_lists[removed_u].discard(removed_v)
                adj_lists[removed_v].discard(removed_u)
                adj_lists[u1].add(u2)
                adj_lists[u2].add(u1)
                
                book_sizes = compute_book_sizes(adj)
                max_book = book_sizes.max()
                
                if max_book < best_max_book:
                    best_max_book = max_book
                    best_adj = adj.copy()
                    best_book_sizes = book_sizes.copy()
                
                break
        
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
    
    # Use the best solution found
    if best_max_book < max_book:
        adj = best_adj
    
    # Extract final edges
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v] == 1:
                result.append([u, v])
    
    # Verify we have exactly 401 edges
    if len(result) != target_edges:
        # Fallback to original construction
        adj2 = np.zeros((n, n), dtype=np.int8)
        for a in A:
            for b in B:
                adj2[a, b] = 1
                adj2[b, a] = 1
        
        B_edges2 = []
        degree2 = {v: 0 for v in B}
        all_B_pairs2 = list(itertools.combinations(B, 2))
        random.shuffle(all_B_pairs2)
        
        for u, v in all_B_pairs2:
            if len(B_edges2) == needed_in_B:
                break
            if degree2[u] < 10 and degree2[v] < 10:
                B_edges2.append((u, v))
                degree2[u] += 1
                degree2[v] += 1
                adj2[u, v] = 1
                adj2[v, u] = 1
        
        if len(B_edges2) < needed_in_B:
            for u, v in all_B_pairs2:
                if len(B_edges2) == needed_in_B:
                    break
                if adj2[u, v] == 0:
                    B_edges2.append((u, v))
                    adj2[u, v] = 1
                    adj2[v, u] = 1
        
        result = []
        for u in range(n):
            for v in range(u+1, n):
                if adj2[u, v] == 1:
                    result.append([u, v])
    
    return result
>>>>>>> REPLACE
