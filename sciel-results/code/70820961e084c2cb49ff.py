import random
import time
import numpy as np
import heapq
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Strategy: Deterministic constraint-satisfaction with targeted edge replacement
    # 1. Start with a highly structured quasi-random graph (circulant-based)
    # 2. Identify edges with high book size
    # 3. Systematically replace them with low-impact non-edges
    # 4. Use a priority queue ordered by "book reduction potential"
    
    # Step 1: Construct initial quasi-random graph
    # Use a circulant graph structure: vertex i connected to i±1, i±2, i±3, i±4, i±5, i±6, i±7, i±8, i±9, i±10
    # This gives degree 20, total edges = 40*20/2 = 400. We need 1 more.
    # But circulant with these connections might have high books.
    
    # Better: Use a "quasi-random" construction based on modular arithmetic
    # that minimizes triangles while maintaining high edge count.
    
    # Start with K20,20 (bipartite, 400 edges, 0 triangles)
    # Then add 1 edge within a part, but choose it to minimize the book size.
    # If we add (u,v) within part A, the book size is the number of common neighbors in B.
    # In K20,20, every pair in A has 20 common neighbors (all of B).
    # So book size = 20. That's bad.
    
    # Alternative: Start with a different structure.
    # Let's use a random graph but with a specific seed, then apply the targeted replacement.
    
    rng = np.random.default_rng(123)
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initial random selection
    initial_indices = rng.choice(len(all_edges), size=m, replace=False)
    edges = set(all_edges[i] for i in initial_indices)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    edge_list = list(edges)
    all_possible_set = set(all_edges)
    non_edge_list = list(all_possible_set - edges)
    
    # Precompute C = adj @ adj (book sizes for all pairs)
    C = adj @ adj
    
    def compute_books(edge_list, C):
        """Compute book sizes for all edges in edge_list."""
        books = {}
        for u, v in edge_list:
            books[(u, v)] = C[u, v]
        return books
    
    def get_max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    current_books = compute_books(edge_list, C)
    current_max_book = get_max_book(current_books)
    
    best_max_book = current_max_book
    best_books = current_books.copy()
    best_adj = adj.copy()
    best_edges = set(edge_list)
    
    # Priority queue: (book_size, edge) for edges with high book size
    # We want to replace edges with the highest book size first.
    # But we also need to consider the "cost" of adding a replacement edge.
    
    # Define a scoring function for candidate swaps:
    # For removing edge e_rm and adding edge e_add:
    # - Benefit: reduction in book size of e_rm (and potentially other edges)
    # - Cost: increase in book size of e_add (and potentially other edges)
    # - Net score: benefit - cost, weighted by second-order effects
    
    # To keep it tractable, we'll use a greedy approach:
    # 1. Find the edge with the maximum book size.
    # 2. Find the non-edge with the minimum book size.
    # 3. If the swap reduces the maximum book size (or keeps it the same but reduces the sum of squares), accept it.
    # 4. Repeat until no improvement is possible or time runs out.
    
    # To avoid getting stuck, we'll use a more sophisticated approach:
    # Maintain a priority queue of "hot" edges (high book size) and "cold" non-edges (low book size).
    # For each hot edge, find the best cold non-edge to swap with.
    
    no_improvement_count = 0
    max_iterations = 10000
    
    iteration = 0
    while time.time() < deadline and iteration < max_iterations:
        iteration += 1
        
        # Find hot edges: those with book size >= current_max_book - 1
        hot_edges = [(e, b) for e, b in current_books.items() if b >= current_max_book - 1]
        
        if not hot_edges:
            break
        
        # Sort by book size descending
        hot_edges.sort(key=lambda x: x[1], reverse=True)
        
        # Find cold non-edges: those with book size < current_max_book
        cold_non_edges = []
        for e in non_edge_list:
            u, v = e
            b = C[u, v]
            if b < current_max_book:
                cold_non_edges.append((e, b))
        
        # Sort by book size ascending
        cold_non_edges.sort(key=lambda x: x[1])
        
        best_swap = None
        best_new_max = current_max_book
        best_new_sum_sq = sum(b*b for b in current_books.values())
        
        # Try swapping the hottest edge with the coldest non-edge
        # But also try a few alternatives to avoid local minima
        
        for e_rm, b_rm in hot_edges[:5]:  # Top 5 hottest
            u, v = e_rm
            
            for e_add, b_add in cold_non_edges[:10]:  # Top 10 coldest
                a, b = e_add
                
                # Quick check: if the new edge's book size is too high, skip
                if b_add >= best_new_max:
                    continue
                
                # Simulate swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                # Recalculate C
                C_new = adj @ adj
                
                # Calculate new books for affected edges
                # Only edges incident to u, v, a, b are affected
                affected_edges = set()
                for e in edge_list:
                    if e == (u, v):
                        continue
                    # Check if e is affected by the swap
                    # e is affected if it shares a vertex with (u,v) or (a,b)
                    eu, ev = e
                    if eu in (u, v, a, b) or ev in (u, v, a, b):
                        affected_edges.add(e)
                
                # Also add the new edge (a,b)
                affected_edges.add((a, b))
                
                # Calculate new max and sum_sq
                new_books = current_books.copy()
                # Remove old edge
                if (u, v) in new_books:
                    del new_books[(u, v)]
                
                # Update affected edges
                for e in affected_edges:
                    eu, ev = e
                    new_books[e] = C_new[eu, ev]
                
                new_max = get_max_book(new_books)
                new_sum_sq = sum(b*b for b in new_books.values())
                
                # Revert adj
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0
                
                # Compare
                if new_max < best_new_max or (new_max == best_new_max and new_sum_sq < best_new_sum_sq):
                    best_new_max = new_max
                    best_new_sum_sq = new_sum_sq
                    best_swap = (e_rm, e_add)
        
        if best_swap:
            e_rm, e_add = best_swap
            u, v = e_rm
            a, b = e_add
            
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1
            
            edge_list.remove(e_rm)
            edge_list.append(e_add)
            
            non_edge_list.append(e_rm)
            non_edge_list.remove(e_add)
            
            # Update C
            C = adj @ adj
            
            # Update books
            current_books = compute_books(edge_list, C)
            current_max_book = get_max_book(current_books)
            
            # Update best
            if current_max_book < best_max_book or (current_max_book == best_max_book and best_new_sum_sq < sum(b*b for b in best_books.values())):
                best_max_book = current_max_book
                best_books = current_books.copy()
                best_adj = adj.copy()
                best_edges = set(edge_list)
                
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            # If stuck, do a random perturbation
            if no_improvement_count > 20:
                # Random swap
                if edge_list and non_edge_list:
                    e1 = random.choice(edge_list)
                    e2 = random.choice(non_edge_list)
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    adj[u2, v2] = 1
                    adj[v2, u2] = 1
                    
                    edge_list.remove(e1)
                    edge_list.append(e2)
                    
                    non_edge_list.append(e1)
                    non_edge_list.remove(e2)
                    
                    C = adj @ adj
                    current_books = compute_books(edge_list, C)
                    current_max_book = get_max_book(current_books)
                    
                    if current_max_book < best_max_book:
                        best_max_book = current_max_book
                        best_books = current_books.copy()
                        best_adj = adj.copy()
                        best_edges = set(edge_list)
                    
                    no_improvement_count = 0
    
    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
                
    # Sanity check
    if len(result) != 401:
        # Fallback: K20,20 + 1 edge
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
