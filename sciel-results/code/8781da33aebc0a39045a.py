import numpy as np
import time
import random
from collections import defaultdict

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0
    
    def build_bipartite_init():
        """Create a balanced bipartite-like structure with exactly m edges."""
        # Split into two groups of 20
        left = list(range(20))
        right = list(range(20, 40))
        
        # Start with complete bipartite K20,20 = 400 edges
        edges = set()
        for u in left:
            for v in right:
                edges.add((u, v))
        
        # Need 1 more edge to reach 401
        # Add an edge within one partition
        edges.add((0, 1))
        
        return edges
    
    def compute_books(A):
        """Compute book sizes for all edges using matrix multiplication."""
        A2 = A.astype(np.int32) @ A.astype(np.int32)
        np.fill_diagonal(A2, 0)
        # Book size of edge (u,v) is A2[u,v] if A[u,v]=1
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    books[(i,j)] = A2[i,j]
        return books
    
    def targeted_local_search(edges, A, max_iter=5000):
        """Local search that targets edges with high book sizes."""
        books = compute_books(A)
        max_b = max(books.values()) if books else 0
        sum_sq = sum(b*b for b in books.values())
        
        # Sort edges by book size descending
        sorted_edges = sorted(books.keys(), key=lambda e: books[e], reverse=True)
        
        no_improve = 0
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if no_improve > 1000:
                break
                
            # Try to improve by modifying high-book edges
            improved = False
            for rem_edge in sorted_edges[:20]:  # Only try top 20 high-book edges
                if time.time() > deadline - 5:
                    break
                if rem_edge not in edges:
                    continue
                    
                ru, rv = rem_edge
                edges.discard(rem_edge)
                A[ru, rv] = 0
                A[rv, ru] = 0
                
                # Find a non-edge to add that minimizes increase in max book
                best_add = None
                best_new_max = float('inf')
                best_new_sum = float('inf')
                
                # Sample some candidate non-edges
                candidates = []
                for _ in range(50):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        e = (min(u,v), max(u,v))
                        if e not in edges:
                            candidates.append(e)
                
                for add_edge in candidates:
                    au, av = add_edge
                    A[au, av] = 1
                    A[av, au] = 1
                    
                    # Quick estimate of new books
                    A2 = A.astype(np.int32) @ A.astype(np.int32)
                    np.fill_diagonal(A2, 0)
                    
                    new_max = 0
                    new_sum = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            if A[i,j] == 1:
                                b = A2[i,j]
                                new_max = max(new_max, b)
                                new_sum += b*b
                    
                    if new_max < best_new_max or (new_max == best_new_max and new_sum < best_new_sum):
                        best_new_max = new_max
                        best_new_sum = new_sum
                        best_add = add_edge
                    
                    # Undo
                    A[au, av] = 0
                    A[av, au] = 0
                
                if best_add is not None and best_new_max < max_b:
                    au, av = best_add
                    edges.add(best_add)
                    A[au, av] = 1
                    A[av, au] = 1
                    max_b = best_new_max
                    sum_sq = best_new_sum
                    no_improve = 0
                    improved = True
                    # Recompute books and re-sort
                    books = compute_books(A)
                    sorted_edges = sorted(books.keys(), key=lambda e: books[e], reverse=True)
                else:
                    # Revert
                    A[ru, rv] = 1
                    A[rv, ru] = 1
                    edges.add(rem_edge)
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return max_b, sum_sq
    
    best_edges = None
    best_max_b = float('inf')
    best_sum_s = float('inf')
    
    # Multiple restarts with different initial structures
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            # Start with bipartite + 1
            edges = build_bipartite_init()
        elif restart == 1:
            # Slightly perturbed bipartite
            edges = build_bipartite_init()
            # Remove a random bipartite edge and add within-partition
            bip_edges = [(u,v) for u,v in edges if u < 20 and v >= 20]
            if bip_edges:
                rem = random.choice(bip_edges)
                edges.discard(rem)
                # Add within left or right
                side = random.choice([0,1])
                if side == 0:
                    u, v = random.sample(range(20), 2)
                else:
                    u, v = random.sample(range(20,40), 2)
                edges.add((min(u,v), max(u,v)))
        else:
            # Random graph
            edges = set()
            while len(edges) < m:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    edges.add((min(u,v), max(u,v)))
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Evaluate
        books = compute_books(A)
        max_b = max(books.values()) if books else 0
        sum_s = sum(b*b for b in books.values())
        
        if max_b < best_max_b or (max_b == best_max_b and sum_s < best_sum_s):
            best_max_b = max_b
            best_sum_s = sum_s
            best_edges = set(edges)
        
        # Run targeted local search
        max_b, sum_s = targeted_local_search(edges, A, max_iter=3000)
        
        if max_b < best_max_b or (max_b == best_max_b and sum_s < best_sum_s):
            best_max_b = max_b
            best_sum_s = sum_s
            best_edges = set(edges)
        
        # If we still have time, do some simulated annealing
        if time.time() < deadline - 15:
            T = 1.0
            T_min = 0.01
            no_improve = 0
            max_no_improve = 2000
            
            while time.time() < deadline - 5:
                if no_improve > max_no_improve:
                    break
                    
                # Random swap
                rem_edge = None
                for _ in range(20):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        e = (min(u,v), max(u,v))
                        if e in edges:
                            rem_edge = e
                            break
                if rem_edge is None:
                    continue
                
                ru, rv = rem_edge
                edges.discard(rem_edge)
                A[ru, rv] = 0
                A[rv, ru] = 0
                
                add_edge = None
                for _ in range(20):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        e = (min(u,v), max(u,v))
                        if e not in edges:
                            add_edge = e
                            break
                if add_edge is None:
                    edges.add(rem_edge)
                    A[ru, rv] = 1
                    A[rv, ru] = 1
                    continue
                
                au, av = add_edge
                edges.add(add_edge)
                A[au, av] = 1
                A[av, au] = 1
                
                books = compute_books(A)
                new_max_b = max(books.values()) if books else 0
                new_sum_s = sum(b*b for b in books.values())
                
                if new_max_b < max_b or (new_max_b == max_b and new_sum_s < sum_s):
                    max_b = new_max_b
                    sum_s = new_sum_s
                    no_improve = 0
                    if new_max_b < best_max_b or (new_max_b == best_max_b and new_sum_s < best_sum_s):
                        best_max_b = new_max_b
                        best_sum_s = new_sum_s
                        best_edges = set(edges)
                else:
                    delta = (new_max_b - max_b) * 100 + (new_sum_s - sum_s)
                    if random.random() < np.exp(-delta / T):
                        max_b = new_max_b
                        sum_s = new_sum_s
                        no_improve += 1
                    else:
                        # Revert
                        edges.discard(add_edge)
                        edges.add(rem_edge)
                        A[au, av] = 0
                        A[av, au] = 0
                        A[ru, rv] = 1
                        A[rv, ru] = 1
                        no_improve += 1
                
                T = max(T_min, T * 0.999)
    
    return [[u, v] for u, v in best_edges]
