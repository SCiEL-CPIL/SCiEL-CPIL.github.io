"""Optimized graph construction for large book minimization."""
import time
import random
import numpy as np

def construct():
    n = 40
    m_target = 401
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Helper to compute book sizes
    def compute_books(adj, n):
        # adj is a set of tuples (u,v) with u<v
        # For each edge, count common neighbors
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if (u,v) in adj:
                    # Common neighbors
                    common = 0
                    for w in range(n):
                        if w != u and w != v:
                            if (min(u,w), max(u,w)) in adj and (min(v,w), max(v,w)) in adj:
                                common += 1
                    books[(u,v)] = common
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    def edge_score(adj, n):
        books = compute_books(adj, n)
        return max_book(books)
    
    # Initial construction: Turan-like T(2,40) + 1 edge
    # T(2,40): bipartite with parts of size 20, 20
    # 20*20 = 400 edges, plus 1 edge within a part = 401
    def make_turan_plus1():
        part1 = list(range(20))
        part2 = list(range(20, 40))
        edges = set()
        for a in part1:
            for b in part2:
                edges.add((a, b))
        # Add one edge within part1
        edges.add((0, 1))
        return edges
    
    # Better initial: use a more balanced approach
    # Start with K20,20 + 1 edge
    best_adj = make_turan_plus1()
    best_score = edge_score(best_adj, n)
    
    # Local search: try to swap edges to reduce max book
    def local_search(adj, n, deadline, max_iter=50000):
        adj = set(adj)
        score = edge_score(adj, n)
        no_improve = 0
        iter_count = 0
        
        while time.time() < deadline - 5.0 and no_improve < 2000 and iter_count < max_iter:
            iter_count += 1
            # Try random edge swap
            # Remove a random edge, add a random non-edge
            edges_list = list(adj)
            u1, v1 = random.choice(edges_list)
            
            # Find a non-edge to add
            found = False
            attempts = 0
            while not found and attempts < 50:
                attempts += 1
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                a, b = min(a,b), max(a,b)
                if (a,b) not in adj:
                    new_adj = adj - {(u1,v1)} | {(a,b)}
                    new_score = edge_score(new_adj, n)
                    if new_score <= score:
                        adj = new_adj
                        if new_score < score:
                            score = new_score
                            no_improve = 0
                        else:
                            no_improve += 1
                        found = True
                    else:
                        no_improve += 1
            if not found:
                no_improve += 100
        
        return adj, score
    
    # Run local search
    best_adj, best_score = local_search(best_adj, n, deadline, max_iter=50000)
    
    # Try multiple random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10.0 and restarts < 3:
        restarts += 1
        # Create a perturbed version
        adj = set(best_adj)
        # Remove 5 random edges and add 5 random non-edges
        for _ in range(5):
            if not adj:
                break
            u, v = random.choice(list(adj))
            adj.remove((u,v))
            # Add a random non-edge
            attempts = 0
            while attempts < 20:
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                a, b = min(a,b), max(a,b)
                if (a,b) not in adj:
                    adj.add((a,b))
                    break
                attempts += 1
        
        adj2, score2 = local_search(adj, n, deadline, max_iter=20000)
        if score2 < best_score:
            best_adj = adj2
            best_score = score2
    
    # Convert to list of [u, v]
    result = []
    for u, v in best_adj:
        result.append([u, v])
    
    # Verify we have exactly 401 edges
    # If not, fix it
    while len(result) < m_target:
        # Add a random edge
        a = random.randint(0, n-1)
        b = random.randint(0, n-1)
        if a == b:
            continue
        a, b = min(a,b), max(a,b)
        if [a,b] not in result:
            result.append([a,b])
    
    if len(result) > m_target:
        result = result[:m_target]
    
    # Verify distinctness
    seen = set()
    final = []
    for e in result:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            final.append([key[0], key[1]])
    
    return final[:m_target]
