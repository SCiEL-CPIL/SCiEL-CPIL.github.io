import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n = 40
    m = 401

    # We will binary search on the target max book size B.
    # Theoretical lower bound: ceil(40/6) = 7.
    # Upper bound: start with something reasonable, say 20.
    
    def try_construct(B, seed=0):
        """
        Try to build a graph with exactly m edges where every edge has book <= B.
        Strategy: 
        1. Greedy forward pass: iterate over all potential edges in a deterministic order.
           For each edge (u,v), if current common neighbors of u,v < B, add the edge.
           Stop when we have m edges.
        2. If we can't fill m edges with book < B, we need to allow some edges with book up to B.
           Actually, the constraint is book <= B. So we add edge if common_neighbors <= B.
           Wait: if we add edge (u,v), its book is exactly common_neighbors(u,v) at the time of addition.
           But adding (u,v) increases the common-neighbor count for other pairs.
           
        Revised greedy:
        - Process edges in a fixed order.
        - Maintain adjacency matrix.
        - For edge (u,v): compute c = adj[u] @ adj[v] (common neighbors).
        - If c <= B: add the edge.
        - Continue until m edges added or all edges processed.
        - If we get m edges, check max book. If max <= B, success.
        - If we don't get m edges, failure (need higher B or different order).
        
        To improve fill rate, we can try multiple random orderings if deterministic fails.
        """
        rng = random.Random(seed)
        
        # Generate all edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Deterministic order first
        order = list(all_edges)
        
        adj = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        edges = []
        
        for u, v in order:
            if edge_count >= m:
                break
            # Compute common neighbors
            c = int(adj[u] @ adj[v])
            if c <= B:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                edge_count += 1
        
        if edge_count < m:
            # Try a few random orderings to see if we can fill
            for attempt in range(50):
                rng2 = random.Random(seed * 1000 + attempt)
                order2 = list(all_edges)
                rng2.shuffle(order2)
                
                adj2 = np.zeros((n, n), dtype=np.int8)
                edge_count2 = 0
                edges2 = []
                
                for u, v in order2:
                    if edge_count2 >= m:
                        break
                    c = int(adj2[u] @ adj2[v])
                    if c <= B:
                        adj2[u, v] = 1
                        adj2[v, u] = 1
                        edges2.append((u, v))
                        edge_count2 += 1
                
                if edge_count2 == m:
                    # Verify max book
                    C = adj2 @ adj2
                    max_book = 0
                    for u, v in edges2:
                        if C[u, v] > max_book:
                            max_book = C[u, v]
                    if max_book <= B:
                        return edges2, max_book
            return None, None
        
        # Verify max book
        C = adj @ adj
        max_book = 0
        for u, v in edges:
            if C[u, v] > max_book:
                max_book = C[u, v]
        
        if max_book <= B:
            return edges, max_book
        else:
            # Greedy filled m edges but some have book > B
            # This can happen because adding edges later increases books of earlier edges
            # Try random orderings
            for attempt in range(50):
                rng2 = random.Random(seed * 1000 + attempt)
                order2 = list(all_edges)
                rng2.shuffle(order2)
                
                adj2 = np.zeros((n, n), dtype=np.int8)
                edge_count2 = 0
                edges2 = []
                
                for u, v in order2:
                    if edge_count2 >= m:
                        break
                    c = int(adj2[u] @ adj2[v])
                    if c <= B:
                        adj2[u, v] = 1
                        adj2[v, u] = 1
                        edges2.append((u, v))
                        edge_count2 += 1
                
                if edge_count2 == m:
                    C2 = adj2 @ adj2
                    max_book2 = 0
                    for u, v in edges2:
                        if C2[u, v] > max_book2:
                            max_book2 = C2[u, v]
                    if max_book2 <= B:
                        return edges2, max_book2
            return None, None

    # Binary search for minimum B
    lo = 7   # theoretical lower bound ceil(40/6)
    hi = 25  # safe upper bound
    
    best_edges = None
    best_B = hi
    
    # First, try to find the minimum feasible B
    while lo <= hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        edges_found, max_book_found = try_construct(mid, seed=42)
        
        if edges_found is not None:
            best_edges = edges_found
            best_B = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search found a solution, use it
    if best_edges is not None:
        result = [[u, v] for u, v in best_edges]
        if len(result) == m:
            return result
    
    # Fallback: use the current best approach (greedy with B=hi)
    # Or just return a valid graph
    # Let's try a simple random graph as ultimate fallback
    rng = np.random.default_rng(123)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = rng.choice(len(all_edges), size=m, replace=False)
    result = [list(all_edges[i]) for i in idx]
    
    return result
