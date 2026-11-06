import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    m = 401
    
    def random_graph(n, m):
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        return edges

    def calc_max_book(A, edges):
        max_b = 0
        max_edge = None
        for u, v in edges:
            b = np.sum(A[u] & A[v])
            if b > max_b:
                max_b = b
                max_edge = (u, v)
        return max_b, max_edge

    best_edges = None
    best_score = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        # Initialize current state
        # 30% chance to start from best, 70% from random
        if best_edges is not None and random.random() < 0.3:
            edges = set(best_edges)
        else:
            edges = random_graph(n, m)
            
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        current_score, current_edge = calc_max_book(A, edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
            
        # Local search
        steps_since_improvement = 0
        max_steps = 3000 # Limit per restart
        
        for _ in range(max_steps):
            if time.time() > deadline - 5.0:
                break
                
            if current_edge is None:
                break
                
            u, v = current_edge
            
            # Find common neighbors
            common = np.where(A[u] & A[v])[0]
            
            if len(common) == 0:
                if current_score == 0:
                    break
                # If score > 0 but this edge has 0 common neighbors, it's not the max?
                # This can happen if there are multiple max edges and we picked one with 0?
                # No, if it's the max, it must have the max number of common neighbors.
                # If max is 0, we break.
                # If max > 0, common must be non-empty.
                # So this case should only trigger if score is 0.
                pass
            
            # Pick a random common neighbor
            w = common[random.randint(0, len(common)-1)]
            
            # Choose which edge to remove: (u,w) or (v,w)
            if random.random() < 0.5:
                e_remove = (min(u, w), max(u, w))
            else:
                e_remove = (min(v, w), max(v, w))
                
            if e_remove not in edges:
                continue
                
            # Pick a random non-edge to add
            e_add = None
            for _ in range(100): # Try 100 times to find a non-edge
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y:
                    if x > y:
                        x, y = y, x
                    candidate = (x, y)
                    if candidate not in edges:
                        e_add = candidate
                        break
                        
            if e_add is None:
                continue
                
            # Apply
            ru, rv = e_remove
            au, av = e_add
            
            A[ru, rv] = 0
            A[rv, ru] = 0
            A[au, av] = 1
            A[av, au] = 1
            
            edges.discard(e_remove)
            edges.add(e_add)
            
            new_score, new_edge = calc_max_book(A, edges)
            
            if new_score < current_score:
                current_score = new_score
                current_edge = new_edge
                steps_since_improvement = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
            else:
                # Revert
                A[ru, rv] = 1
                A[rv, ru] = 1
                A[au, av] = 0
                A[av, au] = 0
                
                edges.discard(e_add)
                edges.add(e_remove)
                
                steps_since_improvement += 1
                
                if steps_since_improvement > 150:
                    # Restart local search or break to outer loop
                    break
                    
    # Ensure we return a valid graph
    if best_edges is None:
        best_edges = random_graph(n, m)
        
    return [[u, v] for u, v in best_edges]
