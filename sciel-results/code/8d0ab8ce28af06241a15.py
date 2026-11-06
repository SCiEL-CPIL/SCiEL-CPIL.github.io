import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    # Use a safety margin for the deadline
    deadline = time.time() + 110.0
    
    # Helper to generate a random graph with exactly m edges
    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges

    def compute_metrics(A):
        A2 = A @ A
        # The diagonal of A2 contains the degrees. We only care about off-diagonal for book sizes.
        # Book size of edge (u,v) is A2[u,v].
        # We need to mask the diagonal to 0 so it doesn't affect max/sum if we were summing all, 
        # but max will be on off-diagonal anyway as degrees are ~20 and books are ~7-10.
        # However, strictly speaking, book size is defined for edges.
        # A2[u,v] for u!=v is the number of common neighbors.
        np.fill_diagonal(A2, 0)
        max_book = np.max(A2)
        sum_sq = np.sum(A2**2)
        return max_book, sum_sq

    best_edges = None
    best_max_b = float('inf')
    best_sum_s = float('inf')
    
    # We will run several restarts to ensure we find a good solution
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        edges = random_graph()
        
        # Build initial A
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
            
        max_b, sum_s = compute_metrics(A)
        
        # Local improvement
        if max_b < best_max_b or (max_b == best_max_b and sum_s < best_sum_s):
            best_max_b = max_b
            best_sum_s = sum_s
            best_edges = set(edges)
            
        T = 2.0
        T_min = 0.01
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5:
            if no_improve_count > max_no_improve:
                break
                
            # Pick random edge to remove
            # Try a few times to find an existing edge
            rem_edge = None
            for _ in range(10):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    e = (min(u, v), max(u, v))
                    if e in edges:
                        rem_edge = e
                        break
            if rem_edge is None:
                continue
                
            rem_u, rem_v = rem_edge
            edges.discard(rem_edge)
            
            # Pick random non-edge to add
            add_edge = None
            for _ in range(10):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    e = (min(u, v), max(u, v))
                    if e not in edges:
                        add_edge = e
                        break
            if add_edge is None:
                edges.add(rem_edge)
                continue
                
            add_u, add_v = add_edge
            edges.add(add_edge)
            
            # Update A
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_u, add_v] = 1
            A[add_v, add_u] = 1
            
            new_max_b, new_sum_s = compute_metrics(A)
            new_cost = (new_max_b, new_sum_s)
            old_cost = (max_b, sum_s)
            
            accept = False
            if new_cost < old_cost:
                accept = True
                max_b = new_max_b
                sum_s = new_sum_s
                no_improve_count = 0
                
                if new_cost < (best_max_b, best_sum_s):
                    best_max_b = new_max_b
                    best_sum_s = new_sum_s
                    best_edges = set(edges)
            else:
                # Calculate delta for SA
                # Penalize increase in max book heavily
                if new_max_b > max_b:
                    delta = 100 + (new_sum_s - sum_s)
                else:
                    delta = new_sum_s - sum_s
                    
                if delta < 0:
                    accept = True
                else:
                    if random.random() < np.exp(-delta / T):
                        accept = True
                        max_b = new_max_b
                        sum_s = new_sum_s
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                        
            if not accept:
                # Revert
                edges.discard(add_edge)
                edges.add(rem_edge)
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
            else:
                no_improve_count += 1 if new_cost >= old_cost else 0
                
            # Cool down
            T = max(T_min, T * 0.9995)

    # Return the best found edges
    return [[u, v] for u, v in best_edges]
