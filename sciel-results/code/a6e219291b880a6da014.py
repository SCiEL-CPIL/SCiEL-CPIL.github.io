import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 40
    target_edges = 401
    
    # Target degrees: 38 vertices of degree 20, 2 vertices of degree 21
    # Sum = 38*20 + 2*21 = 760 + 42 = 802 = 2 * 401
    target_degrees = [20] * 38 + [21, 21]
    
    best_score = float('inf')
    best_edges = None
    
    # We will try a few random restarts of the greedy construction
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle target degrees to break symmetry
        targets = target_degrees[:]
        random.shuffle(targets)
        
        # Current state
        degrees = [0] * n
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # Greedy addition
        for _ in range(target_edges):
            if time.time() > deadline:
                break
                
            best_pair = None
            best_cost = float('inf')
            
            # Iterate over all possible pairs
            # Optimization: Only consider pairs where adding the edge
            # doesn't exceed target degrees
            candidates = []
            for u in range(n):
                if degrees[u] >= targets[u]:
                    continue
                for v in range(u + 1, n):
                    if degrees[v] >= targets[v]:
                        continue
                    if v in neighbors[u]:
                        continue
                    
                    # Calculate cost
                    # 1. Common neighbors (book size contribution)
                    common = len(neighbors[u] & neighbors[v])
                    
                    # 2. Degree deviation penalty
                    # We want to keep degrees close to targets
                    # Penalty if we make one degree much higher than target
                    # relative to the other?
                    # Simple penalty: sum of squared deviations after adding
                    # d_u' = d_u + 1, d_v' = d_v + 1
                    dev_u = abs(degrees[u] + 1 - targets[u])
                    dev_v = abs(degrees[v] + 1 - targets[v])
                    
                    cost = common * 10 + dev_u + dev_v
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (u, v)
            
            if best_pair is None:
                # Stuck, break restart
                break
                
            u, v = best_pair
            neighbors[u].add(v)
            neighbors[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            edges.append([u, v])
            
        if len(edges) != target_edges:
            continue
            
        # Calculate score
        # Score is max book size
        max_book = 0
        for u in range(n):
            for v in neighbors[u]:
                if u < v: # Count each edge once
                    book_size = len(neighbors[u] & neighbors[v])
                    if book_size > max_book:
                        max_book = book_size
                        
        if max_book < best_score:
            best_score = max_book
            best_edges = edges[:]
            
        # Early exit if we hit the theoretical lower bound
        if best_score <= 7:
            break
            
    if best_edges is None:
        # Fallback to seed if greedy failed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
