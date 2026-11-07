"""Rewritten program using combinatorial search to maximize distinct induced degrees."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _induced_degrees_count(mask, adj, n):
    """Count distinct degrees in the induced subgraph defined by mask."""
    k = bin(mask).count('1')
    if k < 2:
        return 1
    degrees = set()
    for i in range(n):
        if mask & (1 << i):
            deg = bin(adj[i] & mask).count('1')
            degrees.add(deg)
    return len(degrees)

def _score(mask, adj, n):
    return _induced_degrees_count(mask, adj, n)

def _random_subset_of_size(k, n):
    subset = random.sample(range(n), k)
    mask = 0
    for v in subset:
        mask |= (1 << v)
    return mask

def _flip_one_bit(mask, n):
    """Randomly flip one bit (add or remove a vertex)."""
    # Try to add a vertex not in mask
    not_in = [i for i in range(n) if not (mask & (1 << i))]
    in_mask = [i for i in range(n) if (mask & (1 << i))]
    
    if not_in and random.random() < 0.5:
        v = random.choice(not_in)
        return mask | (1 << v)
    elif in_mask:
        v = random.choice(in_mask)
        new_mask = mask & ~(1 << v)
        if bin(new_mask).count('1') >= 8:
            return new_mask
    return mask

def _swap_vertices(mask, n):
    """Swap one vertex in with one vertex out."""
    not_in = [i for i in range(n) if not (mask & (1 << i))]
    in_mask = [i for i in range(n) if (mask & (1 << i))]
    if not not_in or not in_mask:
        return mask
    v_out = random.choice(in_mask)
    v_in = random.choice(not_in)
    new_mask = (mask & ~(1 << v_out)) | (1 << v_in)
    return new_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_adj()
    
    best_score = 0
    best_mask = 0
    best_vertices = list(range(8))
    
    # Initialize with a good starting point
    initial_mask = 0
    for v in range(8):
        initial_mask |= (1 << v)
    best_mask = initial_mask
    best_score = _score(initial_mask, adj, n)
    best_vertices = [v for v in range(n) if initial_mask & (1 << v)]
    
    # Try all sizes from 8 to 17
    for k in range(8, n + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Multiple restarts for each size
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Start with random subset of size k
            mask = _random_subset_of_size(k, n)
            current_score = _score(mask, adj, n)
            
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
                best_vertices = [v for v in range(n) if mask & (1 << v)]
            
            # Simulated annealing
            temp = 2.0
            cooling = 0.995
            for step in range(2000):
                if time.time() > deadline - 5.0:
                    break
                
                # Generate neighbor
                if random.random() < 0.3:
                    new_mask = _swap_vertices(mask, n)
                else:
                    new_mask = _flip_one_bit(mask, n)
                
                if new_mask == mask:
                    continue
                
                new_k = bin(new_mask).count('1')
                if new_k < 8:
                    continue
                
                new_score = _score(new_mask, adj, n)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < min(1.0, delta / max(temp, 0.1)):
                    mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                        best_vertices = [v for v in range(n) if mask & (1 << v)]
                
                temp *= cooling
                
                if temp < 0.01:
                    break
    
    # Final check: ensure we have at least 8 vertices
    if len(best_vertices) < 8:
        best_vertices = list(range(8))
    
    # Return as list of Python ints
    return list(best_vertices)
# EVOLVE-BLOCK-END
