import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with K20,20 + 1 random internal edge
    # Vertices 0..19 are A, 20..39 are B
    edges = set()
    for u in range(20):
        for v in range(20, 40):
            edges.add((u, v))
    
    # Add 1 internal edge
    # Randomly pick from A
    u, v = random.sample(range(20), 2)
    edges.add((u, v))
    
    # Ensure size is 401
    # K20,20 is 400. +1 is 401.
    
    # Build masks
    # mask[i] is a bitmask of neighbors of i
    masks = [0] * n
    for u, v in edges:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
    def get_score(masks, edges):
        max_book = 0
        for u, v in edges:
            # Count common neighbors
            common = masks[u] & masks[v]
            # popcount
            # Python 3.8+ has int.bit_count()
            c = common.bit_count()
            if c > max_book:
                max_book = c
        return max_book

    current_score = get_score(masks, edges)
    best_score = current_score
    best_edges = set(edges)
    best_masks = masks.copy()
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to remove
        u, v = random.choice(list(edges))
        edges.remove((u, v))
        
        # Update masks
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        
        # Pick a random new edge to add
        # Ensure it's not already in edges
        # And it's a valid edge (u != v)
        # We can pick from all possible pairs
        # Total possible pairs: 40*39/2 = 780
        # Current edges: 400
        # Non-edges: 380
        
        new_edge = None
        while new_edge is None:
            a, b = random.sample(range(n), 2)
            if (a, b) not in edges and (b, a) not in edges:
                new_edge = (a, b)
        
        edges.add(new_edge)
        a, b = new_edge
        masks[a] |= (1 << b)
        masks[b] |= (1 << a)
        
        new_score = get_score(masks, edges)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
                best_masks = masks.copy()
        else:
            # Simulated Annealing acceptance
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject move
                    # Revert
                    edges.remove(new_edge)
                    masks[a] &= ~(1 << b)
                    masks[b] &= ~(1 << a)
                    
                    edges.add((u, v))
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
            else:
                # Should not happen if delta > 0 check is correct, 
                # but new_score could be equal
                current_score = new_score
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Return best edges found
    # Format: list of [u, v]
    return [[u, v] for u, v in best_edges]
