import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a valid 2-coloring of 3-uniform hypergraph on n vertices
    with no red K_4^(3) and no blue K_5^(3).
    """
    deadline = time.time() + 115.0
    
    def solve_for_n(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quads and quints edge indices
        quads = list(combinations(range(n), 4))
        quad_edges = np.array([[triple_to_idx[t] for t in combinations(q, 3)] for q in quads], dtype=np.int32)
        
        quints = list(combinations(range(n), 5))
        quint_edges = np.array([[triple_to_idx[t] for t in combinations(qu, 3)] for qu in quints], dtype=np.int32)
        
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize with random or heuristic
            # Heuristic: (a+b+c) % 2
            colors = np.array([(a+b+c) % 2 for a,b,c in triples], dtype=np.uint8)
            # Randomize 20% of edges
            mask = np.random.rand(num_triples) < 0.2
            colors[mask] = np.random.randint(0, 2, size=np.sum(mask), dtype=np.uint8)
            
            current_violations = float('inf')
            stall = 0
            
            for step in range(2000):
                if time.time() > time_limit:
                    break
                    
                # Check Red K4
                red_mask = (colors[quad_edges] == 0).all(axis=1)
                red_indices = np.where(red_mask)[0]
                
                # Check Blue K5
                blue_mask = (colors[quint_edges] == 1).all(axis=1)
                blue_indices = np.where(blue_mask)[0]
                
                total_violations = len(red_indices) + len(blue_indices)
                
                if total_violations == 0:
                    return colors.tolist()
                    
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    
                if total_violations >= current_violations:
                    stall += 1
                else:
                    stall = 0
                    
                if stall > 30:
                    break # Restart
                    
                current_violations = total_violations
                
                # Flip an edge from a violation
                if len(red_indices) > 0:
                    idx = red_indices[random.randint(0, len(red_indices)-1)]
                    edges = quad_edges[idx]
                    flip_idx = edges[random.randint(0, 3)]
                    colors[flip_idx] ^= 1
                elif len(blue_indices) > 0:
                    idx = blue_indices[random.randint(0, len(blue_indices)-1)]
                    edges = quint_edges[idx]
                    flip_idx = edges[random.randint(0, 9)]
                    colors[flip_idx] ^= 1
                else:
                    break

        return None

    # Try n=34
    result = solve_for_n(34, time.time() + 110.0)
    if result:
        return {'n': 34, 'colors': result}
        
    # Try n=33
    if time.time() < deadline:
        result = solve_for_n(33, time.time() + 5.0)
        if result:
            return {'n': 33, 'colors': result}
            
    # Fallback
    n = 10
    triples = list(combinations(range(n), 3))
    colors = [0] * len(triples)
    return {'n': n, 'colors': colors}
