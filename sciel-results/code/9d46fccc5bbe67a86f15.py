import random
from itertools import combinations

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def can_merge(p1, p2):
        # p1, p2 are lists of vertices
        # Check if they can be concatenated
        # 1. p1 end == p2 start
        if p1[-1] == p2[0]:
            if len(set(p1 + p2[1:])) == len(p1 + p2[1:]):
                return p1 + p2[1:]
        # 2. p1 start == p2 end
        if p1[0] == p2[-1]:
            if len(set(p2[::-1] + p1[1:])) == len(p2[::-1] + p1[1:]):
                return p2[::-1] + p1[1:]
        # 3. p1 end == p2 end (reverse p2)
        if p1[-1] == p2[-1]:
            if len(set(p1 + p2[-2::-1])) == len(p1 + p2[-2::-1]):
                return p1 + p2[-2::-1]
        # 4. p1 start == p2 start (reverse p2)
        if p1[0] == p2[0]:
            if len(set(p2[::-1] + p1[1:])) == len(p2[::-1] + p1[1:]):
                return p2[::-1] + p1[1:]
        return None

    def minimize_paths(edges, rng):
        paths = [[u, v] for u, v in edges]
        rng.shuffle(paths)
        
        # Use a set of indices to track active paths? 
        # Simpler: list manipulation.
        
        while True:
            merged = False
            # Randomize the order of pairs to check
            indices = list(range(len(paths)))
            rng.shuffle(indices)
            
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                for j_idx in range(i_idx + 1, len(indices)):
                    j = indices[j_idx]
                    # Ensure i and j are still valid and not merged in this pass?
                    # Since we break on first merge, we just need to check current state.
                    # But indices were generated at start of while loop.
                    # If we merge, we break. So i, j are valid.
                    
                    new_path = can_merge(paths[i], paths[j])
                    if new_path:
                        # Remove j, replace i
                        paths.pop(j)
                        paths[i] = new_path
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        best_paths = None
        best_count = float('inf')
        
        # Multiple restarts
        for r in range(50):
            rng = Random(r)
            paths = minimize_paths(edges, rng)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
        results.append(best_paths)
        
    return results

from random import Random
