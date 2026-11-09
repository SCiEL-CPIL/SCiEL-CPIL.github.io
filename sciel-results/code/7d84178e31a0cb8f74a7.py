import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    D = 11
    indices = list(range(D))
    
    def solve_mis_fast(vectors, max_dot_allowed):
        """
        Fast greedy MIS using degree ordering.
        vectors: (N, D) array
        max_dot_allowed: int
        """
        n = len(vectors)
        if n == 0:
            return []
        
        # Compute dot products
        dots = vectors @ vectors.T
        
        # Conflict matrix: True if dot > max_dot_allowed
        # Note: diagonal is norm^2, which is > max_dot_allowed usually.
        # We only care about i != j.
        conflict = dots > max_dot_allowed
        np.fill_diagonal(conflict, False)
        
        # Degree of each node
        degrees = conflict.sum(axis=1)
        
        best_set = []
        best_size = 0
        
        # Run greedy with degree ordering
        # To add randomness, we can perturb degrees slightly
        # Or just run once with sorted degrees.
        # Let's run a few times with random tie-breaking or slight noise.
        
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Add small noise to degrees to break ties randomly
            noisy_degrees = degrees + np.random.rand(n) * 0.1
            order = np.argsort(noisy_degrees)
            
            selected = []
            blocked = np.zeros(n, dtype=bool)
            
            for idx in order:
                if blocked[idx]:
                    continue
                
                selected.append(idx)
                
                # Block neighbors
                neighbors = np.where(conflict[idx])[0]
                blocked[neighbors] = True
                blocked[idx] = True
                
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
                
        if best_set:
            return vectors[best_set].astype(np.int64)
        return np.array([])

    results = []
    
    # R^2 = 2
    # Known to be 220. Can construct directly.
    # Vectors with 2 non-zeros.
    # All 220 are compatible.
    res2 = []
    for comb in combinations(indices, 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[comb] = signs
            res2.append(v)
    if res2:
        res2 = np.array(res2)
        results.append(res2)
        
    # R^2 = 3
    if time.time() < deadline - 30:
        vectors3 = []
        for comb in combinations(indices, 3):
            for signs in product([1, -1], repeat=3):
                v = np.zeros(D, dtype=np.int64)
                v[comb] = signs
                vectors3.append(v)
        if vectors3:
            vectors3 = np.array(vectors3)
            # max_dot = 1
            res3 = solve_mis_fast(vectors3, 1)
            if len(res3) > 0:
                results.append(res3)
                
    # R^2 = 4
    if time.time() < deadline - 10:
        vectors4 = []
        for comb in combinations(indices, 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                v[comb] = signs
                vectors4.append(v)
        if vectors4:
            vectors4 = np.array(vectors4)
            # max_dot = 2
            res4 = solve_mis_fast(vectors4, 2)
            if len(res4) > 0:
                results.append(res4)
                
    if not results:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
    best = max(results, key=len)
    return best

