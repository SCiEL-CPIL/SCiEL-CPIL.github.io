import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    D = 11
    indices = list(range(D))
    
    def get_best_for_norm_sq(target_norm_sq):
        """
        Find max independent set for vectors with exactly target_norm_sq non-zero components?
        No, target_norm_sq is the squared norm.
        For integer vectors with coords in {0, 1, -1}, norm_sq = number of non-zeros.
        """
        # Generate vectors
        vectors = []
        k = target_norm_sq # number of non-zeros
        if k > D:
            return np.array([])
            
        for comb in combinations(indices, k):
            for signs in product([1, -1], repeat=k):
                v = np.zeros(D, dtype=np.int8)
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vectors.append(v)
                
        if not vectors:
            return np.array([])
            
        vectors = np.array(vectors)
        n = len(vectors)
        
        if n == 0:
            return np.array([])
            
        # Compute dot products
        dots = vectors @ vectors.T
        
        # Conflict condition:
        # R^2 = target_norm_sq
        # Need ||u - v||^2 >= R^2
        # ||u||^2 + ||v||^2 - 2 u.v >= R^2
        # 2 R^2 - 2 u.v >= R^2
        # R^2 >= 2 u.v
        # u.v <= R^2 / 2
        
        # If R^2 is odd, e.g. 3, u.v <= 1.5 => u.v <= 1.
        # If R^2 is even, e.g. 4, u.v <= 2.
        
        max_dot_allowed = target_norm_sq // 2
        
        best_set = []
        best_size = 0
        
        # Greedy with random restarts
        # Number of restarts depends on N
        # N=1320, restarts=100. N=5280, restarts=20.
        max_restarts = max(10, 10000 // n)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            perm = np.random.permutation(n)
            selected = []
            
            for idx in perm:
                if time.time() > deadline - 5:
                    break
                
                # Check conflict
                conflict = False
                for sel_idx in selected:
                    if dots[idx, sel_idx] > max_dot_allowed:
                        conflict = True
                        break
                
                if not conflict:
                    selected.append(idx)
                    
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
                
        if best_set:
            return vectors[best_set].astype(np.int64)
        return np.array([])

    # Try different norms
    # Norm 1: k=1
    # Norm 2: k=2
    # Norm 3: k=3
    # Norm 4: k=4
    
    results = []
    
    # Norm 1
    if time.time() < deadline - 50:
        res1 = get_best_for_norm_sq(1)
        if len(res1) > 0:
            results.append(res1)
            
    # Norm 2
    if time.time() < deadline - 40:
        res2 = get_best_for_norm_sq(2)
        if len(res2) > 0:
            results.append(res2)
            
    # Norm 3
    if time.time() < deadline - 20:
        res3 = get_best_for_norm_sq(3)
        if len(res3) > 0:
            results.append(res3)
            
    # Norm 4
    if time.time() < deadline - 10:
        res4 = get_best_for_norm_sq(4)
        if len(res4) > 0:
            results.append(res4)
            
    if not results:
        # Fallback
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
    # Pick the one with max length
    best = max(results, key=len)
    return best

from itertools import product
