import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11
    
    # Generate weight-4 vectors
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
            
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # Precompute dot products
    dot = candidates.astype(np.int16) @ candidates.T.astype(np.int16)
    
    # Precompute conflict lists (edges in conflict graph)
    # Conflict if dot > 2
    conflict_list = []
    for i in range(n):
        conflicts = np.where(dot[i] > 2)[0]
        # Exclude self
        conflicts = [j for j in conflicts if j != i]
        conflict_list.append(conflicts)
        
    def greedy_independent_set(order):
        """
        Greedy Independent Set using conflict lists.
        order: list of vertices to consider in this order.
        """
        selected = []
        available = set(order)
        
        for v in order:
            if v not in available:
                continue
            
            # Select v
            selected.append(v)
            available.remove(v)
            
            # Remove all conflicts of v from available
            for c in conflict_list[v]:
                if c in available:
                    available.remove(c)
                    
        return selected

    best_set = []
    
    # 1. Random Restarts
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        sol = greedy_independent_set(order)
        if len(sol) > len(best_set):
            best_set = sol
            
    # 2. Min-Degree Heuristic
    # Degree in conflict graph
    degrees = np.array([len(cl) for cl in conflict_list])
    order = np.argsort(degrees)
    sol = greedy_independent_set(order)
    if len(sol) > len(best_set):
        best_set = sol
        
    # 3. Additional Random Restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            order = list(range(n))
            random.shuffle(order)
            sol = greedy_independent_set(order)
            if len(sol) > len(best_set):
                best_set = sol
                
    if not best_set:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[np.array(best_set, dtype=np.int64)]
    return result.astype(np.int64)
