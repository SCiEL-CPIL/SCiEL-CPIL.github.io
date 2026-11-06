import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n values in descending order of preference (higher n is better)
    # We try to find a valid coloring for these n values.
    # If we find one for n=28, we return it.
    # If not, we try n=27, etc.
    target_ns = [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    
    best_result = None
    best_n_found = 0
    
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
            
        # Try to solve for this n with a time budget
        # Allocate time based on difficulty? Just give it a chunk of time.
        # If we have 100s left, and 10 targets, give 10s each?
        # Or give more time to higher n?
        # Let's just loop through and if time runs out, return best found so far.
        
        # Precompute data structures for this n
        triples = list(itertools.combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute neighbors
        # four_sets_per_triple[i]: list of tuples of 3 indices (the other triples in the 4-set)
        # five_sets_per_triple[i]: list of tuples of 9 indices (the other triples in the 5-set)
        
        four_sets_per_triple = [[] for _ in range(M)]
        for quad in itertools.combinations(range(n), 4):
            tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            for i in range(4):
                others = [tris[j] for j in range(4) if j != i]
                four_sets_per_triple[tris[i]].append(tuple(others))
                
        five_sets_per_triple = [[] for _ in range(M)]
        for quint in itertools.combinations(range(n), 5):
            tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
            for i in range(10):
                others = [tris[j] for j in range(10) if j != i]
                five_sets_per_triple[tris[i]].append(tuple(others))
                
        # Simulated Annealing
        found = False
        # Number of restarts
        max_restarts = 50
        steps_per_restart = 50000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize random coloring
            C = np.random.randint(0, 2, M).astype(np.int8)
            
            # Calculate initial conflicts
            red_k4_count = 0
            blue_k5_count = 0
            
            # Efficient initial count
            # Iterate all 4-sets
            for quad in itertools.combinations(range(n), 4):
                tris = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
                if all(C[t] == 0 for t in tris):
                    red_k4_count += 1
                    
            for quint in itertools.combinations(range(n), 5):
                tris = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
                if all(C[t] == 1 for t in tris):
                    blue_k5_count += 1
                    
            total_conflicts = red_k4_count + blue_k5_count
            
            if total_conflicts == 0:
                found = True
                break
                
            # SA parameters
            T = 1.0
            T_min = 1e-4
            cooling_rate = 0.9995
            
            step = 0
            while step < steps_per_restart:
                if time.time() > deadline - 2.0:
                    break
                    
                # Pick random triple
                t_idx = np.random.randint(M)
                old_val = C[t_idx]
                new_val = 1 - old_val
                
                # Calculate delta
                delta_red = 0
                delta_blue = 0
                
                # Check 4-sets
                for others in four_sets_per_triple[t_idx]:
                    # others is a tuple of 3 indices
                    # Check if all are red (0)
                    # C is numpy array, so C[others] is fast?
                    # Actually, indexing numpy array with list/tuple of indices creates a copy.
                    # sum(C[others] == 0) == 3
                    if np.sum(C[list(others)] == 0) == 3:
                        if old_val == 0:
                            delta_red -= 1
                        else:
                            delta_red += 1
                            
                # Check 5-sets
                for others in five_sets_per_triple[t_idx]:
                    # others is a tuple of 9 indices
                    if np.sum(C[list(others)] == 1) == 9:
                        if old_val == 1:
                            delta_blue -= 1
                        else:
                            delta_blue += 1
                            
                new_total = total_conflicts + delta_red + delta_blue
                delta = new_total - total_conflicts
                
                # Acceptance probability
                if delta < 0:
                    accept = True
                else:
                    # Boltzmann factor
                    # If T is very small, exp(-delta/T) is 0
                    if T < 1e-5:
                        accept = False
                    else:
                        accept = np.random.rand() < np.exp(-delta / T)
                        
                if accept:
                    C[t_idx] = new_val
                    red_k4_count += delta_red
                    blue_k5_count += delta_blue
                    total_conflicts = new_total
                    
                    if total_conflicts == 0:
                        found = True
                        break
                        
                T *= cooling_rate
                step += 1
                
            if found:
                break
                
        if found:
            best_n_found = n
            best_result = {'n': n, 'colors': C.tolist()}
            # Since we are trying n in descending order, if we find a solution for n,
            # it's the best possible for this run (unless we find a higher one? No, we started high).
            # Wait, if we start with 28 and fail, then 27 and succeed, we return 27.
            # If we succeed at 28, we return 28.
            # So we can return immediately?
            # Yes, because we iterate from high to low.
            return best_result
            
    # If no solution found for any target n, return a trivial valid solution
    # n=1 is always valid.
    # But we can do better.
    # Let's return the best n we tried that was valid? No, we only return if valid.
    # If we didn't find any valid coloring in the targets, we should return a small valid one.
    # n=9 is easy.
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
