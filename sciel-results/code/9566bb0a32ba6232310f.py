"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Generate all candidate vectors with norm 2
    # Type A: 2*e_i and -2*e_i
    # Type B: s1*e_i + s2*e_j for i<j, s1,s2 in {+1,-1}
    
    candidates = []
    # Type A
    for i in range(D):
        v = [0]*D
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)
    
    # Type B
    for i in range(D):
        for j in range(i+1, D):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    n = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Compute dot product matrix
    dots = cand_arr @ cand_arr.T
    
    # Conflict graph: edge if dot > 0
    # We want max independent set (no edges)
    
    def greedy_independent_set(order, dots, n):
        selected = []
        selected_set = set()
        for idx in order:
            if idx in selected_set:
                continue
            # Check if compatible with all selected
            ok = True
            for s in selected:
                if dots[idx, s] > 0:
                    ok = False
                    break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
        return selected
    
    def local_search(selected, dots, n, max_iters=50):
        """Try to add vectors or swap to improve."""
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any unselected vector
            for idx in range(n):
                if idx in selected_set:
                    continue
                ok = all(dots[idx, s] <= 0 for s in selected_set)
                if ok:
                    selected.append(idx)
                    selected_set.add(idx)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one, add two
            for s in list(selected):
                # Try removing s and adding two new ones
                new_selected = [x for x in selected if x != s]
                new_set = set(new_selected)
                added = 0
                for idx in range(n):
                    if idx in new_set or idx == s:
                        continue
                    if added >= 2:
                        break
                    ok = all(dots[idx, x] <= 0 for x in new_set)
                    if ok:
                        new_selected.append(idx)
                        new_set.add(idx)
                        added += 1
                if added >= 2:
                    selected = new_selected
                    selected_set = new_set
                    improved = True
                    break
        return selected
    
    best_selected = []
    best_size = 0
    
    # Multiple greedy restarts with different orderings
    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        # Different ordering strategies
        if restart % 3 == 0:
            order = list(range(n))
            random.shuffle(order)
        elif restart % 3 == 1:
            # Sort by degree (number of conflicts) ascending - prefer low-degree vertices
            degrees = (dots > 0).sum(axis=1)
            order = sorted(range(n), key=lambda x: degrees[x])
            # Add some randomness
            for i in range(n):
                if random.random() < 0.1:
                    j = random.randint(i, n-1)
                    order[i], order[j] = order[j], order[i]
        else:
            # Random
            order = list(range(n))
            random.shuffle(order)
        
        selected = greedy_independent_set(order, dots, n)
        selected = local_search(selected, dots, n)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    # Build result
    result = cand_arr[best_selected]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
