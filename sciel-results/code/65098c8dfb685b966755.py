import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    # Points are 0..11
    points = range(12)
    
    # Generate all 6-subsets (blocks)
    all_6subsets = list(combinations(points, 6))
    n_6 = len(all_6subsets) # 924
    
    # Generate all 5-subsets (constraints)
    all_5subsets = list(combinations(points, 5))
    n_5 = len(all_5subsets) # 792
    
    # Map 5-subset to index
    idx_5 = {s: i for i, s in enumerate(all_5subsets)}
    
    # Build the constraint matrix A (n_5 x n_6)
    # A[i, j] = 1 if 5-subset i is contained in 6-subset j
    # We can build this efficiently
    A = np.zeros((n_5, n_6), dtype=np.int8)
    
    for j, block in enumerate(all_6subsets):
        # Each 6-subset contains 6 different 5-subsets
        for k in range(6):
            # Form the 5-subset by removing the k-th element
            # block is a tuple of 6 sorted elements
            # Removing block[k] gives a sorted 5-tuple
            if k == 0:
                sub = block[1:]
            elif k == 5:
                sub = block[:5]
            else:
                sub = block[:k] + block[k+1:]
            idx = idx_5[sub]
            A[idx, j] = 1

    # Try to solve using SciPy MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n_6)
        
        # Constraints: A @ x <= 1
        # And 0 <= x <= 1
        constraint = LinearConstraint(A, ub=1)
        bounds = Bounds(0, 1)
        
        # Solve
        res = milp(c=c, constraints=constraint, bounds=bounds, integrality=np.ones(n_6))
        
        if res.status == 0 and res.x is not None:
            # res.x contains the optimal solution
            # Select blocks where x_j == 1
            solution_blocks = []
            for j in range(n_6):
                if res.x[j] > 0.5:
                    solution_blocks.append(list(all_6subsets[j]))
            
            # Verify and return
            if len(solution_blocks) >= 100: # Sanity check
                return solution_blocks
            else:
                # If MILP failed to find a good solution (unlikely for this instance), fall through
                pass
        
    except Exception:
        pass

    # Fallback: Fast Backtracking / Greedy
    # Since we have the matrix A, we can do a very fast greedy or backtracking
    
    # Greedy with randomization + local search might be faster to implement safely
    # But backtracking with the matrix is also fast.
    
    # Let's try a simple greedy first to get a valid base, then try to extend?
    # Or just use the A matrix for a faster backtracking.
    
    # We need to ensure we return a valid packing.
    # A simple greedy algorithm:
    used_5subsets = set()
    blocks = []
    # Shuffle blocks to break symmetry/randomness if needed, but for deterministic max, 
    # we might want to try all. However, 924 is small.
    
    # Let's try a backtracking using the precomputed A matrix for speed.
    # State: set of used 5-subsets (bitmask or set)
    # We'll use a set of indices for speed in Python.
    
    best_solution = []
    
    def backtrack(used_5, current_blocks):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if len(current_blocks) > len(best_solution):
            best_solution = current_blocks[:]
            
        if len(current_blocks) == 132:
            return
            
        # Heuristic: Pick the 5-subset with the fewest available blocks?
        # Or just iterate through blocks.
        # To make it faster, we can order blocks.
        
        # Simple approach: iterate all blocks, check if they fit.
        # This is O(924 * 6) per step. Depth 132. 132 * 924 * 6 ~ 700k ops. Very fast.
        
        for j in range(n_6):
            # Check if block j fits
            # Block j covers 6 specific 5-subsets.
            # We need to check if any of those 5-subsets are in used_5.
            
            # To avoid recomputing the 5-subset indices for j every time, 
            # we precomputed A. But A is a matrix. 
            # It's faster to store the list of 5-subset indices for each block.
            
            pass # This loop is too slow if done naively in Python recursion without optimization.
            
    # Let's stick to the Greedy + Random Restarts for the fallback, it's simpler.
    # Or, since MILP is the primary, the fallback is rarely needed.
    # But if MILP fails, we need *something*.
    
    # A very fast greedy:
    # 1. Sort blocks by some metric (e.g., original order).
    # 2. Iterate, add if valid.
    
    def greedy():
        used = set()
        res = []
        for j in range(n_6):
            # Get the 6 5-subsets for block j
            # We need to map block j to its 5-subset indices quickly.
            # Let's precompute this list for the fallback.
            pass
            
    # Precompute block_to_5_indices for fallback
    block_to_5_indices = []
    for j in range(n_6):
        block = all_6subsets[j]
        inds = []
        for k in range(6):
            if k == 0:
                sub = block[1:]
            elif k == 5:
                sub = block[:5]
            else:
                sub = block[:k] + block[k+1:]
            inds.append(idx_5[sub])
        block_to_5_indices.append(tuple(inds))

    # Try a few random greedy passes
    import random
    best_greedy = []
    
    for _ in range(50):
        if time.time() > deadline:
            break
        used = set()
        current = []
        # Shuffle blocks
        indices = list(range(n_6))
        random.shuffle(indices)
        
        for j in indices:
            inds = block_to_5_indices[j]
            if not any(i in used for i in inds):
                used.update(inds)
                current.append(list(all_6subsets[j]))
                
        if len(current) > len(best_greedy):
            best_greedy = current
            if len(best_greedy) == 132:
                return best_greedy

    return best_greedy

# To make the code cleaner and ensure the fallback is robust, 
# I will structure the final code to prioritize MILP and use a simple greedy if it fails.
# The greedy above is O(924 * 6) per pass, which is very fast. 50 passes is negligible.
# If 50 random greedy passes don't find 132, we can try a deterministic one.
# But MILP should find 132.

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    points = range(12)
    all_6subsets = list(combinations(points, 6))
    n_6 = len(all_6subsets)
    
    all_5subsets = list(combinations(points, 5))
    n_5 = len(all_5subsets)
    
    idx_5 = {s: i for i, s in enumerate(all_5subsets)}
    
    A = np.zeros((n_5, n_6), dtype=np.int8)
    block_to_5_indices = []
    
    for j, block in enumerate(all_6subsets):
        inds = []
        for k in range(6):
            if k == 0:
                sub = block[1:]
            elif k == 5:
                sub = block[:5]
            else:
                sub = block[:k] + block[k+1:]
            idx = idx_5[sub]
            A[idx, j] = 1
            inds.append(idx)
        block_to_5_indices.append(tuple(inds))

    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = -np.ones(n_6)
        constraint = LinearConstraint(A, ub=1)
        bounds = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraint, bounds=bounds, integrality=np.ones(n_6))
        
        if res.status == 0 and res.x is not None:
            solution_blocks = []
            for j in range(n_6):
                if res.x[j] > 0.5:
                    solution_blocks.append(list(all_6subsets[j]))
            
            if len(solution_blocks) > 0:
                return solution_blocks
    except Exception:
        pass

    # Fallback: Random Greedy
    import random
    best_greedy = []
    
    # Deterministic greedy first
    used = set()
    current = []
    for j in range(n_6):
        inds = block_to_5_indices[j]
        if not any(i in used for i in inds):
            used.update(inds)
            current.append(list(all_6subsets[j]))
    best_greedy = current
    
    # Randomized Greedy
    for _ in range(100):
        if time.time() > deadline:
            break
        used = set()
        current = []
        indices = list(range(n_6))
        random.shuffle(indices)
        
        for j in indices:
            inds = block_to_5_indices[j]
            if not any(i in used for i in inds):
                used.update(inds)
                current.append(list(all_6subsets[j]))
                
        if len(current) > len(best_greedy):
            best_greedy = current
            if len(best_greedy) == 132:
                return best_greedy

    return best_greedy
