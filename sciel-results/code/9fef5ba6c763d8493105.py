import numpy as np
from math import lcm, gcd
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    weights = np.array([1.0 / x for x in candidates])
    
    # Build conflict matrix
    # C[i, j] = 1 if lcm(candidates[i], candidates[j]) <= 256
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(candidates[i], candidates[j]) <= 256:
                C[i, j] = 1
                C[j, i] = 1
                
    # Try ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Minimize -sum(w_i x_i)
        c = -weights
        
        # Constraints: x_i + x_j <= 1 for all edges
        # We need to form A_ub and b_ub
        # Or use LinearConstraint
        
        # Get edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if C[i, j] == 1:
                    edges.append((i, j))
                    
        if len(edges) == 0:
            # No conflicts, take all
            return candidates
            
        # Number of constraints = len(edges)
        A = np.zeros((len(edges), n))
        b = np.ones(len(edges))
        
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1
            A[k, j] = 1
            
        # Constraints: A x <= b
        # milp expects LinearConstraint(A, lb, ub)
        # Here we want A x <= b, so lb = -inf, ub = b
        constraints = [LinearConstraint(A, -np.inf, b)]
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all integer
        integrality = np.ones(n)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 10})
        
        if res.success and res.x is not None:
            # res.x contains the solution
            # Check if it's valid (due to potential numerical issues, though binary should be clean)
            selected_indices = [i for i in range(n) if res.x[i] > 0.5]
            selected = [candidates[i] for i in selected_indices]
            
            # Verify
            if len(selected) > 0:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if lcm(selected[i], selected[j]) <= 256:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return selected
            
    except Exception as e:
        pass # Fall back to heuristic
        
    # Fallback: Local Search
    # Greedy initialization
    selected = []
    # Sort by weight descending (smaller numbers first? No, 1/x is larger for smaller x)
    # So sort candidates by 1/x descending -> candidates ascending
    # But we must check conflicts.
    # Greedy: iterate candidates in order of decreasing weight (ascending value)
    # Actually, just iterate 2..256 and add if valid.
    # This is the seed program.
    
    # Better Greedy: Randomized Greedy or Sorted by some heuristic.
    # Let's use the seed's greedy as base, then hill climb.
    
    current_selected = []
    for x in range(2, 257):
        if all(lcm(x, y) > 256 for y in current_selected):
            current_selected.append(x)
            
    current_score = sum(1.0 / x for x in current_selected)
    
    # Hill Climbing
    # 1-Swap: Remove one, try to add best other.
    # We can do this repeatedly.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_move = None # (remove_idx, add_val)
        
        # Try removing each element in current_selected
        for i in range(len(current_selected)):
            rem_val = current_selected[i]
            rem_score = 1.0 / rem_val
            
            # Try adding each candidate not in current_selected
            # To speed up, we can pre-filter candidates that don't conflict with the rest
            # But N is small, just loop.
            for add_val in range(2, 257):
                if add_val in current_selected:
                    continue
                # Check if add_val conflicts with any element in current_selected except rem_val
                valid = True
                for j in range(len(current_selected)):
                    if j == i:
                        continue
                    if lcm(add_val, current_selected[j]) <= 256:
                        valid = False
                        break
                if not valid:
                    continue
                
                gain = (1.0 / add_val) - rem_score
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, add_val)
                    
        if best_move:
            i, add_val = best_move
            current_selected.pop(i)
            current_selected.append(add_val)
            current_score += best_gain
            improved = True
            
    # 2-Swap is more complex and might not be worth it if 1-Swap converges well.
    # Given the knowledge that 1-add+2-swap worked, let's add a simple 2-swap if time permits.
    # But 1-Swap is often enough for these dense graphs.
    
    # One more pass: Try to add any element that fits (1-Add)
    # The loop above handles swaps. It doesn't handle pure additions if the set was empty or small initially.
    # But since we start with a valid set, 1-Swap effectively covers "add" if we consider removing a "dummy" or just allowing size change?
    # No, 1-Swap keeps size constant.
    # We should also try 1-Add (increase size) and 1-Remove (decrease size) if they improve score.
    # But removing always decreases score (weights are positive). So we never want to remove unless we add something bigger.
    # So 1-Swap is the core.
    # What about adding a new element without removing?
    # If there is an element that fits, adding it increases score.
    # The greedy initialization tries to add all possible.
    # But maybe a different greedy order would allow a larger set?
    # The hill climbing 1-swap allows changing the set composition.
    # Does it allow size change?
    # If I remove x and add y, size is same.
    # If I can add z without removing, I should.
    # Let's add a "1-Add" loop.
    
    while time.time() < deadline:
        added = False
        for add_val in range(2, 257):
            if add_val in current_selected:
                continue
            if all(lcm(add_val, y) > 256 for y in current_selected):
                current_selected.append(add_val)
                current_score += 1.0 / add_val
                added = True
                break # restart loop to check for more additions
        if not added:
            break
            
    return current_selected
