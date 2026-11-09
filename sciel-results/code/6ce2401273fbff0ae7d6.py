import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i*i <= 2*N:
        squares.add(i*i)
        i += 1
    
    # Identify valid nodes (no self-conflict)
    # If 2*v is a square, v cannot be in the set (diagonal constraint a=b)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2*v not in squares:
            valid_nodes.append(v)
            
    n_vars = len(valid_nodes)
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict matrix
    # conflict_matrix[i][j] = 1 if valid_nodes[i] + valid_nodes[j] is a square
    # We only need upper triangle for constraints x_i + x_j <= 1
    conflict_pairs = []
    for i in range(n_vars):
        v = valid_nodes[i]
        for j in range(i + 1, n_vars):
            u = valid_nodes[j]
            if v + u in squares:
                conflict_pairs.append((i, j))
                
    # If no conflicts, we can take all valid nodes
    if not conflict_pairs:
        return valid_nodes

    # Prepare ILP
    # Objective: Maximize sum(x_i) -> Minimize -sum(x_i)
    c = -np.ones(n_vars)
    
    # Constraints: x_i + x_j <= 1 for all conflicts
    # We need to build the A matrix for LinearConstraint(A @ x <= b)
    # A is dense or sparse? n_vars is ~250. conflict_pairs could be many.
    # Let's count roughly. Average degree in square-sum graph is small.
    # 250^2 / 2 = 31250 pairs. Most are not conflicts.
    # Number of squares up to 512 is 22 (4..512).
    # For each square S, pairs (u, S-u). 
    # Total edges is sum over squares of number of pairs.
    # This is likely < 1000 edges.
    
    if len(conflict_pairs) == 0:
         return valid_nodes

    # Build sparse A matrix for constraints
    # scipy.optimize.milp accepts LinearConstraint with A, lb, ub
    # A should be 2D array.
    # To be efficient, we can use a sparse matrix if available, but milp usually converts.
    # Let's just build a dense matrix if n_vars is small enough, or use COO format if supported.
    # scipy.optimize.milp docs say A can be array-like.
    
    # Actually, building a dense 1000x250 matrix is very fast and small.
    num_constraints = len(conflict_pairs)
    A = np.zeros((num_constraints, n_vars))
    b = np.ones(num_constraints)
    
    for k, (i, j) in enumerate(conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n_vars)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=b),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.success:
            # Extract solution
            solution = res.x
            # Since it's binary, values are close to 0 or 1
            selected_indices = [i for i in range(n_vars) if solution[i] > 0.5]
            selected_nodes = [valid_nodes[i] for i in selected_indices]
            
            # Sanity check: Verify validity
            # Check self-conflicts (already filtered)
            # Check pair conflicts
            is_valid = True
            selected_set = set(selected_nodes)
            for v in selected_nodes:
                if 2*v in squares:
                    is_valid = False
                    break
            if is_valid:
                for i, v in enumerate(selected_nodes):
                    for u in selected_nodes[i+1:]:
                        if v + u in squares:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid and len(selected_nodes) > 0:
                return selected_nodes
            elif is_valid:
                # Fallback if ILP returns empty (shouldn't happen for non-empty graph with no total domination)
                pass
            
        # Fallback if ILP fails or returns invalid
    except Exception as e:
        pass

    # Fallback: Simple Greedy MIS if ILP fails or times out (though milp has no explicit timeout arg in older scipy, 
    # but we have wall clock. If it takes too long, we might not return. 
    # However, for N=256, milp is extremely fast.
    # If we are here, something went wrong. Return a valid seed.
    # Seed: Residue 1 mod 3 is often good.
    seed_set = [v for v in valid_nodes if v % 3 == 1]
    
    # Ensure seed is valid
    valid_seed = []
    s_set = set()
    for v in seed_set:
        ok = True
        if 2*v in squares:
            ok = False
        if ok:
            for u in s_set:
                if v + u in squares:
                    ok = False
                    break
        if ok:
            valid_seed.append(v)
            s_set.add(v)
            
    if valid_seed:
        return valid_seed
    else:
        # Absolute fallback: pick first valid node
        return [valid_nodes[0]] if valid_nodes else [1]
