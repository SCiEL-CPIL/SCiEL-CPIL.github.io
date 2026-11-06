"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit (5s buffer)
    
    # 1. Define the set of candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute incompatibility (conflict) pairs
    # conflict[i][j] is True if lcm(candidates[i], candidates[j]) <= 256
    conflicts = []
    # For efficiency, we can precompute lcm or just check on the fly.
    # N=255, pairs ~ 32k. lcm calculation is fast.
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflicts.append((i, j))
    
    # 3. Try to solve exactly using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Weights: 1/x
        weights = np.array([1.0 / x for x in candidates])
        
        # Variables: x_i in {0, 1}
        integrality = np.ones(n)
        
        # Objective: Maximize sum(w_i * x_i)
        # scipy milp minimizes, so we minimize -sum(w_i * x_i)
        c = -weights
        
        # Constraints:
        # 1. x_i + x_j <= 1 for each conflict (i, j)
        if len(conflicts) > 0:
            rows = []
            cols = []
            data = []
            for k, (i, j) in enumerate(conflicts):
                rows.extend([k, k])
                cols.extend([i, j])
                data.extend([1, 1])
            
            A = sp.coo_matrix((data, (rows, cols)), shape=(len(conflicts), n)).tocsr()
            b_ub = np.ones(len(conflicts))
            
            # LinearConstraint for x_i + x_j <= 1
            constraints = [
                LinearConstraint(A, ub=b_ub)
            ]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        
        # Set time limit for the solver
        # milp doesn't take a direct time limit argument in all versions, 
        # but we can wrap it or just trust it's fast for N=255.
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:  # Optimal solution found
            # Extract solution
            x_sol = res.x
            selected_indices = np.where(x_sol > 0.5)[0]
            selected_values = [candidates[i] for i in selected_indices]
            
            # Verify validity just in case
            valid = True
            for i in range(len(selected_values)):
                for j in range(i + 1, len(selected_values)):
                    if lcm(selected_values[i], selected_values[j]) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Try to polish with local search to see if we can improve
                # (MILP should be optimal, but this is a safety net)
                current_set = set(selected_indices)
                current_sum = sum(1.0 / candidates[i] for i in current_set)
                
                # Precompute incompatibility sets for fast lookup
                incompatible_sets = [set() for _ in range(n)]
                for i, j in conflicts:
                    incompatible_sets[i].add(j)
                    incompatible_sets[j].add(i)
                
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    # Try 1-swap: remove one, add one
                    for i in list(current_set):
                        trial_set = current_set - {i}
                        # Find best candidate to add
                        best_gain = 0
                        best_candidate = -1
                        for j in range(n):
                            if j in trial_set:
                                continue
                            if not (incompatible_sets[j] & trial_set):
                                gain = 1.0 / candidates[j] - 1.0 / candidates[i]
                                if gain > best_gain + 1e-12:
                                    best_gain = gain
                                    best_candidate = j
                        if best_candidate != -1:
                            current_set.remove(i)
                            current_set.add(best_candidate)
                            improved = True
                            break
                            
                return [candidates[i] for i in current_set]
            else:
                # Fallback if verification fails (shouldn't happen)
                pass
                
    except Exception as e:
        # Fallback to heuristic if scipy fails or is not available
        pass

    # 4. Fallback: Greedy + Local Search
    # Precompute incompatibility sets for fast lookup
    incompatible_sets = [set() for _ in range(n)]
    for i, j in conflicts:
        incompatible_sets[i].add(j)
        incompatible_sets[j].add(i)
    
    # Greedy: Sort by weight descending (smallest x first)
    selected = []
    selected_set = set()
    for i in range(n):
        # Check if i conflicts with any selected
        if not (incompatible_sets[i] & selected_set):
            selected.append(candidates[i])
            selected_set.add(i)
    
    # Local Search: Try to swap or add/remove
    current_sum = sum(1.0/x for x in selected)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing one and adding one
        for i in list(selected_set):
            trial_set = selected_set - {i}
            # Find best candidate to add
            best_gain = 0
            best_candidate = -1
            for j in range(n):
                if j in trial_set:
                    continue
                if not (incompatible_sets[j] & trial_set):
                    gain = 1.0 / candidates[j] - 1.0 / candidates[i]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_candidate = j
            if best_candidate != -1:
                selected_set.remove(i)
                selected_set.add(best_candidate)
                improved = True
                break
                
    return [candidates[i] for i in selected_set]

# EVOLVE-BLOCK-END
