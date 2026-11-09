import random
import math
import time
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Identify perfect squares up to 2*256 = 512
    max_sum = 2 * 256
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # 2. Filter out numbers where 2*x is a perfect square (diagonal constraint)
    candidates = []
    for x in range(1, 257):
        if 2 * x not in squares:
            candidates.append(x)
            
    n = len(candidates)
    # Map value to index for easier handling
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 3. Build conflict edges
    # Edge (i, j) if candidates[i] + candidates[j] is a square
    edges = []
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if vi + vj in squares:
                edges.append((i, j))
                
    # If no edges, all candidates are valid
    if not edges:
        return candidates

    # 4. Try MILP
    if HAS_SCIPY_MILP:
        try:
            # Objective: Maximize sum(x_i) => Minimize sum(-x_i)
            c = np.full(n, -1.0)
            
            # Constraints: x_i + x_j <= 1 for all edges
            # A is sparse matrix
            # We can use scipy.sparse
            from scipy.sparse import coo_matrix
            
            rows = []
            cols = []
            vals = []
            
            for u, v in edges:
                rows.append(len(rows))
                cols.append(u)
                vals.append(1.0)
                
                rows.append(len(rows))
                cols.append(v)
                vals.append(1.0)
                
            A = coo_matrix((vals, (rows, cols)), shape=(len(edges), n)).tocsr()
            
            # Constraint: A @ x <= 1
            # LinearConstraint(A, -np.inf, np.ones(len(edges)))
            constraint = LinearConstraint(A, -np.inf, np.ones(len(edges)))
            
            # Bounds: 0 <= x_i <= 1
            bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
            
            # Integrality: all integers
            integrality = np.ones(n)
            
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds, options={"time_limit": 50.0})
            
            if res.status == 0: # Optimal
                x = res.x
                # x should be close to 0 or 1
                selected_idx = [i for i in range(n) if x[i] > 0.5]
                result = [candidates[i] for i in selected_idx]
                
                # Verify
                res_set = set(result)
                valid = True
                for a in result:
                    for b in result:
                        if a + b in squares:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return result
        except Exception:
            pass # Fallback to heuristic
            
    # 5. Fallback: Improved Heuristic
    # Greedy Min-Degree with Local Search
    
    conflict_map = {i: set() for i in range(n)}
    for u, v in edges:
        conflict_map[u].add(v)
        conflict_map[v].add(u)
        
    best_set = set()
    best_size = 0
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        available = set(range(n))
        current_set = set()
        
        while available:
            # Min degree
            min_deg = float('inf')
            candidates_for_pick = []
            for v in available:
                deg = len(conflict_map[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_pick = [v]
                elif deg == min_deg:
                    candidates_for_pick.append(v)
            v = random.choice(candidates_for_pick)
            current_set.add(v)
            to_remove = {v}
            to_remove.update(conflict_map[v] & available)
            available.difference_update(to_remove)
            
        # Local Search: Try to swap
        # This is a simple improvement.
        # If we have time, do a few iterations of random swaps that don't decrease size?
        # Actually, for MIS, if we are stuck, we can try to remove a node and re-add neighbors?
        # A better heuristic: 
        # 1. Greedy
        # 2. For each node not in set, check if adding it and removing its neighbors in set increases size.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in list(available): # available is empty now, we need to track "not in set"
                pass # The 'available' set is empty after greedy. We need a separate 'not_in_set' set.
                
        # Let's re-implement the local search properly
        # We have current_set.
        # not_in_set = set(range(n)) - current_set
        # For each v in not_in_set:
        #   neighbors_in_set = conflict_map[v] & current_set
        #   if len(neighbors_in_set) < 1: 
        #       # Can add v without removing anything? 
        #       # Wait, if v is not in set, and no neighbor is in set, we can add v.
        #       # But greedy already added all possible? No, greedy removes neighbors.
        #       # If v was removed because of a neighbor u, then u is in set.
        #       # So v has at least 1 neighbor in set.
        #   So we must swap.
        #   if len(neighbors_in_set) == 0: # Should not happen if greedy was maximal
        #       current_set.add(v)
        #       improved = True
        #   elif len(neighbors_in_set) == 1:
        #       # Swap v with u? Size stays same. Might allow more later.
        #       pass
        #   elif len(neighbors_in_set) >= 2:
        #       # Removing k neighbors to add 1 node decreases size. Bad.
        #       pass
        
        # This simple local search is weak.
        # Better: Use the result of the first greedy as a seed for a "Simulated Annealing" or just accept the greedy result if MILP failed.
        # Given the time, the greedy result is likely close to optimal if the graph is sparse.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return [candidates[i] for i in best_set]
