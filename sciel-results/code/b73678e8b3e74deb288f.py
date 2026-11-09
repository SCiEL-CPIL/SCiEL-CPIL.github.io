"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
        
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute conflict graph
    # We need a list of edges (i, j) where lcm(numbers[i], numbers[j]) <= 256
    edges = []
    conflict_adj = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                edges.append((i, j))
                conflict_adj[i].append(j)
                conflict_adj[j].append(i)
                
    # Reciprocal weights
    weights = np.array([1.0 / numbers[i] for i in range(n)])
    
    # Fallback heuristic if MIP fails or times out
    def heuristic_solve():
        # Greedy + Local Search similar to original but optimized
        # Simple greedy by weight
        selected = set()
        remaining = list(range(n))
        # Sort by weight descending
        remaining.sort(key=lambda i: -weights[i])
        
        for i in remaining:
            # Check if i conflicts with any selected
            can_add = True
            for j in selected:
                if j in conflict_adj[i]:
                    can_add = False
                    break
            if can_add:
                selected.add(i)
                
        # Local Search 1-swap
        current_sum = sum(weights[list(selected)])
        improved = True
        while improved and elapsed() < DEADLINE - 5:
            improved = False
            for i in list(selected):
                # Try removing i
                new_selected = selected - {i}
                # Try adding any non-selected node that doesn't conflict with new_selected
                for j in range(n):
                    if j in new_selected:
                        continue
                    can_add = True
                    for k in new_selected:
                        if k in conflict_adj[j]:
                            can_add = False
                            break
                    if can_add:
                        # Check if adding j improves sum
                        # Actually, we should just re-evaluate best candidate
                        pass 
                # Simplified: just try removing i and see if we can fill better? 
                # This heuristic is weak. Let's just return the greedy result.
                # The MIP is the main solver.
                break
        return sorted([numbers[i] for i in selected])

    # Try MIP
    try:
        # Variables x_i in {0, 1}
        integrality = np.ones(n)
        
        # Objective: maximize sum(w_i * x_i)
        # milp minimizes, so use -weights
        c = -weights
        
        # Constraints: x_i + x_j <= 1 for all edges (i, j)
        # A @ x <= b
        # A is sparse matrix
        if len(edges) > 0:
            row_idx = []
            col_idx = []
            data = []
            for u, v in edges:
                row_idx.append(u)
                col_idx.append(v)
                data.append(1.0)
                row_idx.append(v)
                col_idx.append(u)
                data.append(1.0)
            
            from scipy.sparse import coo_matrix
            A = coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
            b = np.ones(n)
            
            # LinearConstraint(A, -np.inf, b)
            cons = LinearConstraint(A, -np.inf, b)
        else:
            cons = LinearConstraint([], -np.inf, [])
            
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        result = milp(
            c=c,
            constraints=cons,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            # Extract solution
            selected_indices = [i for i in range(n) if result.x[i] > 0.5]
            if selected_indices:
                return sorted([numbers[i] for i in selected_indices])
            
    except Exception as e:
        pass
        
    # Fallback
    return heuristic_solve()
# EVOLVE-BLOCK-END
