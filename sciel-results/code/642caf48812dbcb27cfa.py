"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin within 90s limit
    
    N = 256
    
    # Perfect squares up to 512
    squares = set()
    for k in range(1, 23):
        sq = k * k
        if sq <= 512:
            squares.add(sq)
    
    # Diagonal constraint: 2a must not be a perfect square
    diagonal_bad = set()
    for k in range(1, 16):
        val = 2 * k * k
        if val <= N:
            diagonal_bad.add(val)
    
    # Filter to only valid vertices (not in diagonal_bad)
    valid_vertices = [a for a in range(1, N + 1) if a not in diagonal_bad]
    n_valid = len(valid_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build adjacency list among valid vertices
    adj = [set() for _ in range(n_valid)]
    for i, a in enumerate(valid_vertices):
        for j in range(i + 1, n_valid):
            b = valid_vertices[j]
            if (a + b) in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Conflict graph as adjacency matrix for LP
    # A[i,j] = 1 if i and j conflict
    A_conflict = np.zeros((n_valid, n_valid), dtype=np.float64)
    for i in range(n_valid):
        for j in adj[i]:
            A_conflict[i, j] = 1
    
    # Objective: maximize sum of x_i
    c = np.ones(n_valid, dtype=np.float64)
    
    # Constraints: for each edge (i,j), x_i + x_j <= 1
    # Also: 0 <= x_i <= 1
    edge_constraints = []
    for i in range(n_valid):
        for j in adj[i]:
            if i < j:
                row = np.zeros(n_valid, dtype=np.float64)
                row[i] = 1
                row[j] = 1
                edge_constraints.append(row)
    
    if edge_constraints:
        A_ub = np.array(edge_constraints, dtype=np.float64)
        b_ub = np.ones(len(edge_constraints), dtype=np.float64)
    else:
        A_ub = np.zeros((1, n_valid), dtype=np.float64)
        b_ub = np.array([1.0], dtype=np.float64)
    
    bounds = [(0, 1)] * n_valid
    
    # Solve LP relaxation
    lp_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if lp_result.status == 0:
        lp_solution = lp_result.x
    else:
        # Fallback to greedy
        lp_solution = None
    
    # Branch and bound
    best_solution = None
    best_value = 0
    
    # If LP gives integer solution, we're done
    if lp_solution is not None:
        lp_int = np.round(lp_solution).astype(int)
        if np.all(lp_int == lp_solution) and np.all(lp_int >= 0) and np.all(lp_int <= 1):
            # Verify feasibility
            feasible = True
            for i in range(n_valid):
                for j in adj[i]:
                    if i < j and lp_int[i] + lp_int[j] > 1:
                        feasible = False
                        break
                if not feasible:
                    break
            if feasible:
                best_value = int(np.sum(lp_int))
                best_solution = lp_int
    
    # Branch and bound if needed
    def solve_lp_with_fixed(fixed_vars, fixed_vals):
        """Solve LP with some variables fixed."""
        n = n_valid
        c_new = np.ones(n, dtype=np.float64)
        A_ub_new = []
        b_ub_new = []
        bounds_new = []
        
        for i in range(n):
            if i in fixed_vars:
                bounds_new.append((fixed_vals[i], fixed_vals[i]))
            else:
                bounds_new.append((0, 1))
        
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    row = np.zeros(n, dtype=np.float64)
                    row[i] = 1
                    row[j] = 1
                    A_ub_new.append(row)
                    b_ub_new.append(1.0)
        
        if A_ub_new:
            A_ub_arr = np.array(A_ub_new, dtype=np.float64)
            b_ub_arr = np.array(b_ub_new, dtype=np.float64)
        else:
            A_ub_arr = np.zeros((1, n), dtype=np.float64)
            b_ub_arr = np.array([1.0], dtype=np.float64)
        
        res = linprog(c_new, A_ub=A_ub_arr, b_ub=b_ub_arr, bounds=bounds_new, method='highs')
        if res.status == 0:
            return res.x, res.fun
        return None, 0
    
    def branch_and_bound(fixed_vars, fixed_vals, depth=0):
        nonlocal best_solution, best_value
        
        if time.time() > deadline - 2:
            return
        
        # Solve LP relaxation
        lp_sol, lp_val = solve_lp_with_fixed(fixed_vars, fixed_vals)
        if lp_sol is None:
            return
        
        # Check if all free variables are integer
        free_vars = [i for i in range(n_valid) if i not in fixed_vars]
        all_integer = True
        for i in free_vars:
            if abs(lp_sol[i] - round(lp_sol[i])) > 1e-6:
                all_integer = False
                break
        
        if all_integer:
            # Integer solution found
            val = int(round(lp_val))
            if val > best_value:
                # Verify
                sol = np.zeros(n_valid, dtype=int)
                for i in fixed_vars:
                    sol[i] = fixed_vals[i]
                for i in free_vars:
                    sol[i] = int(round(lp_sol[i]))
                # Check feasibility
                feasible = True
                for i in range(n_valid):
                    for j in adj[i]:
                        if i < j and sol[i] + sol[j] > 1:
                            feasible = False
                            break
                    if not feasible:
                        break
                if feasible:
                    best_solution = sol
                    best_value = val
            return
        
        # Fractional solution: branch on a fractional variable
        # Choose variable with value closest to 0.5
        frac_var = None
        min_dist = 1.0
        for i in free_vars:
            v = lp_sol[i]
            dist = abs(v - round(v))
            if dist < min_dist:
                min_dist = dist
                frac_var = i
        
        if frac_var is None:
            return
        
        # Branch: frac_var = 0
        new_fixed_vars = fixed_vars | {frac_var}
        new_fixed_vals = dict(fixed_vals)
        new_fixed_vals[frac_var] = 0
        branch_and_bound(new_fixed_vars, new_fixed_vals, depth + 1)
        
        if time.time() > deadline - 2:
            return
        
        # Branch: frac_var = 1
        new_fixed_vals[frac_var] = 1
        branch_and_bound(new_fixed_vars, new_fixed_vals, depth + 1)
    
    if best_solution is None:
        branch_and_bound(set(), {})
    
    # If B&B didn't find anything, use greedy as fallback
    if best_solution is None:
        # Greedy: sort by degree (highest first), add if no conflict
        degrees = [len(adj[i]) for i in range(n_valid)]
        order = sorted(range(n_valid), key=lambda i: -degrees[i])
        selected = set()
        for i in order:
            conflict = False
            for j in adj[i]:
                if j in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(i)
        best_solution = np.zeros(n_valid, dtype=int)
        for i in selected:
            best_solution[i] = 1
        best_value = len(selected)
    
    # Convert back to original values
    result = [valid_vertices[i] for i in range(n_valid) if best_solution[i] == 1]
    
    if not result:
        result = [valid_vertices[0]] if valid_vertices else [1]
    
    return result
# EVOLVE-BLOCK-END
