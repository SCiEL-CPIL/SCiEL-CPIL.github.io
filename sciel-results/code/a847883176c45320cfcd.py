import time
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    
    # Build conflict graph
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        a = nodes[i]
        for j in range(i+1, n):
            b = nodes[j]
            g = gcd(a, b)
            lcm_ab = a * b // g
            if lcm_ab <= N:
                conflict[i][j] = True
                conflict[j][i] = True
    
    weights = np.array([1.0 / x for x in nodes])
    
    def solution_value(indices):
        return sum(weights[i] for i in indices)
    
    def is_valid(indices):
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if conflict[indices[i]][indices[j]]:
                    return False
        return True
    
    def local_search(indices, max_iters=500):
        """Improve solution by add, remove, and swap operations."""
        idx_set = set(indices)
        current_val = solution_value(indices)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            
            # Try adding each excluded node
            for k in range(n):
                if k not in idx_set:
                    # Check if k conflicts with any in idx_set
                    ok = True
                    for i in idx_set:
                        if conflict[k][i]:
                            ok = False
                            break
                    if ok:
                        new_val = current_val + weights[k]
                        if new_val > current_val + 1e-12:
                            idx_set.add(k)
                            current_val = new_val
                            improved = True
                            break  # Restart after improvement
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            for i in list(idx_set):
                # Find best replacement
                best_gain = 0
                best_swap = None
                for k in range(n):
                    if k in idx_set:
                        continue
                    # Check if k conflicts with any in idx_set except i
                    ok = True
                    for j in idx_set:
                        if j != i and conflict[k][j]:
                            ok = False
                            break
                    if ok:
                        gain = weights[k] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = k
                if best_swap is not None and best_gain > 1e-12:
                    idx_set.remove(i)
                    idx_set.add(best_swap)
                    current_val += best_gain
                    improved = True
                    break  # Restart after improvement
        
        return list(idx_set)
    
    # Greedy with multiple strategies
    best_solution = None
    best_val = 0.0
    
    # Strategy 1: Highest weight first
    def greedy(order):
        remaining = set(range(n))
        solution = []
        val = 0.0
        for i in order:
            if i not in remaining:
                continue
            # Check if i conflicts with any already selected
            ok = True
            for j in solution:
                if conflict[i][j]:
                    ok = False
                    break
            if ok:
                solution.append(i)
                val += weights[i]
                # Remove i and its conflicts
                to_remove = {i}
                for j in remaining:
                    if conflict[i][j]:
                        to_remove.add(j)
                remaining -= to_remove
        return solution, val
    
    # Standard greedy: highest weight first
    order1 = sorted(range(n), key=lambda i: -weights[i])
    sol1, val1 = greedy(order1)
    sol1 = local_search(sol1)
    val1 = solution_value(sol1)
    if val1 > best_val:
        best_val = val1
        best_solution = sol1
    
    # Strategy 2: Randomized greedy (multiple passes)
    for trial in range(20):
        if time.time() > deadline - 15.0:
            break
        random.seed(trial)
        # Perturb weights slightly
        perturbed = weights + np.random.uniform(0, 0.01, n)
        order2 = sorted(range(n), key=lambda i: -perturbed[i])
        sol2, val2 = greedy(order2)
        sol2 = local_search(sol2)
        val2 = solution_value(sol2)
        if val2 > best_val:
            best_val = val2
            best_solution = sol2
    
    # Try MILP
    if time.time() < deadline - 20.0:
        # Build constraint matrix
        A_rows = []
        b_vals = []
        
        for i in range(n):
            for j in range(i+1, n):
                if conflict[i][j]:
                    row = np.zeros(n)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_rows.append(row)
                    b_vals.append(1.0)
        
        if A_rows:
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            lin_cons = LinearConstraint(A, ub=b)
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n)
            c = -weights
            
            try:
                res = milp(
                    c=c,
                    constraints=lin_cons,
                    integrality=integrality,
                    bounds=bounds,
                    options={"time_limit": min(20.0, deadline - time.time() - 10.0), "mip_rel_gap": 1e-10}
                )
                
                if res.status == 0 and res.x is not None:
                    x = res.x
                    selected_indices = list(np.where(x > 0.5)[0])
                    if len(selected_indices) > 0:
                        selected_indices = local_search(selected_indices)
                        milp_val = solution_value(selected_indices)
                        if milp_val > best_val:
                            best_val = milp_val
                            best_solution = selected_indices
            except Exception:
                pass
    
    # Final local search on best solution
    if best_solution is not None and time.time() < deadline - 2.0:
        final_sol = local_search(best_solution, max_iters=2000)
        final_val = solution_value(final_sol)
        if final_val > best_val:
            best_val = final_val
            best_solution = final_sol
    
    if best_solution is None or len(best_solution) == 0:
        # Fallback: just pick 2
        return [2]
    
    return [nodes[i] for i in best_solution]

if __name__ == "__main__":
    construct()
