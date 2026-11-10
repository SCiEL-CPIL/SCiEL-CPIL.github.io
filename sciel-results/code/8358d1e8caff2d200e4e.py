import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    max_sum = 2 * N
    
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
        
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    edges = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                edges.append((i, j))
                
    n_edges = len(edges)
    
    if n_edges == 0:
        return valid_vertices[:]
        
    A = lil_matrix((n_edges, n_vars))
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1
        A[k, j] = 1
    A = A.tocsr()
    
    c = -np.ones(n_vars)
    b_ub = np.ones(n_edges)
    
    # Precompute neighbors for fallback
    neighbors = {v: set() for v in valid_vertices}
    for i, j in edges:
        vi = valid_vertices[i]
        vj = valid_vertices[j]
        neighbors[vi].add(vj)
        neighbors[vj].add(vi)

    best_solution = None
    best_value = 0
    
    # B&B State
    # Stack of (fixed_vars_dict)
    stack = [{}]
    
    while stack:
        if time.time() > deadline:
            break
            
        fixed = stack.pop()
        
        # Construct bounds
        bounds = []
        for i in range(n_vars):
            if i in fixed:
                val = fixed[i]
                bounds.append((val, val))
            else:
                bounds.append((0, 1))
                
        try:
            res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status != 0:
                continue
                
            # res.fun is min c^T x. We want max sum x, so max is -res.fun
            lp_val = -res.fun
            
            # Prune if LP value is not better than best integer solution
            if lp_val <= best_value:
                continue
                
            x = res.x
            
            # Check if integer
            is_integer = True
            frac_var = -1
            min_dist = 1.0 # Track variable closest to 0.5 for branching
            
            for i in range(n_vars):
                if i in fixed:
                    continue
                xi = x[i]
                if abs(xi - round(xi)) > 1e-6:
                    is_integer = False
                    # Heuristic: branch on variable closest to 0.5
                    dist = abs(xi - 0.5)
                    if dist < min_dist:
                        min_dist = dist
                        frac_var = i
                        
            if is_integer:
                # Update best
                int_val = int(round(lp_val))
                if int_val > best_value:
                    best_value = int_val
                    sol = []
                    for i in range(n_vars):
                        if x[i] > 0.5:
                            sol.append(valid_vertices[i])
                    best_solution = sol
            else:
                if frac_var != -1:
                    # Branch
                    # Add 0 and 1 branches
                    # Order: try 1 first? Or 0 first?
                    # Usually 1 is better for max independent set (include node)
                    stack.append({**fixed, frac_var: 1})
                    stack.append({**fixed, frac_var: 0})
                    
        except Exception:
            continue
            
    if best_solution:
        return best_solution
    else:
        # Fallback greedy
        vs = sorted(valid_vertices, key=lambda v: len(neighbors[v]))
        S = set()
        for v in vs:
            if neighbors[v].isdisjoint(S):
                S.add(v)
        return list(S)
