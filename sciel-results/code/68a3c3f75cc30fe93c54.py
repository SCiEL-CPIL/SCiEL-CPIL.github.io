import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (diagonal constraint a=a => 2a not square)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1] # Fallback, though 1 is valid (2!=sq)
        
    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build constraint matrix for MILP
    constraint_pairs = []
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                constraint_pairs.append((i, j))
                
    m_constraints = len(constraint_pairs)
    
    if m_constraints == 0:
        return valid_vertices
        
    # Initialize A_ub
    A_ub = np.zeros((m_constraints, n_vars), dtype=np.float64)
    b_ub = np.ones(m_constraints, dtype=np.float64)
    
    for k, (i, j) in enumerate(constraint_pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    # Objective: Maximize sum(x) => Minimize -sum(x)
    c = -np.ones(n_vars, dtype=np.float64)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars, dtype=np.int8)
    
    best_solution = None
    
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, ub=b_ub)],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 35} # Leave 20s for local search
        )
        
        if res.status == 0 or res.status == 1:
            solution = []
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    solution.append(valid_vertices[i])
            if solution:
                best_solution = solution
        else:
            # Fallback to greedy
            S = set()
            for v in valid_vertices:
                can_add = True
                for u in S:
                    if u + v in squares:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            best_solution = list(S)
            
    except Exception:
        S = set()
        for v in valid_vertices:
            can_add = True
            for u in S:
                if u + v in squares:
                    can_add = False
                    break
            if can_add:
                S.add(v)
        best_solution = list(S)
    
    # Local search refinement: try to add vertices that were excluded
    if best_solution is None:
        best_solution = []
    
    # Build adjacency for quick conflict checking
    adj = [[] for _ in range(n_vars)]
    for i, j in constraint_pairs:
        adj[i].append(j)
        adj[j].append(i)
    
    # Convert solution to set of indices for O(1) lookup
    sol_set = set()
    for v in best_solution:
        idx = v_to_idx.get(v)
        if idx is not None:
            sol_set.add(idx)
    
    # Try adding each excluded vertex if it doesn't conflict
    for idx in range(n_vars):
        if time.time() > deadline - 5:
            break
        if idx in sol_set:
            continue
        # Check if adding idx conflicts with any existing vertex
        conflict = False
        for neighbor in adj[idx]:
            if neighbor in sol_set:
                conflict = True
                break
        if not conflict:
            sol_set.add(idx)
    
    # Try removing one vertex and adding multiple others (swap)
    for _ in range(3): # Try a few rounds of swaps
        if time.time() > deadline - 5:
            break
        improved = False
        for idx_to_remove in list(sol_set):
            # Try removing idx_to_remove and adding all compatible excluded vertices
            new_set = sol_set - {idx_to_remove}
            added = 0
            for idx in range(n_vars):
                if idx in new_set or idx == idx_to_remove:
                    continue
                conflict = False
                for neighbor in adj[idx]:
                    if neighbor in new_set:
                        conflict = True
                        break
                if not conflict:
                    new_set.add(idx)
                    added += 1
            if added > 1: # Net gain
                sol_set = new_set
                improved = True
                break
        if not improved:
            break
    
    # Convert back to list of values
    result = [valid_vertices[i] for i in sorted(sol_set)]
    if not result:
        result = [valid_vertices[0]]
    
    return result
