import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 255
    # Prime factors of 255
    primes = [3, 5, 17]
    
    # 1. Identify candidates: x in [1, 255] such that gcd(x, 255) > 1
    # This is equivalent to x being divisible by at least one of 3, 5, 17
    candidates = []
    for x in range(1, n + 1):
        if math.gcd(x, n) > 1:
            candidates.append(x)
            
    m = len(candidates)
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Ensure 255 is in candidates (it should be, as gcd(255, 255) = 255 > 1)
    if 255 not in cand_to_idx:
        # Fallback if something went wrong, though impossible by definition
        return list(range(3, n + 1, 3))
        
    idx_255 = cand_to_idx[255]
    
    # 2. Build the conflict graph (non-edges)
    # We want to select a subset such that for any two selected, they are NOT in conflict.
    # Conflict: gcd(u, v) == 1
    # Constraint: y_u + y_v <= 1 for all conflicts.
    
    # For efficiency, we build the list of constraints directly.
    # c_matrix will hold the coefficients for each constraint.
    # Since scipy.optimize.milp expects a dense or sparse matrix, 
    # and m is small (~127), we can use a list of tuples or build a sparse matrix.
    
    # Let's count conflicts first to decide on sparsity
    # Actually, m=127, max constraints ~8000. A dense matrix of 8000x127 is ~1M floats, which is fine.
    # But sparse is better practice.
    
    from scipy.sparse import lil_matrix
    
    # We will build the constraint matrix A_ub and b_ub
    # Variables: y_0 ... y_{m-1}
    
    # List of conflicts (i, j) where gcd(candidates[i], candidates[j]) == 1
    conflicts = []
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            vj = candidates[j]
            if math.gcd(vi, vj) == 1:
                conflicts.append((i, j))
                
    num_conflicts = len(conflicts)
    
    # If no conflicts, all candidates form a clique? No, that's unlikely.
    # If num_conflicts is 0, we can take all.
    
    # Initialize sparse matrix for constraints
    # Rows: one per conflict. Cols: m variables.
    A_ub = lil_matrix((num_conflicts, m), dtype=np.float64)
    b_ub = np.ones(num_conflicts, dtype=np.float64)
    
    for k, (i, j) in enumerate(conflicts):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    # 3. Formulate ILP
    # Maximize sum(y)
    # scipy.optimize.milp minimizes c @ x. So we minimize -sum(y).
    c = np.full(m, -1.0, dtype=np.float64)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(m, dtype=np.int8)
    
    # Bounds: 0 <= y_i <= 1
    bounds = [(0, 1)] * m
    
    # Additional constraint: y_255 = 1
    # We can handle this by fixing the variable or adding constraints.
    # Fixing is cleaner: set bound to (1, 1) for idx_255.
    bounds[idx_255] = (1, 1)
    
    # 4. Solve
    try:
        from scipy.optimize import milp, LinearConstraint
        
        # Note: milp requires the constraint matrix to be in a specific format.
        # It accepts sparse matrices.
        # We have inequality constraints A_ub @ y <= b_ub.
        # We also have bounds.
        
        # Check time
        if time.time() > deadline - 5:
            # Fallback to heuristic
            return list(range(3, n + 1, 3))
            
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, ub=b_ub)],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            # Extract solution
            y_sol = res.x
            selected_indices = [i for i in range(m) if y_sol[i] > 0.5]
            selected_set = [candidates[i] for i in selected_indices]
            
            # Verify validity
            is_valid = True
            if 255 not in selected_set:
                is_valid = False
            else:
                for i in range(len(selected_set)):
                    for j in range(i + 1, len(selected_set)):
                        if math.gcd(selected_set[i], selected_set[j]) == 1:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid:
                # Check if this is better than the heuristic (85)
                if len(selected_set) > 85:
                    return selected_set
                else:
                    # Return heuristic if ILP didn't beat it, or if they are equal
                    # Heuristic is deterministic and fast
                    return list(range(3, n + 1, 3))
            else:
                # ILP returned invalid solution? Should not happen with correct formulation.
                # Fallback
                return list(range(3, n + 1, 3))
                
        else:
            # Solver failed or timeout
            return list(range(3, n + 1, 3))
            
    except Exception:
        # Fallback to heuristic
        return list(range(3, n + 1, 3))
