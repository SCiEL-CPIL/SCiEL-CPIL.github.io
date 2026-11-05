import math
import time
import numpy as np

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    n = 255
    
    # Baseline: multiples of 3 (largest single-prime multiple set)
    baseline = list(range(3, n + 1, 3))
    
    # Try MILP to see if we can do better
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        # All candidates that share a factor with 255
        candidates = [x for x in range(1, n + 1) if math.gcd(x, 255) > 1]
        m = len(candidates)
        
        # Build conflict matrix: conflict[i][j] = True if gcd(candidates[i], candidates[j]) == 1
        # For MILP: x_i + x_j <= 1 for each conflicting pair
        conflicts = []
        for i in range(m):
            for j in range(i + 1, m):
                if math.gcd(candidates[i], candidates[j]) == 1:
                    conflicts.append((i, j))
        
        # Objective: maximize sum(x_i), i.e., minimize -sum(x_i)
        c = np.ones(m)
        
        # Constraints: x_i + x_j <= 1 for each conflict
        # Also: x_{index of 255} = 1 (must include 255)
        idx_255 = candidates.index(255)
        
        # Build A matrix for inequality constraints
        if conflicts:
            rows = []
            cols = []
            vals = []
            for (i, j) in conflicts:
                rows.extend([len(rows) // 2, len(rows) // 2])
                cols.extend([i, j])
                vals.extend([1, 1])
            
            A = csr_matrix((vals, (rows, cols)), shape=(len(conflicts), m))
            lb = np.zeros(len(conflicts))
            ub = np.ones(len(conflicts))
            
            # Equality constraint: x_{idx_255} = 1
            A_eq = np.zeros((1, m))
            A_eq[0, idx_255] = 1
            lb_eq = [1]
            ub_eq = [1]
            
            constraints = [
                LinearConstraint(A, lb, ub),
                LinearConstraint(A_eq, lb_eq, ub_eq)
            ]
        else:
            constraints = [
                LinearConstraint(np.zeros((1, m)), 0, m)  # dummy
            ]
        
        bounds = Bounds(0, 1)
        
        result = milp(
            c=c,
            constraints=constraints,
            integrality=np.ones(m),
            bounds=bounds
        )
        
        if result.success and result.x is not None:
            selected = [candidates[i] for i in range(m) if result.x[i] > 0.5]
            # Verify validity
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if math.gcd(selected[i], selected[j]) == 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(selected) > len(baseline):
                return selected
    
    except Exception:
        pass
    
    return baseline

