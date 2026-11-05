import math
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = 255
    
    # Quick fallback: multiples of 3 (85 elements)
    fallback = [i for i in range(3, n + 1) if i % 3 == 0]
    
    # Try ILP approach
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Numbers that can potentially be in the set must share a factor with 255
        # 255 = 3 * 5 * 17
        candidates = [i for i in range(1, n + 1) if math.gcd(i, 255) > 1]
        # Also must include 255
        if 255 not in candidates:
            candidates.append(255)
        
        m = len(candidates)
        cand_index = {v: i for i, v in enumerate(candidates)}
        idx_255 = cand_index[255]
        
        # Build conflict constraints: for each pair (i,j) with gcd=1, x_i + x_j <= 1
        # But only among candidates (all share factor with 255)
        # Two candidates conflict if gcd(candidates[i], candidates[j]) == 1
        
        # Precompute gcd conflicts
        # This could be O(m^2) which is ~ (170)^2 / 2 ~ 14k pairs - manageable
        A_rows = []
        b_ub = []
        
        for i in range(m):
            for j in range(i + 1, m):
                if math.gcd(candidates[i], candidates[j]) == 1:
                    row = np.zeros(m)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_rows.append(row)
                    b_ub.append(1.0)
        
        if not A_rows:
            # No conflicts among candidates - take all
            return candidates
        
        A = np.array(A_rows)
        b_ub_arr = np.array(b_ub)
        
        # Objective: maximize sum of x_i
        c = -np.ones(m)  # minimize -sum
        
        # Bounds: x_i in [0, 1]
        lb = np.zeros(m)
        ub = np.ones(m)
        
        # x_255 must be 1
        lb[idx_255] = 1.0
        ub[idx_255] = 1.0
        
        constraints = [
            LinearConstraint(A, 0, b_ub_arr),  # x_i + x_j <= 1 for conflicts
            LinearConstraint(np.eye(m), lb, ub)  # bounds
        ]
        
        bounds = Bounds(lb, ub)
        
        # integrality: all binary
        integrality = np.ones(m)
        
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.status == 0 and result.x is not None:
            selected = [candidates[i] for i in range(m) if result.x[i] > 0.5]
            if 255 in selected and len(selected) > len(fallback):
                # Verify
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if math.gcd(selected[i], selected[j]) == 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return selected
        
        return fallback
        
    except Exception:
        return fallback
