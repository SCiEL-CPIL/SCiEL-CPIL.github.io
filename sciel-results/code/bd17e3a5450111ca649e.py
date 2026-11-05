import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    deadline = start + 115.0
    max_size = 32
    target_diffs = 30
    
    # We search for the minimal D_max by trying increasing bounds.
    # A known Sidon set covering 1..30 likely has max element < 100.
    # We start from a lower bound. 
    # Lower bound on max element: 
    # If S has size n, max element M.
    # Number of differences is n(n-1)/2.
    # We need to cover 30 diffs.
    # Also Sidon means all diffs distinct.
    # So n(n-1)/2 >= 30 => n >= 8.
    # Minimal M for Sidon set of size n? 
    # For n=8, M is small. 
    # Let's just start from 30 and go up.
    
    for D_max in range(30, 150):
        if time.time() > deadline - 5:
            break
            
        # Variables:
        # x_i for i in 0..D_max (index 0..D_max)
        # y_{i,d} for i in 0..D_max-d, d in 1..D_max
        # M (index -1 or last)
        
        # Let's map variables to indices.
        # 0..D_max : x_i
        # D_max+1 .. : y_{i,d}
        # Last : M
        
        n_x = D_max + 1
        n_y = sum(D_max - d + 1 for d in range(1, D_max + 1))
        n_vars = n_x + n_y + 1
        
        # Objective: Minimize M (last variable)
        c = np.zeros(n_vars)
        c[-1] = 1.0
        
        # Bounds
        # x_i in [0,1]
        # y_{i,d} in [0,1]
        # M in [0, D_max]
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        ub[-1] = D_max
        
        # Constraints
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. x_0 = 1
        row = np.zeros(n_vars)
        row[0] = 1.0
        A_rows.append(row)
        lb_rows.append(1.0)
        ub_rows.append(1.0)
        
        # 2. Sum x_i <= 32
        row = np.zeros(n_vars)
        for i in range(n_x):
            row[i] = 1.0
        A_rows.append(row)
        lb_rows.append(0.0)
        ub_rows.append(max_size)
        
        # Helper to get y index
        def get_y_idx(i, d):
            # y_{i,d} corresponds to pair (i, i+d)
            # We iterate d from 1 to D_max
            # For a fixed d, i goes from 0 to D_max-d
            # Total y vars before d: sum_{k=1}^{d-1} (D_max - k + 1)
            # Index within d: i
            offset = 0
            for k in range(1, d):
                offset += D_max - k + 1
            return n_x + offset + i

        # 3. Coverage: For d in 1..30, sum y_{i,d} >= 1
        for d in range(1, target_diffs + 1):
            row = np.zeros(n_vars)
            for i in range(D_max - d + 1):
                idx = get_y_idx(i, d)
                row[idx] = 1.0
            A_rows.append(row)
            lb_rows.append(1.0)
            ub_rows.append(np.inf)
            
        # 4. Sidon: For d in 1..D_max, sum y_{i,d} <= 1
        for d in range(1, D_max + 1):
            row = np.zeros(n_vars)
            for i in range(D_max - d + 1):
                idx = get_y_idx(i, d)
                row[idx] = 1.0
            A_rows.append(row)
            lb_rows.append(0.0)
            ub_rows.append(1.0)
            
        # 5. Linearization y_{i,d} <= x_i, y_{i,d} <= x_{i+d}, y_{i,d} >= x_i + x_{i+d} - 1
        for d in range(1, D_max + 1):
            for i in range(D_max - d + 1):
                idx_y = get_y_idx(i, d)
                idx_x1 = i
                idx_x2 = i + d
                
                # y <= x_i
                row = np.zeros(n_vars)
                row[idx_y] = 1.0
                row[idx_x1] = -1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(0.0)
                
                # y <= x_{i+d}
                row = np.zeros(n_vars)
                row[idx_y] = 1.0
                row[idx_x2] = -1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(0.0)
                
                # y >= x_i + x_{i+d} - 1  =>  -y + x_i + x_{i+d} <= 1
                row = np.zeros(n_vars)
                row[idx_y] = -1.0
                row[idx_x1] = 1.0
                row[idx_x2] = 1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(1.0)
                
        # 6. M >= i * x_i  =>  M - i * x_i >= 0  =>  -M + i * x_i <= 0
        for i in range(1, n_x): # i=0 is 0*M
            row = np.zeros(n_vars)
            row[-1] = -1.0
            row[i] = i
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
        A = np.array(A_rows)
        lb_arr = np.array(lb_rows)
        ub_arr = np.array(ub_rows)
        
        constraints = LinearConstraint(A, lb_arr, ub_arr)
        bounds = Bounds(lb, ub)
        
        # Integrality: x and y are binary (0/1), M is integer
        integrality = np.zeros(n_vars)
        for i in range(n_x):
            integrality[i] = 1
        for i in range(n_x, n_x + n_y):
            integrality[i] = 1
        integrality[-1] = 1
        
        try:
            res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
            if res.status == 0 and res.fun is not None:
                # Extract solution
                S = []
                for i in range(n_x):
                    if res.x[i] > 0.5:
                        S.append(i)
                if 0 in S:
                    # Verify Sidon and Coverage
                    diffs = set()
                    valid = True
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            d = S[j] - S[i]
                            if d in diffs:
                                valid = False
                                break
                            diffs.add(d)
                        if not valid: break
                    if valid:
                        if all(d in diffs for d in range(1, 31)):
                            return S
                # If verification fails, it might be a numerical issue or infeasibility misreported
                # Continue to next D_max or break?
                # If D_max is tight, maybe increase.
                pass 
        except Exception as e:
            pass
            
    # Fallback to greedy if ILP fails or times out
    S = [0]
    used_diffs = set()
    covered = set()
    for x in range(1, 200):
        if time.time() > deadline - 2:
            break
        if len(S) >= max_size:
            break
        valid = True
        new_covered = 0
        for s in S:
            d = x - s
            if d < 0: continue
            if d in used_diffs:
                valid = False
                break
            if 1 <= d <= 30:
                new_covered += 1
        if valid:
            # Heuristic: prefer x that covers more new diffs
            # Simple greedy: just take first valid? Or best?
            # Let's take the one that covers the most missing diffs
            # This is O(N) per candidate.
            pass
            
    # Better greedy fallback
    S = [0]
    used_diffs = set()
    covered_mask = 0
    full_mask = (1 << 30) - 1
    
    while (covered_mask & full_mask) != full_mask and len(S) < max_size:
        if time.time() > deadline - 2:
            break
        best_x = None
        best_gain = -1
        for x in range(S[-1] + 1, 200):
            valid = True
            gain = 0
            for s in S:
                d = x - s
                if d <= 0: continue
                if d in used_diffs:
                    valid = False
                    break
                if 1 <= d <= 30:
                    if not (covered_mask & (1 << (d-1))):
                        gain += 1
            if valid:
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            d = best_x - s
            if d > 0:
                used_diffs.add(d)
        for s in S[:-1]:
            d = best_x - s
            if 1 <= d <= 30:
                covered_mask |= (1 << (d-1))
                
    return S
