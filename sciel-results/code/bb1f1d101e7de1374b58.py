import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    n = 128
    # Variables:
    # x[0..n] : binary, membership in A
    # z[i, j] : binary (or continuous), represents x[i]*x[j] for i < j
    # M : continuous, max multiplicity
    
    # We will iterate through sums k in [128, 256]
    # For each k, we need to track the pairs.
    
    # Pre-calculate pairs for each k
    pairs_by_k = {}
    for k in range(128, 257):
        pairs = []
        for a in range(0, n + 1):
            b = k - a
            if b < a:
                continue
            if b > n:
                continue
            # a <= b
            if a <= b:
                pairs.append((a, b))
        pairs_by_k[k] = pairs

    # Total number of x variables: n + 1
    # Total number of z variables: count of all pairs (a,b) with a < b that sum to some k in [128, 256]
    # Note: A pair (a,b) might be relevant for only one k (a+b).
    # So we can map each pair (a,b) with a < b to a unique index.
    
    z_indices = {}
    z_count = 0
    for k in range(128, 257):
        for a, b in pairs_by_k[k]:
            if a < b:
                if (a, b) not in z_indices:
                    z_indices[(a, b)] = z_count
                    z_count += 1
    
    # M is the last variable
    M_idx = (n + 1) + z_count
    total_vars = M_idx + 1
    
    # Objective: Minimize M
    c = np.zeros(total_vars)
    c[M_idx] = 1.0
    
    # Bounds:
    # x_i in [0, 1]
    # z_{a,b} in [0, 1]
    # M in [0, inf] (or a safe upper bound like 129)
    lb = np.zeros(total_vars)
    ub = np.full(total_vars, 1.0)
    ub[M_idx] = 129.0 # Max possible multiplicity is roughly (n+1)/2
    
    # Integrality:
    # x_i are integer
    # z_{a,b} are integer (or we can relax them, but linear constraints usually force them to 0/1 if x are 0/1)
    # Let's mark x and z as integer to be safe, though z might be continuous in some solvers. 
    # Actually, for z = x_a x_b, if x are binary, z is determined. 
    # Standard linearization:
    # z <= x_a
    # z <= x_b
    # z >= x_a + x_b - 1
    # z >= 0
    # If x_a, x_b are binary, z will be binary.
    
    integrality = np.zeros(total_vars)
    integrality[:n+1] = 1.0 # x vars
    integrality[n+1:n+1+z_count] = 1.0 # z vars
    
    # Constraints
    # 1. Linearization constraints for each z_{a,b}
    # z_{a,b} - x_a <= 0
    # z_{a,b} - x_b <= 0
    # -z_{a,b} + x_a + x_b >= 1  =>  x_a + x_b - z_{a,b} >= 1
    
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    for (a, b), z_idx in z_indices.items():
        # z - x_a <= 0
        row = np.zeros(total_vars)
        row[z_idx + n + 1] = 1.0
        row[a] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # z - x_b <= 0
        row = np.zeros(total_vars)
        row[z_idx + n + 1] = 1.0
        row[b] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # x_a + x_b - z >= 1
        row = np.zeros(total_vars)
        row[z_idx + n + 1] = -1.0
        row[a] = 1.0
        row[b] = 1.0
        A_rows.append(row)
        lb_rows.append(1.0)
        ub_rows.append(np.inf)

    # 2. Coverage and Peak constraints for each k in [128, 256]
    # Sum of contributions <= M
    # Sum of contributions >= 1
    
    for k in range(128, 257):
        pairs = pairs_by_k[k]
        if not pairs:
            continue # Should not happen for this range given n=128
            
        # Constraint: Sum <= M
        row_le = np.zeros(total_vars)
        row_ge = np.zeros(total_vars)
        
        for a, b in pairs:
            if a < b:
                z_idx = z_indices[(a, b)]
                row_le[z_idx + n + 1] = 1.0
                row_ge[z_idx + n + 1] = 1.0
            else: # a == b
                row_le[a] = 1.0
                row_ge[a] = 1.0
        
        # Sum <= M  =>  Sum - M <= 0
        row_le[M_idx] = -1.0
        A_rows.append(row_le)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # Sum >= 1
        A_rows.append(row_ge)
        lb_rows.append(1.0)
        ub_rows.append(np.inf)

    A_mat = np.array(A_rows)
    lb_vec = np.array(lb_rows)
    ub_vec = np.array(ub_rows)
    
    cons = LinearConstraint(A_mat, lb_vec, ub_vec)
    bounds = Bounds(lb, ub)
    
    # Set a deadline
    start_time = time.time()
    deadline = 50.0 # Leave 10s margin for overhead and fallback
    
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x_vals = res.x[:n+1]
            # Extract A
            A = [i for i in range(n+1) if x_vals[i] > 0.5]
            if A:
                return A
        else:
            # Fallback if solver fails
            return list(range(64, 129))
            
    except Exception:
        # Fallback
        return list(range(64, 129))

    # If we reach here, return seed
    return list(range(64, 129))
