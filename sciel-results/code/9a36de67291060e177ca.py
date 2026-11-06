# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction that maximizes expected revenue.
    Uses a nested optimization approach: outer loop optimizes allocations,
    inner loop solves for optimal payments given allocations.
    """
    # Types: all combinations of (v1, v2) where vi in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Get probabilities
    if isinstance(distribution_spec, dict) and 'probs' in distribution_spec:
        probs = np.array(distribution_spec['probs'], dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probs sum to 1
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types

    # Precompute values matrix
    V_mat = np.array([[v1, v2] for v1, v2 in types], dtype=float)
    
    def compute_revenue(x_alloc):
        """
        Given allocations x_alloc (16, 2), compute max expected revenue.
        """
        # Calculate valuations for each type given allocation
        # V_t = v1[t]*x1[t] + v2[t]*x2[t]
        V_vals = x_alloc @ V_mat.T # (16,)
        
        # Setup LP for payments q
        # Variables: q[0..15]
        n_vars = n_types
        
        # Objective: Maximize sum(probs * q) -> Minimize -sum(probs * q)
        c = -probs
        
        A_ub_rows = []
        b_ub_vals = []
        
        # IR Constraints: q[t] <= V[t]
        # q[t] - V[t] <= 0
        for t in range(n_types):
            row = np.zeros(n_vars)
            row[t] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(V_vals[t])
            
        # IC Constraints: q[t] - q[t'] <= V[t] - V[t']
        # For all t != t'
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                row = np.zeros(n_vars)
                row[t] = 1.0
                row[t_prime] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(V_vals[t] - V_vals[t_prime])
                
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        # Bounds: q >= 0
        bounds = [(0, None)] * n_types
        
        # Solve inner LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            # Max revenue is -res.fun
            return -res.fun
        else:
            # Infeasible, return penalty
            return -1e9

    def objective(x_flat):
        x_alloc = x_flat.reshape(n_types, 2)
        # Clip to ensure bounds during optimization steps if needed, 
        # though L-BFGS-B respects bounds
        x_alloc = np.clip(x_alloc, 0, 1)
        rev = compute_revenue(x_alloc)
        return -rev

    # Initial guess: Uniform allocation
    x0 = np.ones(n_types * 2) * 0.5
    
    # Bounds for allocations
    bounds = [(0, 1)] * (n_types * 2)
    
    # Optimize allocations
    # Using L-BFGS-B with finite differences for gradient
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success:
        x_alloc = result.x.reshape(n_types, 2)
        x_alloc = np.clip(x_alloc, 0, 1)
        
        # Recover optimal payments for the final allocation
        # We need to solve the inner LP one more time to get q
        V_vals = x_alloc @ V_mat.T
        n_vars = n_types
        c = -probs
        A_ub_rows = []
        b_ub_vals = []
        for t in range(n_types):
            row = np.zeros(n_vars)
            row[t] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(V_vals[t])
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime: continue
                row = np.zeros(n_vars)
                row[t] = 1.0
                row[t_prime] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(V_vals[t] - V_vals[t_prime])
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds_q = [(0, None)] * n_types
        res_q = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_q, method='highs')
        
        if res_q.success:
            payments = res_q.x
            # Ensure non-negative
            payments = np.maximum(payments, 0)
            return {'allocations': x_alloc, 'payments': payments}
        else:
            # Fallback
            return {'allocations': x_alloc, 'payments': np.zeros(n_types)}
    else:
        # Fallback to uniform if optimization fails
        x_alloc = np.ones((n_types, 2)) * 0.5
        return {'allocations': x_alloc, 'payments': np.zeros(n_types)}
# EVOLVE-BLOCK-END
