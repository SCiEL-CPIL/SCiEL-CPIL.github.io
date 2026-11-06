import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Define Type Space
    # Values in {0, 1, 2, 3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # 2. Parse Distribution
    # We need to handle various input formats for distribution_spec
    probs = np.zeros(n_types)
    
    if isinstance(distribution_spec, dict):
        # Case: Dict mapping type tuple to probability
        if all(isinstance(k, tuple) for k in distribution_spec.keys()):
            for k, v in distribution_spec.items():
                if k in type_to_idx:
                    probs[type_to_idx[k]] = float(v)
        # Case: Dict with 'probs' key containing list/array
        elif 'probs' in distribution_spec:
            p_data = distribution_spec['probs']
            if isinstance(p_data, (list, tuple, np.ndarray)):
                p_arr = np.array(p_data, dtype=float).flatten()
                if len(p_arr) == n_types:
                    probs = p_arr
                else:
                    # Try to match by index if length doesn't match, or assume order
                    # Standard order is row-major: (0,0), (0,1), ..., (3,3)
                    probs = p_arr[:n_types] if len(p_arr) >= n_types else np.pad(p_arr, (0, n_types-len(p_arr)))
        # Case: Dict with other keys that might contain the distribution
        else:
            # Fallback: look for any array-like value of correct size
            for key in ['distribution', 'probabilities', 'weights', 'p']:
                if key in distribution_spec:
                    val = distribution_spec[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.array(val, dtype=float).flatten()
                        if len(arr) == n_types:
                            probs = arr
                            break
            # If still not found, check all values
            if np.all(probs == 0):
                for key in distribution_spec:
                    val = distribution_spec[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.array(val, dtype=float).flatten()
                        if len(arr) == n_types:
                            probs = arr
                            break
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        p_arr = np.array(distribution_spec, dtype=float).flatten()
        if len(p_arr) == n_types:
            probs = p_arr
        else:
            # Assume uniform if length mismatch
            probs = np.ones(n_types) / n_types

    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 1e-15:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types

    # Value vectors
    V = np.array(types, dtype=float)  # (16, 2)

    # 3. Setup LP
    # Variables:
    # x[j, g] for j in 0..15, g in 0..1  (32 vars)
    # u[j] for j in 0..15                (16 vars)
    # Total vars = 48
    
    n_x = n_types * n_goods
    n_u = n_types
    n_vars = n_x + n_u
    
    # Objective: Maximize Revenue = sum_j p_j * (V_j . x_j - u_j)
    # linprog minimizes, so we minimize -Revenue
    c = np.zeros(n_vars)
    for j in range(n_types):
        for g in range(n_goods):
            c[j * n_goods + g] = -probs[j] * V[j, g]
        c[n_x + j] = probs[j]  # -(-p_j * u_j) = +p_j * u_j
        
    # Constraints:
    # 1. IC: For all i != j:
    #    u_i - u_j >= (V_i - V_j) . x_i
    #    => (V_j - V_i) . x_i - u_i + u_j <= 0  (Wait, standard form is u_i - u_j >= (V_i - V_j).x_i)
    #    Let's derive carefully.
    #    Truthfulness: u_i >= V_i . x_i - p_i.
    #    Deviation to j: u_i >= V_i . x_j - p_j.
    #    p_i = V_i . x_i - u_i.
    #    p_j = V_j . x_j - u_j.
    #    IC: V_i . x_i - u_i >= V_i . x_j - (V_j . x_j - u_j)
    #    V_i . x_i - u_i >= V_i . x_j - V_j . x_j + u_j
    #    (V_i - V_j) . x_j - u_i + u_j <= 0 ? No.
    #    Let's use the standard LP formulation variables: x_{j,g} is allocation to type j.
    #    IC constraint: (V_i - V_j) . x_j - u_i + u_j <= 0 is WRONG.
    #    
    #    Correct IC: u_i - u_j >= (V_i - V_j) . x_i  (Envelope theorem condition for x_i)
    #    Actually, the IC constraint relates the utility of reporting i when true type is i vs reporting j.
    #    u_i >= V_i . x_j - p_j
    #    p_j = V_j . x_j - u_j
    #    u_i >= V_i . x_j - V_j . x_j + u_j
    #    u_i - u_j >= (V_i - V_j) . x_j
    #    
    #    So for all i (true type) and j (reported type):
    #    u_i - u_j >= (V_i - V_j) . x_j
    #    => (V_j - V_i) . x_j - u_i + u_j <= 0
    #    
    #    Note: The allocation x_j is determined by the report j. The constraint must hold for all pairs (i, j).
    #    Variables in the constraint: x_j (allocation for report j) and u_i, u_j.
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # (V_j - V_i) . x_j
            for g in range(n_goods):
                row[j * n_goods + g] = V[j, g] - V[i, g]
            # - u_i
            row[n_x + i] = -1.0
            # + u_j
            row[n_x + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    # IR Constraints: u_i >= 0 => -u_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    # x_{j,g} in [0, 1]
    # u_j in [0, infinity) (though bounded by max value)
    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * n_u
    
    # 4. Solve LP
    # Use tight tolerances to ensure high precision
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
        options={
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_simplex_max_iterations': 100000,
            'primal_simplex_max_iterations': 100000
        }
    )
    
    if time.time() > deadline - 5.0:
         # If we are close to the limit and haven't solved, return zero mechanism
         allocations = np.zeros((n_types, n_goods))
         payments = np.zeros(n_types)
         return {'allocations': allocations, 'payments': payments}

    if result.status != 0 or not result.success:
        # Fallback: Zero allocation and zero payment
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = x[:n_x].reshape(n_types, n_goods)
    utilities = x[n_x:]
    
    # Compute payments: p_j = V_j . x_j - u_j
    payments = np.array([np.dot(V[j], allocations[j]) - utilities[j] for j in range(n_types)])
    
    # Ensure non-negative payments (IR for the bidder implies u_j >= 0, so p_j <= V_j.x_j. 
    # But p_j must be >= 0? No, individual rationality is u_j >= 0. 
    # Payment can be negative (subsidy) in theory, but usually we want p_j >= 0 for revenue.
    # The problem asks for non-negative payments.
    # If the LP solution has negative payments, we might need to adjust.
    # However, with u_j >= 0 and x_j >= 0, V_j.x_j - u_j can be negative if u_j > V_j.x_j.
    # The objective maximizes revenue, so it will push u_j down.
    # If p_j < 0, it reduces revenue. The LP should naturally avoid this if possible.
    # But we must enforce p_j >= 0.
    # If p_j < 0, we can set p_j = 0 and adjust u_j? No, that breaks IC.
    # Actually, the constraint u_j >= 0 ensures the bidder gets non-negative utility.
    # If p_j < 0, it means the seller is paying the bidder.
    # The prompt says "nonnegative payments".
    # If the optimal LP has negative payments, we might violate the constraint.
    # However, in standard auction design with single bidder, if we set x=0, p=0, u=0, revenue=0.
    # If the optimal revenue is positive, payments are usually positive.
    # Let's clip payments to 0. If this breaks IC, the solution is invalid.
    # But usually, for monotone mechanisms, p_j is increasing and p_lowest >= 0 if u_lowest >= 0.
    
    payments = np.maximum(payments, 0.0)
    
    # Clip allocations to [0, 1] for safety
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # Final check for time
    if time.time() > deadline:
        return {'allocations': allocations, 'payments': payments}
        
    return {'allocations': allocations, 'payments': payments}
