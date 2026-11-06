import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction using the envelope theorem.
    
    Strategy: Reformulate the auction design as a linear program over the utility
    function u(v) for each of the 16 types. The envelope theorem gives us that
    for any two types i and j:
        u(i) >= u(j) + v_i . (x(i) - x(j))
    and by monotonicity, x(i) >= x(j) implies u(i) >= u(j) + (v_i - v_j).x(j)
    
    We use the fact that in optimal auctions, the allocation is determined by
    the utility function via the envelope condition. We directly optimize over
    utilities and derive allocations from them.
    
    Actually, a cleaner approach: We note that the IC constraints can be written
    as:
        u(i) - u(j) >= (v_i - v_j) . x(j) for all i, j
    
    And IR: u(i) >= 0 for all i.
    
    Revenue = sum_i p_i * [v_i . x(i) - u(i)]
    
    But we need to express x(i) in terms of u. This is tricky. Instead, let's
    keep the allocation variables but use a smarter formulation.
    
    Alternative: Use the fact that for single-parameter or multi-parameter with
    independent values, we can use the "ironing" approach. But with 16 types and
    2 goods, let's try a direct LP with the envelope theorem to reduce constraints.
    
    Key insight: The IC constraint u(i) >= u(j) + (v_i - v_j).x(j) is equivalent
    to saying that u is "convex" in a certain sense. We can parameterize by u and
    recover x from the subgradient.
    
    For a finite type space, the optimal mechanism has the property that x(i) is
    the subgradient of u at v_i. So:
        x(i) in subgrad(u, v_i)
    
    This means: u(j) >= u(i) + (v_j - v_i).x(i) for all j
    
    So the constraints become:
        u(j) - u(i) >= (v_j - v_i).x(i) for all i, j
    
    And we want to maximize revenue = sum_i p_i * [v_i.x(i) - u(i)]
    
    Subject to:
        x(i) in [0,1]^2
        u(i) >= 0 (IR)
        u(j) - u(i) >= (v_j - v_i).x(i) for all i, j (IC via envelope)
    
    This is still a linear program! Let's formulate it.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)
        
    if probs.shape != (16,):
        probs = probs.flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))
    
    n_types = 16
    
    # Variables: x1_0..x1_15, x2_0..x2_15, u_0..u_15
    # Total: 16 + 16 + 16 = 48 variables
    # Index: x1_i at i, x2_i at 16+i, u_i at 32+i
    
    n_vars = 48
    
    # Objective: maximize revenue = sum_i probs_i * (v_i.x(i) - u_i)
    # = sum_i probs_i * (v1_i*x1_i + v2_i*x2_i - u_i)
    # Minimize negative: -sum_i probs_i * (v1_i*x1_i + v2_i*x2_i - u_i)
    c = np.zeros(n_vars)
    for i in range(n_types):
        v1, v2 = grid_values[i]
        c[i] = -probs[i] * v1       # coefficient for x1_i
        c[16 + i] = -probs[i] * v2  # coefficient for x2_i
        c[32 + i] = probs[i]        # coefficient for u_i (negative of -u_i)
    
    A_rows = []
    b_rows = []
    
    # IR constraints: u_i >= 0 => -u_i <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = -1
        A_rows.append(row)
        b_rows.append(0)
    
    # IC constraints via envelope theorem:
    # u_j - u_i >= (v_j - v_i).x_i for all i, j
    # => u_j - u_i - (v_j1 - v_i1)*x1_i - (v_j2 - v_i2)*x2_i >= 0
    # => -u_j + u_i + (v_j1 - v_i1)*x1_i + (v_j2 - v_i2)*x2_i <= 0
    
    for i in range(n_types):
        v1_i, v2_i = grid_values[i]
        for j in range(n_types):
            if i == j:
                continue
            v1_j, v2_j = grid_values[j]
            
            row = np.zeros(n_vars)
            # +u_i
            row[32 + i] = 1
            # -u_j
            row[32 + j] = -1
            # +(v_j1 - v_i1)*x1_i
            row[i] = (v1_j - v1_i)
            # +(v_j2 - v_i2)*x2_i
            row[16 + i] = (v2_j - v2_i)
            
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    bounds = []
    for i in range(32):  # x1 and x2 in [0, 1]
        bounds.append((0, 1))
    for i in range(16):  # u can be any real number, but IR says u >= 0
        bounds.append((0, None))
        
    # 3. Solve
    from scipy.optimize import linprog
    
    try:
        # Use HiGHS solver with time limit
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Not enough time, return trivial solution
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {
                "allocations": allocations,
                "payments": payments
            }
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'time_limit': max(1, remaining_time - 2)})
        
        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            utilities = np.zeros(16)
            
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                utilities[i] = x[32 + i]
            
            # Compute payments: p_i = v_i.x_i - u_i
            payments = np.zeros(16)
            for i in range(16):
                v1, v2 = grid_values[i]
                payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - utilities[i]
            
            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            return {
                "allocations": allocations,
                "payments": payments
            }
        else:
            # Fallback if LP fails
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {
                "allocations": allocations,
                "payments": payments
            }
    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {
            "allocations": allocations,
            "payments": payments
        }

# EVOLVE-BLOCK-END
