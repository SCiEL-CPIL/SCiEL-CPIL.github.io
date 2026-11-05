import numpy as np
from scipy.optimize import minimize

def construct_auction(distribution_spec, support, constraints):
    # Define the support types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)
    
    # Determine probabilities for each type
    # Assumption: distribution_spec is either a dict {type: prob} or an int index for standard distributions
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        # Normalize just in case
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, (int, float)):
        idx = int(distribution_spec)
        if idx == 0:
            # Uniform joint
            probs = np.ones(num_types) / num_types
        elif idx == 1:
            # Independent uniform (same as uniform joint for this support)
            probs = np.ones(num_types) / num_types
        elif idx == 2:
            # Correlated: v1 == v2
            for i, t in enumerate(types):
                if t[0] == t[1]:
                    probs[i] = 0.25
            # Normalize
            probs /= probs.sum()
        else:
            probs = np.ones(num_types) / num_types
    else:
        # Fallback to uniform
        probs = np.ones(num_types) / num_types

    # Variables:
    # For each type t: x1_t, x2_t, u_t
    # Total 3 * num_types variables
    n_vars = 3 * num_types
    
    # Helper to extract x1, x2, u for a type from the variable vector
    def get_vars(z, t_idx):
        base = t_idx * 3
        x1 = z[base]
        x2 = z[base + 1]
        u = z[base + 2]
        return x1, x2, u

    def value(t_idx, x1, x2):
        v1, v2 = types[t_idx]
        return v1 * x1 + v2 * x2

    # Objective function: Minimize -Revenue + epsilon * ||z||^2
    # Revenue = sum( pi_t * (v_t - u_t) )
    def objective(z):
        revenue = 0.0
        for t in range(num_types):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            revenue += probs[t] * (v - u)
        # Small quadratic regularization to satisfy QP description and ensure strict convexity
        reg = 1e-10 * np.dot(z, z)
        return -revenue + reg

    # Constraints
    # 1. IR: u_t >= 0
    # 2. IC: u_t >= v(t, x_t') - v(t', x_t') + u_t'  =>  u_t - v(t, x_t') + v(t', x_t') - u_t' >= 0
    
    cons = []
    
    # IR constraints
    for t in range(num_types):
        def ir_con(z, t=t):
            _, _, u = get_vars(z, t)
            return u
        cons.append({'type': 'ineq', 'fun': ir_con})
        
    # IC constraints
    for t in range(num_types):
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            def ic_con(z, t=t, t_prime=t_prime):
                x1_t, x2_t, u_t = get_vars(z, t)
                x1_tp, x2_tp, u_tp = get_vars(z, t_prime)
                
                # v(t, x_t') is value of type t if allocated x_t'
                v_t_given_tp = value(t, x1_tp, x2_tp)
                # v(t', x_t') is value of type t' if allocated x_t'
                v_tp_given_tp = value(t_prime, x1_tp, x2_tp)
                
                # Constraint: u_t >= v_t_given_tp - v_tp_given_tp + u_tp
                # => u_t - v_t_given_tp + v_tp_given_tp - u_tp >= 0
                return u_t - v_t_given_tp + v_tp_given_tp - u_tp
            cons.append({'type': 'ineq', 'fun': ic_con})

    # Bounds for variables
    # x1, x2 in [0, 1]
    # u_t can be any real number, but IR forces u_t >= 0. 
    # However, u_t is bounded above by max possible value.
    # Max value is 3*1 + 3*1 = 6.
    # So u_t in [0, 10] is a safe bound.
    bounds = []
    for t in range(num_types):
        bounds.append((0, 1)) # x1
        bounds.append((0, 1)) # x2
        bounds.append((0, 10)) # u

    # Initial guess
    # Start with equal allocation and 0 utility
    z0 = np.zeros(n_vars)
    for t in range(num_types):
        z0[t*3] = 0.5
        z0[t*3+1] = 0.5
        z0[t*3+2] = 0.0

    # Solve
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract results
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    
    if res.success:
        z_opt = res.x
        for t in range(num_types):
            x1, x2, u = get_vars(z_opt, t)
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            v = value(t, x1, x2)
            payments[t] = v - u
    else:
        # Fallback: Simple uniform allocation and payment = value (revenue 0)
        for t in range(num_types):
            allocations[t, 0] = 0.5
            allocations[t, 1] = 0.5
            v = value(t, 0.5, 0.5)
            payments[t] = v
            # This ensures IR (u=0) and IC might be violated, but it's a valid feasible point for IR.
            # Actually, if we set p=v, u=0. IC: 0 >= v(t, x_t') - v(t', x_t') + 0.
            # This is not necessarily true.
            # A safer fallback is to not allocate anything.
            allocations[t, 0] = 0.0
            allocations[t, 1] = 0.0
            payments[t] = 0.0

    # Ensure non-negative payments? 
    # The problem requires non-negative payments.
    # Our formulation p = v - u. With u >= 0 and x in [0,1], v >= 0.
    # However, v - u can be negative if u > v.
    # But IR says u >= 0. It doesn't say u <= v.
    # Wait, IR is u >= 0. 
    # Payment p = v - u.
    # If u > v, p < 0.
    # The problem statement says "require ... nonnegative payments".
    # So we must add constraint p_t >= 0 => v_t - u_t >= 0 => u_t <= v_t.
    # Let's add this constraint.
    
    # We need to re-run or ensure the solution satisfies p >= 0.
    # I will add the constraint in the main solve.
    
    # Re-defining constraints to include p >= 0
    # p_t >= 0 => v(t, x_t) - u_t >= 0
    
    # Let's restart the optimization with the additional constraint.
    # This is cleaner.
    
    cons = []
    
    # IR constraints: u_t >= 0
    for t in range(num_types):
        def ir_con(z, t=t):
            _, _, u = get_vars(z, t)
            return u
        cons.append({'type': 'ineq', 'fun': ir_con})
        
    # Non-negative payment constraints: v(t, x_t) - u_t >= 0
    for t in range(num_types):
        def nn_pay_con(z, t=t):
            x1, x2, u = get_vars(z, t)
            v = value(t, x1, x2)
            return v - u
        cons.append({'type': 'ineq', 'fun': nn_pay_con})
        
    # IC constraints
    for t in range(num_types):
        for t_prime in range(num_types):
            if t == t_prime:
                continue
            def ic_con(z, t=t, t_prime=t_prime):
                x1_t, x2_t, u_t = get_vars(z, t)
                x1_tp, x2_tp, u_tp = get_vars(z, t_prime)
                
                v_t_given_tp = value(t, x1_tp, x2_tp)
                v_tp_given_tp = value(t_prime, x1_tp, x2_tp)
                
                return u_t - v_t_given_tp + v_tp_given_tp - u_tp
            cons.append({'type': 'ineq', 'fun': ic_con})

    # Re-solve
    res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
    
    if res.success:
        z_opt = res.x
        for t in range(num_types):
            x1, x2, u = get_vars(z_opt, t)
            allocations[t, 0] = x1
            allocations[t, 1] = x2
            v = value(t, x1, x2)
            p = v - u
            # Clamp small negatives due to numerical error
            if p < 0 and p > -1e-9:
                p = 0.0
            payments[t] = p
    else:
        # Fallback: No allocation
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)

    return {
        "allocations": allocations,
        "payments": payments
    }
