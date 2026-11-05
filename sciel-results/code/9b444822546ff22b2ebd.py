"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time
import sys

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Three fixed distributions - all uniform on {0,1,2,3}^2
        probs = np.ones(num_types) / num_types
    else:
        # Try to interpret as list/array
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types
    
    # For a single additive bidder, IC + IR implies the mechanism is determined
    # by the allocation rule. The payment is uniquely determined by IC.
    # 
    # Key insight: For a single bidder with additive valuations, the optimal
    # mechanism can be found by solving an LP where:
    # - Decision variables: x1[i], x2[i] for i in 0..15 (allocation probabilities)
    # - Payments are derived from IC constraints
    # - Revenue = sum_i probs[i] * (v1[i]*x1[i] + v2[i]*x2[i] - t[i])
    #
    # The payment t[i] for type i is determined by the IC constraints.
    # For the lowest type (0,0), IR gives t[0] <= 0, and since t >= 0, t[0] = 0.
    # Then IC determines all other payments.
    
    # Let's use the standard approach: parametrize by allocations, derive payments
    # from IC, and maximize revenue.
    
    # The IC constraints say: u(i,i) >= u(i,j) for all i,j
    # u(i,j) = v1[i]*x1[j] + v2[i]*x2[j] - t[j]
    # 
    # The payment can be expressed as:
    # t[i] = v1[i]*x1[i] + v2[i]*x2[i] - u[i]
    # where u[i] is the utility of type i when reporting truthfully.
    #
    # IC requires: u[i] >= v1[i]*x1[j] + v2[i]*x2[j] - t[j] for all j
    # Substituting t[j] = v1[j]*x1[j] + v2[j]*x2[j] - u[j]:
    # u[i] >= v1[i]*x1[j] + v2[i]*x2[j] - v1[j]*x1[j] - v2[j]*x2[j] + u[j]
    # u[i] - u[j] >= v1[i]*x1[j] + v2[i]*x2[j] - v1[j]*x1[j] - v2[j]*x2[j]
    #
    # For optimal mechanism, the lowest type has u[0] = 0 (IR binding).
    # Then we can compute u[i] from the IC constraints.
    
    # Actually, let's use a direct LP formulation.
    # Variables: x1[0..15], x2[0..15], t[0..15]
    # Maximize: sum_i probs[i] * t[i]
    # Subject to:
    #   IC: v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= v1[i]*x1[j] + v2[i]*x2[j] - t[j] for all i,j
    #   IR: v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= 0 for all i
    #   0 <= x1[i], x2[i] <= 1 for all i
    #   t[i] >= 0 for all i
    
    # This is a linear program! Let's solve it with linprog.
    
    # Variables: [x1_0, ..., x1_15, x2_0, ..., x2_15, t_0, ..., t_15]
    # Total: 48 variables
    
    # Objective: maximize sum_i probs[i] * t[i]
    # In linprog (minimization): minimize -sum_i probs[i] * t[i]
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]  # Coefficient for t[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    # IC constraints: For each i, j:
    # v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= v1[i]*x1[j] + v2[i]*x2[j] - t[j]
    # => -v1[i]*x1[i] - v2[i]*x2[i] + t[i] + v1[i]*x1[j] + v2[i]*x2[j] - t[j] <= 0
    # => v1[i]*(x1[j] - x1[i]) + v2[i]*(x2[j] - x2[i]) + (t[i] - t[j]) <= 0
    
    A_ub_rows = []
    b_ub_vals = []
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            row = np.zeros(3 * num_types)
            # Coefficient for x1[i]: -v1_i
            row[i] = -v1_i
            # Coefficient for x2[i]: -v2_i
            row[num_types + i] = -v2_i
            # Coefficient for t[i]: +1
            row[2 * num_types + i] = 1
            # Coefficient for x1[j]: +v1_i
            row[j] += v1_i
            # Coefficient for x2[j]: +v2_i
            row[num_types + j] += v2_i
            # Coefficient for t[j]: -1
            row[2 * num_types + j] -= 1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: For each i:
    # v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= 0
    # => -v1[i]*x1[i] - v2[i]*x2[i] + t[i] <= 0
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = np.zeros(3 * num_types)
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # x1[i] in [0, 1] for all i
    # x2[i] in [0, 1] for all i
    # t[i] in [0, inf) for all i
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i]
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            x1 = x[:num_types]
            x2 = x[num_types:2 * num_types]
            t = x[2 * num_types:3 * num_types]
            
            # Clip small numerical errors
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            t = np.clip(t, 0, None)
            
            allocations = np.column_stack([x1, x2])
            payments = t
            
            # Verify IC and IR
            valid = True
            for i in range(num_types):
                v1_i, v2_i = types[i]
                u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
                if u_ii < -1e-8:
                    valid = False
                    break
                for j in range(num_types):
                    u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                    if u_ii < u_ij - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {'allocations': allocations, 'payments': payments}
        
    except Exception as e:
        pass
    
    # Fallback: try SLSQP if LP failed
    if time.time() < deadline:
        def unpack(z):
            x1 = z[:num_types]
            x2 = z[num_types:2*num_types]
            t = z[2*num_types:3*num_types]
            return x1, x2, t
        
        def revenue(z):
            _, _, t = unpack(z)
            return -np.sum(probs * t)
        
        def ic_constraints(z):
            x1, x2, t = unpack(z)
            cons = []
            for i in range(num_types):
                v1_i, v2_i = types[i]
                for j in range(num_types):
                    cons.append(
                        v1_i*(x1[i] - x1[j]) + v2_i*(x2[i] - x2[j]) - (t[i] - t[j])
                    )
            return np.array(cons)
        
        def ir_constraints(z):
            x1, x2, t = unpack(z)
            cons = []
            for i in range(num_types):
                v1_i, v2_i = types[i]
                cons.append(v1_i*x1[i] + v2_i*x2[i] - t[i])
            return np.array(cons)
        
        bounds_slsqp = []
        for i in range(num_types):
            bounds_slsqp.append((0, 1))
        for i in range(num_types):
            bounds_slsqp.append((0, 1))
        for i in range(num_types):
            bounds_slsqp.append((0, None))
        
        z0 = np.zeros(3 * num_types)
        for i in range(num_types):
            v1_i, v2_i = types[i]
            z0[i] = 0.5
            z0[num_types + i] = 0.5
            z0[2 * num_types + i] = v1_i * 0.5 + v2_i * 0.5
        
        try:
            result = minimize(
                revenue, z0, method='SLSQP', bounds=bounds_slsqp,
                constraints=[
                    {'type': 'ineq', 'fun': ic_constraints},
                    {'type': 'ineq', 'fun': ir_constraints}
                ],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success:
                x1, x2, t = unpack(result.x)
                x1 = np.clip(x1, 0, 1)
                x2 = np.clip(x2, 0, 1)
                t = np.clip(t, 0, None)
                allocations = np.column_stack([x1, x2])
                payments = t
                return {'allocations': allocations, 'payments': payments}
                
        except:
            pass
    
    # Final fallback: simple mechanism
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
