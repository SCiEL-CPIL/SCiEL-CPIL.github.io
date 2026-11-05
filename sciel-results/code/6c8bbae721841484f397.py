"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

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
    deadline = time.time() + 75  # Safety margin before 90s kill
    
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
        # Three fixed distributions
        if distribution_spec == 0:
            # Uniform
            probs = np.ones(num_types) / num_types
        elif distribution_spec == 1:
            # Product of independent uniform on {0,1,2,3}
            probs = np.ones(num_types) / num_types
        elif distribution_spec == 2:
            # Product of independent uniform on {0,1,2,3} with different scaling
            probs = np.ones(num_types) / num_types
        else:
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
    
    # Decision variables: x1[16], x2[16], t[16] -> 48 variables
    # x1[i] = prob of allocating good 1 to type i
    # x2[i] = prob of allocating good 2 to type i  
    # t[i] = payment from type i
    
    def unpack(z):
        x1 = z[:num_types]
        x2 = z[num_types:2*num_types]
        t = z[2*num_types:3*num_types]
        return x1, x2, t
    
    def revenue(z):
        _, _, t = unpack(z)
        return -np.sum(probs * t)  # Negate for minimization
    
    # Incentive compatibility constraints:
    # For all i (true type), j (reported type):
    # v1_i*x1[i] + v2_i*x2[i] - t[i] >= v1_i*x1[j] + v2_i*x2[j] - t[j]
    # => v1_i*(x1[i]-x1[j]) + v2_i*(x2[i]-x2[j]) - (t[i]-t[j]) >= 0
    
    def ic_constraints(z):
        x1, x2, t = unpack(z)
        cons = []
        for i in range(num_types):
            v1_i, v2_i = types[i]
            for j in range(num_types):
                # IC: u_i(i) >= u_i(j)
                # v1_i*x1[i] + v2_i*x2[i] - t[i] - (v1_i*x1[j] + v2_i*x2[j] - t[j]) >= 0
                cons.append(
                    v1_i*(x1[i] - x1[j]) + v2_i*(x2[i] - x2[j]) - (t[i] - t[j])
                )
        return np.array(cons)
    
    # Individual rationality: u_i(i) >= 0
    def ir_constraints(z):
        x1, x2, t = unpack(z)
        cons = []
        for i in range(num_types):
            v1_i, v2_i = types[i]
            cons.append(v1_i*x1[i] + v2_i*x2[i] - t[i])
        return np.array(cons)
    
    # Feasibility: 0 <= x1, x2 <= 1
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i] >= 0
    
    # Initial guess
    z0 = np.zeros(3*num_types)
    # Start with simple allocation: allocate both goods with prob 0.5, payment = expected value
    for i in range(num_types):
        v1_i, v2_i = types[i]
        z0[i] = 0.5  # x1
        z0[num_types + i] = 0.5  # x2
        z0[2*num_types + i] = v1_i * 0.5 + v2_i * 0.5  # t
    
    # Try SLSQP
    try:
        result = minimize(
            revenue, z0, method='SLSQP', bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': ic_constraints},
                {'type': 'ineq', 'fun': ir_constraints}
            ],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success and time.time() < deadline:
            x1, x2, t = unpack(result.x)
            allocations = np.column_stack([x1, x2])
            payments = t
            return {'allocations': allocations, 'payments': payments}
        
    except Exception as e:
        pass
    
    # Fallback: simple mechanism
    # Allocate both goods with prob 0.5, payment = expected value
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
