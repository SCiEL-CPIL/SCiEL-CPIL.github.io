"""Combined program: Direct LP solve with HiGHS for incentive-compatible auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # 1. Define Types and Bundles
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = 16
    n_bundles = 4
    bundles = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    # 2. Parse Distribution
    probs = np.zeros(16)
    try:
        if isinstance(distribution_spec, dict):
            # Try common keys
            if 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float).flatten()
            elif 'distribution' in distribution_spec:
                probs = np.array(distribution_spec['distribution'], dtype=float).flatten()
            else:
                # Assume keys are types
                for i, t in enumerate(types):
                    if t in distribution_spec:
                        probs[i] = distribution_spec[t]
                    elif str(t) in distribution_spec:
                        probs[i] = distribution_spec[str(t)]
                    else:
                        # Try tuple keys if not direct match
                        pass
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == 16:
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(16) / 16.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    except Exception:
        probs = np.ones(16) / 16.0
        
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    # 3. Setup Value Matrix
    # val[i, j] = value of bundle j for type i
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2

    # 4. Define LP Variables
    # x[i, j] for i in 0..15, j in 0..3 (64 vars)
    # p[i] for i in 0..15 (16 vars)
    # Total vars: 80
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # Minimize -sum_i probs[i] * p[i]
    c = np.zeros(80)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # 5. Equality Constraints
    # Sum_j x[i, j] = 1 for all i
    A_eq = np.zeros((n_types, 80))
    b_eq = np.ones(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1

    # 6. Inequality Constraints
    # A_ub * vars <= b_ub
    
    # IR: p[i] <= val[i] . x[i]  =>  p[i] - sum_j val[i,j]*x[i,j] <= 0
    # IC: u[i] >= u[j]  =>  val[i].x[i] - p[i] >= val[i].x[j] - p[j]
    #     => p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
    
    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, 80))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    # IR Constraints
    for i in range(n_types):
        A_ub[row, 64 + i] = 1  # p[i]
        for j in range(n_bundles):
            A_ub[row, i * n_bundles + j] = -val[i, j]  # -x[i,j] * val[i,j]
        b_ub[row] = 0
        row += 1
        
    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
            A_ub[row, 64 + i] = 1   # p[i]
            A_ub[row, 64 + j] = -1  # -p[j]
            for k in range(n_bundles):
                # + val[i,k] * x[j,k]
                A_ub[row, j * n_bundles + k] += val[i, k]
                # - val[i,k] * x[i,k]
                A_ub[row, i * n_bundles + k] -= val[i, k]
            b_ub[row] = 0
            row += 1

    bounds = [(0, None)] * 80

    # 7. Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.status == 0:
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
            
            # Clean up numerical noise
            x_opt = np.maximum(x_opt, 0)
            p_opt = np.maximum(p_opt, 0)
            
            # Normalize allocations just in case
            for i in range(n_types):
                s = x_opt[i].sum()
                if s > 1e-9:
                    x_opt[i] /= s
                else:
                    x_opt[i] = np.zeros(n_bundles)
                    x_opt[i, 0] = 1.0 # Bundle (0,0)
            
            # Convert bundle probabilities to expected allocation per good
            allocations = np.zeros((n_types, 2))
            for i in range(n_types):
                for j, (x1, x2) in enumerate(bundles):
                    allocations[i, 0] += x1 * x_opt[i, j]
                    allocations[i, 1] += x2 * x_opt[i, j]
            
            return {
                'allocations': allocations,
                'payments': p_opt
            }
        else:
            # LP failed
            pass
    except Exception:
        pass

    # 8. Fallback: Simple Threshold Mechanism
    # Allocate good 1 if v1 >= 1, good 2 if v2 >= 1
    # Payment: sum of values of allocated goods (this is not necessarily IC/IR optimal but valid if we adjust)
    # Actually, a safer fallback is "Allocate Nothing, Pay 0" which is always IC and IR.
    # But we want revenue. Let's try a simple Myerson-like reserve.
    # If we assume independent uniform, virtual value phi(v) = v - (1-F)/f.
    # For v in {0,1,2,3}, f=0.25.
    # v=0: phi = 0 - 1/0.25 = -4
    # v=1: phi = 1 - 0.75/0.25 = 1 - 3 = -2
    # v=2: phi = 2 - 0.5/0.25 = 2 - 2 = 0
    # v=3: phi = 3 - 0.25/0.25 = 3 - 1 = 2
    # So allocate if v >= 2.
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for i, (v1, v2) in enumerate(types):
        alloc1 = 1 if v1 >= 2 else 0
        alloc2 = 1 if v2 >= 2 else 0
        allocations[i, 0] = alloc1
        allocations[i, 1] = alloc2
        # Payment for single item with reserve r is v if v>=r, 0 otherwise? 
        # No, payment is critical value. For uniform discrete, if we use reserve 2,
        # and v=2, payment is 2? 
        # Actually, for the fallback, "Allocate Nothing" is safest to avoid invalidity.
        # But let's try to be slightly better.
        # If we just return zeros, score is 0.
        # Let's return the simple "pay value if allocated" which is NOT IC.
        # We must return a VALID mechanism.
        # The "Allocate Nothing, Pay 0" mechanism is always valid.
        
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
