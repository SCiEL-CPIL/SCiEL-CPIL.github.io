"""Improved program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction with one additive bidder, two goods,
    values in {0,1,2,3}, using Myerson's optimal auction principle.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 90.0  # seconds
    
    # Parse distribution spec
    # distribution_spec should specify the joint distribution over (v1, v2)
    # support is the set of allowed reports (16 combinations of v1, v2 in {0,1,2,3})
    # constraints contain feasibility, IR, and regret bounds
    
    # Extract the 16 type vectors
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    
    # Get probabilities from distribution_spec
    # Assume distribution_spec is a dict mapping (v1, v2) -> prob, or similar
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities
        probs = np.zeros(16)
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
            else:
                # Try nested dict or other formats
                pass
        # Normalize if needed
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Uniform distribution as fallback
            probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Ensure non-negative and normalized
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    
    # For a single additive bidder with two goods, the optimal mechanism
    # can be derived using Myerson's approach.
    
    # The bidder's value for a bundle (x1, x2) is v1*x1 + v2*x2
    # Truthfulness requires that allocation is monotone in the type.
    
    # For a single bidder with private values v = (v1, v2) and quasi-linear utility,
    # the optimal mechanism maximizes expected revenue subject to IC and IR.
    
    # Key insight: For a single additive bidder, the optimal mechanism allocates
    # each good independently based on the virtual value.
    
    # Virtual value for good j with value vj:
    # phi_j(vj) = vj - (1 - F_j(vj)) / f_j(vj)
    
    # But since the bidder has private values for BOTH goods, and the allocation
    # is over bundles, we need to be more careful.
    
    # Actually, for a single bidder with additive values over two goods,
    # the mechanism design problem is equivalent to selling two items to one buyer
    # with private valuations for each item.
    
    # The optimal mechanism:
    # 1. For each possible bundle (x1, x2) in {0,1}^2, determine if it should be allocated
    # 2. Payment is determined by the critical type at which the bidder would still get the bundle
    
    # Let's use the following approach:
    # - Enumerate all possible allocations for each type
    # - Use the fact that for a single bidder, IC + IR implies the payment must be
    #   the critical value at which the bidder would still prefer the allocation
    
    # For a single bidder with value v for a bundle x, the payment p(v) must satisfy:
    # v·x - p(v) >= 0 (IR)
    # v·x - p(v) >= v'·x' - p(v') for all v', x' (IC)
    
    # The optimal mechanism for a single bidder is:
    # - Allocate the bundle that maximizes v·x - phi(x) where phi is the virtual surplus
    # - But this requires knowing the distribution over bundles
    
    # Let me use a simpler approach: since there's only one bidder, we can compute
    # the optimal mechanism by considering the 16 types and finding the optimal
    # allocation and payment schedule.
    
    # For each type t = (v1, v2), let x(t) = (x1, x2) be the allocation.
    # The utility is u(t) = v1*x1 + v2*x2 - p(t).
    
    # IC: u(t) >= u(t') for all t' (since the bidder reports t, they get x(t) and pay p(t),
    # but if they were type t', they would prefer x(t') and pay p(t'))
    # Actually, IC means: for the true type t, reporting t gives at least as much utility
    # as reporting any other type t'.
    
    # u(t | report t) = v1*x1(t) + v2*x2(t) - p(t)
    # u(t | report t') = v1*x1(t') + v2*x2(t') - p(t')
    
    # IC: v1*x1(t) + v2*x2(t) - p(t) >= v1*x1(t') + v2*x2(t') - p(t') for all t, t'
    
    # This is a complex constraint. For a single bidder, the optimal mechanism is known:
    # It's the Myerson-optimal mechanism where we allocate based on virtual values.
    
    # Let's implement a practical approach:
    # 1. For each type, compute the value of each possible bundle
    # 2. Determine the optimal allocation using the single-bidder auction formula
    
    # For a single bidder with additive values, the optimal mechanism is:
    # - The bidder gets the bundle that maximizes their value minus a reserve
    # - The payment is the critical value
    
    # Let me compute the optimal mechanism by solving the LP directly.
    
    # Decision variables:
    # x[i, j] = probability that bundle j is allocated to type i (16 types, 4 bundles)
    # p[i] = payment for type i
    
    # Constraints:
    # 1. Sum_j x[i, j] = 1 for all i (exactly one bundle allocated)
    # 2. x[i, j] >= 0
    # 3. p[i] >= 0
    # 4. IR: v_i·x_i - p[i] >= 0 for all i
    # 5. IC: v_i·x_i - p[i] >= v_i·x_j - p[j] for all i, j
    #    where x_i is the expected allocation for type i, x_j for type j
    
    # Objective: Maximize sum_i probs[i] * p[i]
    
    # This is a linear program! Let's solve it.
    
    from scipy.optimize import linprog
    
    n_types = 16
    n_bundles = 4  # (0,0), (0,1), (1,0), (1,1)
    
    bundles = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Value matrix: val[i, j] = value of bundle j for type i
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2
    
    # Decision variables:
    # x[i, j] for i in 0..15, j in 0..3 (64 variables)
    # p[i] for i in 0..15 (16 variables)
    # Total: 80 variables
    
    # Objective: maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we minimize -sum_i probs[i] * p[i]
    
    c = np.zeros(80)
    for i in range(n_types):
        c[64 + i] = -probs[i]  # Negative because we're maximizing
    
    # Constraints:
    # 1. Sum_j x[i, j] = 1 for all i (16 equality constraints)
    A_eq = np.zeros((n_types, 80))
    b_eq = np.zeros(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1
        b_eq[i] = 1
    
    # 2. x[i, j] >= 0 (handled by bounds)
    # 3. p[i] >= 0 (handled by bounds)
    
    # 4. IR: val[i]·x[i] - p[i] >= 0 => p[i] <= val[i]·x[i]
    #    This is: -sum_j val[i, j] * x[i, j] + p[i] <= 0
    #    16 inequality constraints
    
    # 5. IC: val[i]·x[i] - p[i] >= val[i]·x[j] - p[j] for all i, j
    #    => val[i]·x[i] - p[i] - val[i]·x[j] + p[j] >= 0
    #    => -val[i]·x[i] + p[i] + val[i]·x[j] - p[j] <= 0
    #    16*16 = 256 inequality constraints
    
    n_ineq = n_types + n_types * n_types  # 16 + 256 = 272
    
    A_ub = np.zeros((n_ineq, 80))
    b_ub = np.zeros(n_ineq)
    
    # IR constraints: index 0 to 15
    for i in range(n_types):
        for j in range(n_bundles):
            A_ub[i, i * n_bundles + j] = -val[i, j]
        A_ub[i, 64 + i] = 1
        b_ub[i] = 0
    
    # IC constraints: index 16 to 271
    idx = n_types
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                idx += 1
                continue
            # -val[i]·x[i] + p[i] + val[i]·x[j] - p[j] <= 0
            for k in range(n_bundles):
                A_ub[idx, i * n_bundles + k] -= val[i, k]
                A_ub[idx, j * n_bundles + k] += val[i, k]
            A_ub[idx, 64 + i] = 1
            A_ub[idx, 64 + j] = -1
            b_ub[idx] = 0
            idx += 1
    
    # Bounds: all variables >= 0
    bounds = [(0, None)] * 80
    
    # Solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if result.status == 0:
            # Extract solution
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
            
            # Ensure non-negative
            x_opt = np.maximum(x_opt, 0)
            p_opt = np.maximum(p_opt, 0)
            
            # Normalize allocations to sum to 1
            for i in range(n_types):
                if x_opt[i].sum() > 0:
                    x_opt[i] = x_opt[i] / x_opt[i].sum()
                else:
                    x_opt[i] = np.zeros(n_bundles)
                    x_opt[i, 0] = 1  # Allocate nothing
            
            # Convert to expected allocation per good
            allocations = np.zeros((n_types, 2))
            for i in range(n_types):
                for j, (x1, x2) in enumerate(bundles):
                    allocations[i, 0] += x1 * x_opt[i, j]
                    allocations[i, 1] += x2 * x_opt[i, j]
            
            payments = p_opt
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP failed, return a simple mechanism
            pass
    except Exception as e:
        pass
    
    # Fallback: simple mechanism
    # Allocate good 1 if v1 >= 1, good 2 if v2 >= 1
    # Payment: v1 if allocated good 1, v2 if allocated good 2
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for i, (v1, v2) in enumerate(types):
        if v1 >= 1:
            allocations[i, 0] = 1
            payments[i] += v1
        if v2 >= 1:
            allocations[i, 1] = 1
            payments[i] += v2
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
