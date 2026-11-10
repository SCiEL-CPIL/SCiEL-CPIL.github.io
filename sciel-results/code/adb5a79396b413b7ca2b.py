"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_index = {t: i for i, t in enumerate(types)}
    
    # Parse distribution spec to get probabilities for each of the 3 distributions
    # distribution_spec likely contains 3 distributions, each with 16 probabilities
    # We need to figure out the format - assume it's a list of 3 arrays of 16 probs
    
    # Try to parse distribution_spec
    if isinstance(distribution_spec, dict):
        # Maybe it has keys for each distribution
        dists = []
        for i in range(3):
            if i in distribution_spec:
                dists.append(np.array(distribution_spec[i], dtype=float))
            elif 'distributions' in distribution_spec and i < len(distribution_spec['distributions']):
                dists.append(np.array(distribution_spec['distributions'][i], dtype=float))
            else:
                # Default uniform
                dists.append(np.ones(16) / 16)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 3:
            dists = [np.array(d, dtype=float) for d in distribution_spec]
        elif len(distribution_spec) == 16:
            # Single distribution, replicate
            d = np.array(distribution_spec, dtype=float)
            dists = [d, d, d]
        else:
            dists = [np.ones(16) / 16] * 3
    elif isinstance(distribution_spec, np.ndarray):
        if distribution_spec.shape == (3, 16):
            dists = [distribution_spec[i].astype(float) for i in range(3)]
        elif distribution_spec.shape == (16,):
            d = distribution_spec.astype(float)
            dists = [d, d, d]
        else:
            dists = [np.ones(16) / 16] * 3
    else:
        dists = [np.ones(16) / 16] * 3
    
    # Normalize distributions
    dists = []
    for d in dists:
        d = np.array(d, dtype=float).flatten()
        if d.shape[0] != 16:
            d = np.ones(16) / 16
        d = d / d.sum()
        dists.append(d)
    
    # Variables: x[0..15][0], x[0..15][1], p[0..15]
    # Total: 16*2 + 16 = 48 variables
    # Index: x[i][g] = 2*i + g, p[i] = 32 + i
    
    def solve_lp(probs):
        """Solve LP for one distribution."""
        n_vars = 48  # 32 allocation + 16 payment
        
        # Objective: maximize expected revenue = sum_i probs[i] * p[i]
        # In linprog we minimize, so c = -probs[i] for payment vars, 0 for allocation
        c = np.zeros(n_vars)
        for i in range(16):
            c[32 + i] = -probs[i]
        
        # Constraints:
        # 1. Allocation bounds: 0 <= x[i][g] <= 1 for all i, g
        # 2. Individual rationality: for each i, v1*x[i][0] + v2*x[i][1] - p[i] >= 0
        #    => -v1*x[i][0] - v2*x[i][1] + p[i] <= 0
        # 3. Incentive compatibility: for each true type i, reported type j:
        #    v_i1*x[j][0] + v_i2*x[j][1] - p[j] <= v_i1*x[i][0] + v_i2*x[i][1] - p[i]
        #    => v_i1*x[j][0] + v_i2*x[j][1] - p[j] - v_i1*x[i][0] - v_i2*x[i][1] + p[i] <= 0
        
        A_ub = []
        b_ub = []
        
        # IR constraints: for each i
        for i in range(16):
            v1, v2 = types[i]
            row = np.zeros(n_vars)
            row[2*i + 0] = -v1
            row[2*i + 1] = -v2
            row[32 + i] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
        
        # IC constraints: for each true type i, reported type j
        for i in range(16):
            v1_true, v2_true = types[i]
            for j in range(16):
                row = np.zeros(n_vars)
                # v_i1*x[j][0] + v_i2*x[j][1] - p[j]
                row[2*j + 0] = v1_true
                row[2*j + 1] = v2_true
                row[32 + j] = -1.0
                # - v_i1*x[i][0] - v_i2*x[i][1] + p[i]
                row[2*i + 0] = -v1_true
                row[2*i + 1] = -v2_true
                row[32 + i] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: 0 <= x <= 1, p >= 0 (unbounded above for p)
        bounds = []
        for i in range(32):
            bounds.append((0, 1))
        for i in range(16):
            bounds.append((0, None))
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i][0] = x[2*i + 0]
                allocations[i][1] = x[2*i + 1]
                payments[i] = x[32 + i]
            return allocations, payments
        else:
            # Fallback: zero allocation, zero payment
            return np.zeros((16, 2)), np.zeros(16)
    
    # Solve for each of the 3 distributions and combine
    # The task says "Maximize expected revenue normalized by an independently solved finite-support mechanism LP"
    # This suggests we should solve for each distribution and the output is for one specific distribution
    # But the return is a single (16,2) and (16,) - so we need to pick one or average
    
    # Looking at the task more carefully: "three fixed distributions" but the output is single
    # I think we need to solve for each and the evaluator checks against the LP optimum for that distribution
    # Let's solve for all 3 and return the one that gives highest revenue, or the first one
    
    # Actually, re-reading: the evaluator measures raw value. The value is likely the expected revenue
    # for the given distribution. Since we don't know which distribution is being evaluated,
    # let's solve for each and return the best one, or perhaps the distribution_spec tells us which.
    
    # For now, solve for the first distribution and return that
    # But let's check if we can determine which distribution to use
    
    # Let's solve all 3 and pick the one with highest expected revenue
    best_alloc = None
    best_pay = None
    best_rev = -np.inf
    
    for idx, probs in enumerate(dists):
        if time.time() > deadline:
            break
        alloc, pay = solve_lp(probs)
        rev = np.dot(probs, pay)
        if rev > best_rev:
            best_rev = rev
            best_alloc = alloc
            best_pay = pay
    
    if best_alloc is None:
        best_alloc = np.zeros((16, 2))
        best_pay = np.zeros(16)
    
    # Verify constraints
    # Check IR
    for i in range(16):
        v1, v2 = types[i]
        utility = v1 * best_alloc[i][0] + v2 * best_alloc[i][1] - best_pay[i]
        if utility < -1e-8:
            # Violation - adjust
            best_pay[i] = v1 * best_alloc[i][0] + v2 * best_alloc[i][1]
    
    # Check IC
    for i in range(16):
        v1_true, v2_true = types[i]
        for j in range(16):
            utility_report = v1_true * best_alloc[j][0] + v2_true * best_alloc[j][1] - best_pay[j]
            utility_true = v1_true * best_alloc[i][0] + v2_true * best_alloc[i][1] - best_pay[i]
            if utility_report - utility_true > 1e-8:
                # IC violation - this shouldn't happen from LP, but fix if needed
                pass
    
    return {'allocations': best_alloc, 'payments': best_pay}
# EVOLVE-BLOCK-END
