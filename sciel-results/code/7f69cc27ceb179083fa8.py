import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism for a single additive bidder
    with two goods, values in {0,1,2,3}. Returns allocations (16,2) and payments (16,).
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
        total = 1.0
    probs = probs / total

    # Value of each type (additive: v1 + v2)
    values = np.array([v1 + v2 for v1, v2 in types], dtype=float)

    # LP formulation for single additive bidder with two goods
    # Variables: x_0..x_15 (allocation probability of bundle), p_0..p_15 (payments)
    #
    # For a single additive bidder, the optimal mechanism allocates the bundle (both goods)
    # with some probability x_i. The allocation for each good is the same as x_i.
    #
    # Objective: maximize sum_i probs[i] * p_i
    #
    # Constraints:
    # 1. 0 <= x_i <= 1
    # 2. p_i >= 0
    # 3. IR: p_i <= v_i * x_i (utility >= 0)
    # 4. IC: v_i * x_i - p_i >= v_i * x_j - p_j for all i,j
    #    => p_j - p_i >= v_i * (x_j - x_i)
    #    => -v_i * x_i + v_i * x_j + p_i - p_j <= 0

    n_vars = 2 * n  # x_0..x_15, p_0..p_15

    # Objective: minimize -sum(probs[i] * p_i)
    c = np.zeros(n_vars)
    for i in range(n):
        c[n + i] = -probs[i]

    # Inequality constraints: A_ub @ vars <= b_ub
    A_ub_rows = []
    b_ub_vals = []

    # x_i <= 1
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

    # -x_i <= 0 (x_i >= 0)
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # -p_i <= 0 (p_i >= 0)
    for i in range(n):
        row = np.zeros(n_vars)
        row[n + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IR: p_i - v_i * x_i <= 0  =>  -v_i * x_i + p_i <= 0
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -values[i]
        row[n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IC: For all i != j: v_i * x_i - p_i >= v_i * x_j - p_j
    #    => -v_i * x_i + v_i * x_j + p_i - p_j <= 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -values[i]
            row[j] = values[i]
            row[n + i] = 1.0
            row[n + j] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    bounds = [(0, None)] * n_vars

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-9,
                              'primal_feasibility_tolerance': 1e-9})

    # Check time
    if time.time() > deadline - 5:
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: allocate nothing, collect nothing
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}

    x = result.x[:n]
    p = result.x[n:]

    # Clean up numerical noise
    x = np.clip(x, 0.0, 1.0)
    p = np.clip(p, 0.0, None)

    # Enforce IR strictly: p_i <= v_i * x_i
    for i in range(n):
        if values[i] > 0:
            max_p = values[i] * x[i]
            if p[i] > max_p + 1e-9:
                p[i] = max_p
        else:
            p[i] = 0.0

    # Enforce nonnegativity
    p = np.clip(p, 0.0, None)
    x = np.clip(x, 0.0, 1.0)

    # Verify IC constraints with regret check
    max_regret = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Utility of type i reporting i: v_i * x_i - p_i
            # Utility of type i reporting j: v_i * x_j - p_j
            util_i = values[i] * x[i] - p[i]
            util_j = values[i] * x[j] - p[j]
            regret = util_j - util_i
            if regret > max_regret:
                max_regret = regret

    # If regret exceeds tolerance, we need to fix the mechanism
    if max_regret > 1e-8:
        # Use the envelope theorem approach: fix x and compute minimal payments
        # that satisfy IC and IR
        # For single bidder, the optimal payment given allocation x is:
        # p_i = min over paths from 0 to i of sum of v_k * (x_k - x_{k-1})
        # where we start from a type with v=0
        
        # Find type with value 0 (index 0: v1=0, v2=0)
        # Use dynamic programming / shortest path on the type lattice
        
        # Actually, for a single additive bidder with values v1+v2, the IC constraints
        # imply monotonicity of x. The payment is determined by the envelope condition:
        # p_i = u_i + v_i * x_i, where u_i is the utility of type i
        # and u_0 = 0 (IR binding at lowest type)
        
        # For the lattice structure, we can compute u_i using:
        # u_i = min over all j < i (in some ordering) of u_j + integral of x along path
        # But for discrete types, u_i >= u_j + (v_i - v_j) * x_i if we move in one direction
        
        # Simpler: just re-solve with tighter constraints or use the LP result directly
        # and adjust payments to be IC
        
        # Compute minimal IC payments given x:
        # u_0 = 0 (type with v=0)
        # For each type i, u_i = max(0, max over j: u_j + v_i*(x_i - x_j) - v_j*(x_i - x_j))
        # Actually: IC says u_i >= u_j + v_i*(x_i - x_j) for all j where we can compare
        
        # Let's use the standard formula: u_i = max(0, max_{j} [u_j + v_i * x_i - v_j * x_j - (v_i - v_j)*x_j])
        # This is getting complex. Let's just use the LP solution and trust it.
        
        # If IC is violated, scale down payments slightly
        # Better: recompute payments using the envelope theorem
        # For single agent, p_i = v_i * x_i - u_i where u_i is utility
        # u_i must satisfy: u_i >= u_j + (v_i - v_j) * x_i for all j (monotonicity of x)
        # and u_0 = 0 (IR at zero value)
        
        # Let's compute u using a shortest path approach on the type graph
        # Edge from j to i exists if we can go from j to i
        # u_i >= u_j + v_i * (x_i - x_j) for all j
        
        # This is equivalent to: u_i = max_j (u_j + v_i * x_i - v_i * x_j)
        #                       = v_i * x_i + max_j (u_j - v_i * x_j)
        
        # Initialize u with infinity, u[0] = 0 (type 0 has v=0)
        u = np.full(n, np.inf)
        u[0] = 0.0
        
        # Iterate to convergence
        for _ in range(100):
            changed = False
            for i in range(n):
                if i == 0:
                    continue
                best = u[0] + values[i] * x[i] - values[0] * x[0]
                for j in range(n):
                    if j == i or j == 0:
                        continue
                    candidate = u[j] + values[i] * (x[i] - x[j])
                    if candidate > best:
                        best = candidate
                if best > u[i] + 1e-12:
                    u[i] = best
                    changed = True
            if not changed:
                break
        
        # Recompute payments
        for i in range(n):
            p[i] = values[i] * x[i] - u[i]
        
        # Enforce IR: p_i <= v_i * x_i => u_i >= 0
        for i in range(n):
            if u[i] < 0:
                u[i] = 0.0
                p[i] = values[i] * x[i]
            p[i] = max(0.0, p[i])

    # Final verification and cleanup
    x = np.clip(x, 0.0, 1.0)
    p = np.clip(p, 0.0, None)
    
    # Ensure IR one more time
    for i in range(n):
        if values[i] > 0:
            p[i] = min(p[i], values[i] * x[i])
        else:
            p[i] = 0.0
        p[i] = max(0.0, p[i])

    # Construct allocations: both goods allocated with probability x_i
    allocations = np.column_stack([x, x])

    return {'allocations': allocations, 'payments': p}
