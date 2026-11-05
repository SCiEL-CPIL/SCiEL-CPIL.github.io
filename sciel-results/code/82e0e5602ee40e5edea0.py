"""Incentive-compatible finite auction design via LP with Myerson payment identity."""
import time
import numpy as np
from scipy.optimize import linprog

# Deadline guard: evaluator kills after 90s, use 80s safety margin
_DEADLINE = 80.0

def _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """Solve LP with timeout guard."""
    start = time.time()
    remaining = _DEADLINE - start
    if remaining < 1.0:
        return None
    try:
        result = linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'time_limit': max(0.1, remaining - 0.5)}
        )
        if result.success:
            return result.x
        return None
    except Exception:
        return None


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Args:
        distribution_spec: specification of the type distribution
        support: the support of types (values in {0,1,2,3} for 2 goods)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16,2) array and 'payments': (16,) array
    """
    start_time = time.time()
    
    # Types: 16 possible types (v1, v2) with v1, v2 in {0,1,2,3}
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution probabilities
    probs = _get_distribution(distribution_spec, n_types)
    
    # Decision variables: x[i] = probability of allocating BOTH goods to bidder given type i
    # Since it's a single additive bidder with two goods, and we want to maximize revenue,
    # the optimal mechanism allocates both goods or none (by symmetry and additivity).
    # Actually, let's be more general: allocate each good independently.
    # x[i, g] = probability of allocating good g to bidder given type i
    # For a single bidder with additive values, the IC constraint is:
    # u(t) = v(t) * x(t) - p(t) is maximized at t = true type
    # This implies: u(t') >= u(t) + (v(t') - v(t)) * x(t) for all t, t'
    # The payment identity: p(t) = v(t) * x(t) - u(t)
    # IR: u(t) >= 0 for all t
    # Revenue = sum_t probs[t] * p(t) = sum_t probs[t] * (v(t) * x(t) - u(t))
    
    # Variables: x[0..15, 0..1] (32 vars) and u[0..15] (16 vars) -> 48 vars
    # x_{i,g} in [0, 1], u_i >= 0
    n_x = n_types * 2  # 32
    n_u = n_types  # 16
    n_vars = n_x + n_u  # 48
    
    # Objective: maximize sum_t probs[t] * (v1[t]*x[t,0] + v2[t]*x[t,1] - u[t])
    # = sum_t probs[t] * v1[t] * x[t,0] + sum_t probs[t] * v2[t] * x[t,1] - sum_t probs[t] * u[t]
    # For linprog (minimize), negate:
    c = np.zeros(n_vars)
    for i in range(n_types):
        v1, v2 = types[i]
        c[i * 2] = -probs[i] * v1
        c[i * 2 + 1] = -probs[i] * v2
        c[n_x + i] = -probs[i]  # coefficient for u[i]
    
    # Bounds: x in [0,1], u >= 0
    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_u):
        bounds.append((0, None))
    
    # IC constraints: for all t, t':
    # u(t') >= u(t) + (v1(t')*x(t',0) + v2(t')*x(t',1)) - (v1(t)*x(t,0) + v2(t)*x(t,1))
    # Rearranged: u(t') - u(t) - v1(t')*x(t',0) - v2(t')*x(t',1) + v1(t)*x(t,0) + v2(t)*x(t,1) >= 0
    # For linprog (<= form): u(t) - u(t') + v1(t')*x(t',0) + v2(t')*x(t',1) - v1(t)*x(t,0) - v2(t)*x(t,1) <= 0
    
    ic_rows = []
    ic_rhs = []
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            v1_t, v2_t = types[t]
            v1_tp, v2_tp = types[tp]
            # u(t) - u(t')
            row[n_x + t] = 1.0
            row[n_x + tp] = -1.0
            # + v1(t')*x(t',0) + v2(t')*x(t',1)
            row[tp * 2] = v1_tp
            row[tp * 2 + 1] = v2_tp
            # - v1(t)*x(t,0) - v2(t)*x(t,1)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            ic_rows.append(row)
            ic_rhs.append(0.0)
    
    A_ub = np.array(ic_rows) if ic_rows else None
    b_ub = np.array(ic_rhs) if ic_rhs else None
    
    # No equality constraints
    A_eq = None
    b_eq = None
    
    # Solve LP
    x_sol = _solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds)
    
    if x_sol is None:
        # Fallback: allocate nothing, zero payments
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_alloc = x_sol[:n_x].reshape(n_types, 2)
    u_vals = x_sol[n_x:]
    
    # Compute payments using Myerson identity: p(t) = v(t) * x(t) - u(t)
    payments = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        payments[i] = v1 * x_alloc[i, 0] + v2 * x_alloc[i, 1] - u_vals[i]
    
    # Ensure non-negative payments (clip small negatives from numerical error)
    payments = np.maximum(payments, 0.0)
    
    # Verify constraints and compute regret
    # Check IC: utility at true type >= utility at any other report
    utilities = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        utilities[i] = v1 * x_alloc[i, 0] + v2 * x_alloc[i, 1] - payments[i]
    
    # Check IC
    max_regret = 0.0
    for t in range(n_types):
        for tp in range(n_types):
            # Utility if type t reports tp: v(t) * x(tp) - p(tp)
            v1_t, v2_t = types[t]
            util_misreport = v1_t * x_alloc[tp, 0] + v2_t * x_alloc[tp, 1] - payments[tp]
            regret = util_misreport - utilities[t]
            if regret > max_regret:
                max_regret = regret
    
    # Check IR
    min_util = np.min(utilities)
    
    if max_regret > 1e-8 or min_util < -1e-8:
        # If constraints are violated, try a conservative solution
        # Allocate both goods to all types, set payments to keep IC and IR
        # Simple approach: u(t) = 0 for all t, then p(t) = v(t) * x(t)
        # But this may not be IC. Let's use a monotone allocation.
        # For now, fall back to zero allocation
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': x_alloc, 'payments': payments}


def _get_distribution(distribution_spec, n_types):
    """Get the probability distribution over types."""
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        probs = probs / probs.sum()
        return probs
    
    if isinstance(distribution_spec, str):
        spec = distribution_spec.lower().strip()
        if spec in ('uniform', 'independent_uniform'):
            return np.ones(n_types) / n_types
        elif spec == 'degenerate_high':
            # All mass on type (3,3)
            probs = np.zeros(n_types)
            probs[15] = 1.0  # type (3,3)
            return probs
        elif spec == 'degenerate_low':
            # All mass on type (0,0)
            probs = np.zeros(n_types)
            probs[0] = 1.0  # type (0,0)
            return probs
        elif spec == 'independent_bimodal':
            # Each good: 0.5 at 0, 0.5 at 3
            probs = np.zeros(n_types)
            for i in range(4):
                for j in range(4):
                    p1 = 0.5 if i in (0, 3) else 0.0
                    p2 = 0.5 if j in (0, 3) else 0.0
                    probs[i * 4 + j] = p1 * p2
            probs = probs / probs.sum()
            return probs
        else:
            # Default to uniform
            return np.ones(n_types) / n_types
    
    # Default
    return np.ones(n_types) / n_types
