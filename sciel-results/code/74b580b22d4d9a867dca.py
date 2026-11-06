"""Incentive-compatible finite auction design: MODEL-AND-OPTIMIZE via LP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: the support of values (e.g., [0,1,2,3])
        constraints: constraint dict
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    # Deadline guard: kill before 90s, return best found with safety margin
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin
    
    # Parse inputs
    if support is None:
        support = [0, 1, 2, 3]
    if not isinstance(support, (list, tuple)):
        support = [0, 1, 2, 3]
    support = list(support)
    n_vals = len(support)
    
    # Enumerate all type profiles
    types = list(product(range(n_vals), repeat=2))
    n_types = len(types)
    
    # Determine probabilities from distribution_spec
    probs = _parse_distribution(distribution_spec, n_vals, n_types)
    
    # Values matrix
    V = np.array([[support[types[i][0]], support[types[i][1]]] for i in range(n_types)], dtype=np.float64)
    
    # Budget check
    if time.time() > deadline - 5.0:
        return _baseline_result(V, probs)
    
    from scipy.optimize import linprog
    
    # Variables: x = [a[0,0], a[0,1], ..., a[n-1,0], a[n-1,1], p[0], ..., p[n-1]]
    def idx_a(t, j):
        return 2 * t + j
    
    def idx_p(t):
        return 2 * n_types + t
    
    n_vars = 3 * n_types
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so minimize -sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[idx_p(t)] = -probs[t]
    
    # Build constraint matrix
    # Truthfulness: for all (t, t'), V[t]·a[t] - p[t] >= V[t]·a[t'] - p[t']
    #   => -V[t]·a[t] + V[t]·a[t'] + p[t] - p[t'] <= 0
    # IR: for all t, V[t]·a[t] - p[t] >= 0
    #   => -V[t]·a[t] + p[t] <= 0
    
    n_ineq_truthfulness = n_types * n_types
    n_ineq_ir = n_types
    n_ineq = n_ineq_truthfulness + n_ineq_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    for t in range(n_types):
        for tp in range(n_types):
            # Truthfulness constraint
            for j in range(2):
                A_ub[row, idx_a(t, j)] = -V[t, j]
                A_ub[row, idx_a(tp, j)] = V[t, j]
            A_ub[row, idx_p(t)] = 1.0
            A_ub[row, idx_p(tp)] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    for t in range(n_types):
        # IR constraint
        for j in range(2):
            A_ub[row, idx_a(t, j)] = -V[t, j]
        A_ub[row, idx_p(t)] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # Bounds: a[t,j] in [0,1], p[t] >= 0
    bounds = []
    for t in range(n_types):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        bounds.append((0.0, None))
    
    # Check time
    if time.time() > deadline - 3.0:
        return _baseline_result(V, probs)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return _baseline_result(V, probs)
    
    if result is not None and result.success:
        x = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            for j in range(2):
                allocations[t, j] = x[idx_a(t, j)]
            payments[t] = x[idx_p(t)]
        
        # Clip to ensure feasibility due to numerical precision
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify constraints
        if _verify_mechanism(V, probs, allocations, payments, tol=1e-8):
            return {
                'allocations': allocations,
                'payments': payments
            }
    
    # Fallback to baseline
    return _baseline_result(V, probs)


def _parse_distribution(distribution_spec, n_vals, n_types):
    """Parse distribution spec into probability vector."""
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=np.float64).flatten()
                if probs.shape[0] == n_types:
                    probs = np.maximum(probs, 0.0)
                    total = probs.sum()
                    if total > 1e-12:
                        return probs / total
            elif 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=np.float64).flatten()
                if probs.shape[0] == n_types:
                    probs = np.maximum(probs, 0.0)
                    total = probs.sum()
                    if total > 1e-12:
                        return probs / total
            elif 'pmf' in distribution_spec:
                probs = np.array(distribution_spec['pmf'], dtype=np.float64).flatten()
                if probs.shape[0] == n_types:
                    probs = np.maximum(probs, 0.0)
                    total = probs.sum()
                    if total > 1e-12:
                        return probs / total
            elif 'type' in distribution_spec:
                dtype = distribution_spec['type']
                if dtype == 'uniform' or dtype == 'product_uniform':
                    return np.ones(n_types) / n_types
                if 'params' in distribution_spec:
                    params = distribution_spec['params']
                    if isinstance(params, dict) and 'p1' in params and 'p2' in params:
                        p1 = np.array(params['p1'], dtype=np.float64)
                        p2 = np.array(params['p2'], dtype=np.float64)
                        if len(p1) == n_vals and len(p2) == n_vals:
                            probs = np.zeros(n_types)
                            for i in range(n_vals):
                                for j in range(n_vals):
                                    probs[i * n_vals + j] = p1[i] * p2[j]
                            probs = np.maximum(probs, 0.0)
                            total = probs.sum()
                            if total > 1e-12:
                                return probs / total
    return np.ones(n_types) / n_types


def _baseline_result(V, probs):
    """Return a valid baseline mechanism (always allocate nothing, charge 0)."""
    n_types = len(V)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }


def _verify_mechanism(V, probs, allocations, payments, tol=1e-8):
    """Verify that the mechanism satisfies all constraints."""
    n_types = len(V)
    
    # Check feasibility: 0 <= a[t,j] <= 1
    if np.any(allocations < -tol) or np.any(allocations > 1.0 + tol):
        return False
    
    # Check nonnegative payments
    if np.any(payments < -tol):
        return False
    
    # Check truthfulness
    for t in range(n_types):
        for tp in range(n_types):
            u_t_t = np.dot(V[t], allocations[t]) - payments[t]
            u_t_tp = np.dot(V[t], allocations[tp]) - payments[tp]
            if u_t_t < u_t_tp - tol:
                return False
    
    # Check individual rationality
    for t in range(n_types):
        u_t_t = np.dot(V[t], allocations[t]) - payments[t]
        if u_t_t < -tol:
            return False
    
    # Check probability feasibility
    if probs.sum() < 1.0 - tol or probs.sum() > 1.0 + tol:
        return False
    if np.any(probs < -tol):
        return False
    
    return True
# EVOLVE-BLOCK-END
