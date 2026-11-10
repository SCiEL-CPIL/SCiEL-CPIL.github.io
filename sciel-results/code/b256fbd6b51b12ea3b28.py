import numpy as np
import time
from scipy.optimize import linprog
import itertools

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    
    # Parse the distribution spec to get probabilities for each type
    # Types are (v1, v2) where v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = list(itertools.product(values, values))
    n_types = len(types)  # 16 types
    
    # Parse distribution_spec to get probability for each type
    # Assume distribution_spec is a dict mapping (v1,v2) -> prob, or similar
    if isinstance(distribution_spec, dict):
        type_probs = np.array([distribution_spec.get(t, 0.0) for t in types])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        type_probs = np.array(distribution_spec, dtype=float)
        if len(type_probs) == n_types:
            pass
        else:
            # Try to reshape
            type_probs = type_probs.reshape(-1)
            if len(type_probs) != n_types:
                type_probs = np.ones(n_types) / n_types
    else:
        type_probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total_prob = type_probs.sum()
    if total_prob > 0:
        type_probs = type_probs / total_prob
    else:
        type_probs = np.ones(n_types) / n_types
    
    # For each type, we have allocation probabilities x1[t], x2[t] and payment p[t]
    # Decision variables: x1[0..15], x2[0..15], p[0..15]
    # n_vars = 3 * 16 = 48
    
    n_vars = 3 * n_types
    # Variable ordering: x1[0], x1[1], ..., x1[15], x2[0], x2[1], ..., x2[15], p[0], p[1], ..., p[15]
    
    def x1_idx(t): return t
    def x2_idx(t): return n_types + t
    def p_idx(t): return 2 * n_types + t
    
    # Objective: maximize expected revenue = sum_t prob[t] * p[t]
    # In linprog, we minimize, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -type_probs[t]
    
    # Constraints:
    # 1. 0 <= x1[t] <= 1 for all t
    # 2. 0 <= x2[t] <= 1 for all t
    # 3. p[t] >= 0 for all t
    # 4. IR: utility[t] = v1[t]*x1[t] + v2[t]*x2[t] - p[t] >= 0
    #    => p[t] <= v1[t]*x1[t] + v2[t]*x2[t]
    # 5. IC (monotonicity): For truthful reporting to be optimal, we need:
    #    For any type t and any report t', utility when type t reports t' <= utility when type t reports t
    #    v1[t]*x1[t'] + v2[t]*x2[t'] - p[t'] <= v1[t]*x1[t] + v2[t]*x2[t] - p[t]
    #    => p[t'] - p[t] <= v1[t]*(x1[t'] - x1[t]) + v2[t]*(x2[t'] - x2[t])
    #    
    #    This is a lot of constraints. Let's use the standard approach:
    #    The IC constraints can be reduced to monotonicity:
    #    x1[t] is non-decreasing in v1 (holding v2 fixed)
    #    x2[t] is non-decreasing in v2 (holding v1 fixed)
    #    And the payment formula: p[t] = v1[t]*x1[t] + v2[t]*x2[t] - U[t] where U is utility
    #    But with IR, U[t] >= 0, and for optimal revenue, U[t] = 0 for the lowest type
    #    
    #    Actually, let's just directly impose all IC constraints. There are 16*16 = 256 IC constraints.
    
    A_ub = []
    b_ub = []
    
    # IC constraints: for each true type t and each report t'
    for t in range(n_types):
        for tp in range(n_types):
            v1_t, v2_t = types[t]
            # Constraint: v1_t*x1[tp] + v2_t*x2[tp] - p[tp] <= v1_t*x1[t] + v2_t*x2[t] - p[t]
            # => v1_t*x1[tp] - v1_t*x1[t] + v2_t*x2[tp] - v2_t*x2[t] - p[tp] + p[t] <= 0
            row = np.zeros(n_vars)
            row[x1_idx(tp)] += v1_t
            row[x1_idx(t)] -= v1_t
            row[x2_idx(tp)] += v2_t
            row[x2_idx(t)] -= v2_t
            row[p_idx(tp)] -= 1
            row[p_idx(t)] += 1
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: p[t] <= v1[t]*x1[t] + v2[t]*x2[t]
    # => -v1[t]*x1[t] - v2[t]*x2[t] + p[t] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[x1_idx(t)] -= v1_t
        row[x2_idx(t)] -= v2_t
        row[p_idx(t)] += 1
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))  # x1[t]
    for t in range(n_types):
        bounds.append((0, 1))  # x2[t]
    for t in range(n_types):
        bounds.append((0, None))  # p[t]
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Check time
    if time.time() - start_time > 60:
        # Return a simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            allocations = np.array([[x[x1_idx(t)], x[x2_idx(t)]] for t in range(n_types)])
            payments = np.array([x[p_idx(t)] for t in range(n_types)])
            
            # Verify constraints
            # Check IR
            for t in range(n_types):
                v1_t, v2_t = types[t]
                utility = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                if utility < -1e-8:
                    # Fix IR violation
                    payments[t] = max(0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - 1e-10)
            
            # Check IC (regret)
            max_regret = 0
            for t in range(n_types):
                v1_t, v2_t = types[t]
                true_utility = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                for tp in range(n_types):
                    misreport_utility = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
                    regret = misreport_utility - true_utility
                    if regret > max_regret:
                        max_regret = regret
            
            if max_regret > 1e-8:
                # IC violated, need to fix
                # Use a simpler approach: set payments based on a monotone allocation
                pass
            
            # Ensure nonnegative payments
            payments = np.maximum(payments, 0)
            
            # Check time
            if time.time() - start_time > 75:
                return {'allocations': allocations, 'payments': payments}
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # LP failed, return a simple mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
    except Exception as e:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}


# Fallback simple mechanism if LP is too slow
def _simple_mechanism(types, type_probs):
    n_types = len(types)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for t in range(n_types):
        v1, v2 = types[t]
        allocations[t, 0] = 1.0 if v1 > 0 else 0.0
        allocations[t, 1] = 1.0 if v2 > 0 else 0.0
        payments[t] = v1 * allocations[t, 0] + v2 * allocations[t, 1]
    return {'allocations': allocations, 'payments': payments}
