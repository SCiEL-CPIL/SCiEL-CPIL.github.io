"""Incentive-compatible finite auction design - LP-based solution (Utility Formulation)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the type distribution
        support: Support of types (values in {0,1,2,3} for 2 goods)
        constraints: Additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = list(product([0, 1, 2, 3], repeat=2))
    n_types = len(types)  # 16 types
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, types)
    
    # Try to solve the LP
    result = _solve_auction_lp_utility(types, probs, n_goods, deadline)
    
    if result is not None:
        return result
    
    # Fallback: baseline mechanism (no allocation, no payment)
    return _baseline_result(n_types)


def _parse_distribution(distribution_spec, types):
    """Parse the distribution specification into probabilities for each type."""
    n_types = len(types)
    
    # Handle various distribution specifications
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    # Check if it's already a probability vector
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        spec = np.array(distribution_spec, dtype=float)
        if spec.shape == (n_types,) and np.isclose(spec.sum(), 1.0, atol=1e-6):
            return spec / spec.sum()
    
    # Check for 'uniform' keyword
    if isinstance(distribution_spec, str):
        spec_lower = distribution_spec.lower()
        if 'uniform' in spec_lower:
            return np.ones(n_types) / n_types
    
    # Try to extract probabilities from dict format
    if isinstance(distribution_spec, dict):
        # Could be {type: prob} format
        probs = np.zeros(n_types)
        type_map = {t: i for i, t in enumerate(types)}
        for t, p in distribution_spec.items():
            if t in type_map:
                probs[type_map[t]] = p
        if probs.sum() > 0:
            return probs / probs.sum()
        
        # Could have 'probs' or 'probabilities' key
        for key in ['probs', 'probabilities', 'weights']:
            if key in distribution_spec:
                spec = np.array(distribution_spec[key], dtype=float)
                if spec.shape == (n_types,):
                    return spec / spec.sum()
    
    # Default to uniform
    return np.ones(n_types) / n_types


def _solve_auction_lp_utility(types, probs, n_goods, deadline):
    """
    Solve the LP to maximize expected revenue subject to IC and IR constraints.
    Uses utility variables u_t instead of payments p_t for potentially better sparsity.
    
    Variables:
        u[t] = ex-post utility for type t (n_types vars)
        x[t, g] = probability of allocating good g to type t (n_types * n_goods vars)
    
    Objective: Maximize sum_t probs[t] * (v_t @ x[t] - u[t])
    
    Constraints:
        - Feasibility: 0 <= x[t, g] <= 1, u[t] >= 0
        - IC: u[t] >= u[t'] + v_t @ x[t'] - v_t' @ x[t']
        - IR: u[t] >= 0
    """
    from scipy.optimize import linprog
    
    n_types = len(types)
    
    # Variable indexing:
    # 0 .. n_types-1 : u[t]
    # n_types .. n_types + n_types*n_goods - 1 : x[t, g]
    
    def var_util(t):
        return t

    def var_alloc(t, g):
        return n_types + t * n_goods + g

    n_vars = n_types + n_types * n_goods
    
    # Objective: Maximize sum_t probs[t] * (v_t @ x[t] - u[t])
    # Minimize -Objective
    c = np.zeros(n_vars)
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        # Coeff for u[t] is -probs[t]
        c[var_util(t)] = -probs[t]
        # Coeff for x[t, g] is probs[t] * v_t[g]
        for g in range(n_goods):
            c[var_alloc(t, g)] = probs[t] * v_t[g]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: u[t] >= u[t'] + v_t @ x[t'] - v_t' @ x[t']
    # Rearranged: u[t'] + (v_t - v_t') @ x[t'] - u[t] <= 0
    
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            v_tp = np.array(types[t_prime], dtype=float)
            
            row = np.zeros(n_vars)
            
            # -u[t]
            row[var_util(t)] = -1.0
            
            # +u[t']
            row[var_util(t_prime)] = 1.0
            
            # +(v_t - v_t') @ x[t']
            for g in range(n_goods):
                row[var_alloc(t_prime, g)] = v_t[g] - v_tp[g]
            
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # Convert to arrays
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds for variables
    bounds = []
    # u[t] >= 0
    for t in range(n_types):
        bounds.append((0.0, None))
    # 0 <= x[t, g] <= 1
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    
    # Check time before solving
    if time.time() > deadline - 5.0:
        return None
    
    # Solve LP
    try:
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='HiGHS',
            options={'time_limit': 75.0, 'presolve': True}
        )
    except Exception:
        return None
    
    if time.time() > deadline:
        return None
    
    if res.status != 0:
        return None
    
    # Extract solution
    x = res.x
    
    utilities = np.zeros(n_types)
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        utilities[t] = x[var_util(t)]
        for g in range(n_goods):
            allocations[t, g] = x[var_alloc(t, g)]
        # Payment is v_t @ x[t] - u[t]
        v_t = np.array(types[t], dtype=float)
        payments[t] = v_t @ allocations[t] - utilities[t]
    
    # Clip to valid range
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    # Verify the mechanism
    if _verify_mechanism(types, probs, allocations, payments):
        return {
            'allocations': allocations,
            'payments': payments
        }
    
    return None


def _verify_mechanism(types, probs, allocations, payments):
    """Verify that the mechanism satisfies all constraints."""
    n_types = len(types)
    eps = 1e-8
    
    # Check feasibility
    if np.any(allocations < -eps) or np.any(allocations > 1 + eps):
        return False
    if np.any(payments < -eps):
        return False
    
    # Check IC: For each type t, truth-telling utility >= misreporting utility
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_truth = v_t @ allocations[t] - payments[t]
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            util_misreport = v_t @ allocations[t_prime] - payments[t_prime]
            if util_misreport > util_truth + eps:
                return False
    
    # Check IR: Utility >= 0 for all types
    for t in range(n_types):
        v_t = np.array(types[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        if util_t < -eps:
            return False
    
    return True


def _baseline_result(n_types):
    """Return a simple baseline mechanism that is always feasible."""
    # Allocate nothing, charge nothing - always IR and IC
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
