# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def _parse_probs(distribution_spec, n_types):
    """Parse distribution probabilities from various spec formats."""
    if isinstance(distribution_spec, dict):
        for key in ('probs', 'probabilities', 'weights', 'distribution'):
            if key in distribution_spec:
                probs = np.array(distribution_spec[key], dtype=float)
                if probs.size == n_types:
                    return _normalize(probs, n_types)
        for key in distribution_spec:
            val = distribution_spec[key]
            if isinstance(val, (list, tuple, np.ndarray)):
                probs = np.array(val, dtype=float)
                if probs.size == n_types:
                    return _normalize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.size == n_types:
            return _normalize(probs, n_types)
    return np.ones(n_types) / n_types


def _normalize(probs, n_types):
    probs = probs.reshape(n_types)
    s = probs.sum()
    if s > 0:
        return probs / s
    return np.ones(n_types) / n_types


def _solve_lp(probs, V, n_types):
    """Solve the direct LP for maximum expected revenue."""
    # Variables: x[g,t] for g=0,1; t=0..15  (32 vars)
    #             p[t] for t=0..15           (16 vars)
    # Total: 48 vars

    def x_idx(g, t):
        return g * n_types + t

    def p_idx(t):
        return 2 * n_types + t

    n_vars = 3 * n_types

    # Objective: maximize sum_t probs[t]*p[t]  => minimize -sum_t probs[t]*p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]

    A_rows = []
    b_vals = []

    # IC: u(t,t) >= u(t, t')  =>  sum_g V[t,g]*x[g,t] - p[t] >= sum_g V[t,g]*x[g,t'] - p[t']
    # => sum_g V[t,g]*x[g,t'] - sum_g V[t,g]*x[g,t] - p[t'] + p[t] <= 0
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[x_idx(g, tp)] += V[t, g]
                row[x_idx(g, t)] -= V[t, g]
            row[p_idx(tp)] -= 1.0
            row[p_idx(t)] += 1.0
            A_rows.append(row)
            b_vals.append(0.0)

    # IR: u(t,t) >= 0  =>  -sum_g V[t,g]*x[g,t] + p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(2):
            row[x_idx(g, t)] -= V[t, g]
        row[p_idx(t)] += 1.0
        A_rows.append(row)
        b_vals.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)

    bounds = [(0.0, 1.0)] * (2 * n_types) + [(0.0, None)] * n_types

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})
        if not result.success:
            return None
        x_opt = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            for g in range(2):
                allocations[t, g] = x_opt[x_idx(g, t)]
            payments[t] = x_opt[p_idx(t)]
        return allocations, payments
    except Exception:
        return None


def _verify(allocations, payments, V, probs, n_types, tol=1e-8):
    """Check IR and IC constraints. Return (max_violation, expected_revenue)."""
    max_viol = 0.0
    for t in range(n_types):
        u_tt = float(np.dot(V[t], allocations[t])) - payments[t]
        if u_tt < -tol:
            max_viol = max(max_viol, -u_tt)
        for tp in range(n_types):
            if t == tp:
                continue
            u_t_tp = float(np.dot(V[t], allocations[tp])) - payments[tp]
            viol = u_t_tp - u_tt
            if viol > tol:
                max_viol = max(max_viol, viol)
    revenue = float(np.dot(probs, payments))
    return max_viol, revenue


def _dual_solution(V, n_types):
    """
    Solve the dual LP to get upper bound on revenue.
    Dual: max sum_t mu_t * V[t,0] + sum_t nu_t * V[t,1]
          s.t. mu_t + nu_t >= 1  for all t
               mu_t >= 0, nu_t >= 0
    """
    n_vars_dual = 2 * n_types
    c = np.zeros(n_vars_dual)
    for t in range(n_types):
        c[t] = V[t, 0]
        c[n_types + t] = V[t, 1]

    # mu_t + nu_t >= 1  =>  -mu_t - nu_t <= -1
    A_ub = np.zeros((n_types, n_vars_dual))
    for t in range(n_types):
        A_ub[t, t] = -1.0
        A_ub[t, n_types + t] = -1.0
    b_ub = np.ones(n_types) * (-1.0)

    bounds = [(0.0, None)] * n_vars_dual

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})
        if not result.success:
            return None
        return float(-result.fun)  # linprog minimizes, we want max
    except Exception:
        return None


def _payoff_matrix(V, allocations, payments, n_types):
    """Return matrix M where M[t, tp] = sum_g V[t,g]*alloc[tp,g] - payments[tp]."""
    M = np.zeros((n_types, n_types))
    for t in range(n_types):
        for tp in range(n_types):
            M[t, tp] = float(np.dot(V[t], allocations[tp])) - payments[tp]
    return M


def _repair_mechanism(allocations, payments, V, probs, n_types, max_iters=200):
    """
    Repair a mechanism by iteratively fixing the worst IC violation.
    For a violation where u(t,tp) > u(t,t), increase payments[tp] by the gap.
    If that breaks IR for tp, decrease allocations[tp,g] for the good g
    that contributes most to tp's utility, which also reduces t's utility from tp.
    """
    allocations = allocations.copy()
    payments = payments.copy()
    tol = 1e-9

    for iteration in range(max_iters):
        M = _payoff_matrix(V, allocations, payments, n_types)
        u_tt = np.array([M[t, t] for t in range(n_types)])
        # Find worst violation: max over t, tp (t != tp) of M[t,tp] - M[t,t]
        best_viol = 0.0
        best_t = -1
        best_tp = -1
        for t in range(n_types):
            for tp in range(n_types):
                if t == tp:
                    continue
                viol = M[t, tp] - M[t, t]
                if viol > best_viol:
                    best_viol = viol
                    best_t = t
                    best_tp = tp

        if best_viol <= tol:
            # Check IR
            for t in range(n_types):
                if u_tt[t] < -tol:
                    # Fix IR by reducing payment
                    payments[t] = max(0.0, u_tt[t] + tol)
            # Re-verify
            M2 = _payoff_matrix(V, allocations, payments, n_types)
            max_v = 0.0
            for t in range(n_types):
                for tp in range(n_types):
                    if t != tp:
                        v = M2[t, tp] - M2[t, t]
                        if v > max_v:
                            max_v = v
            if max_v <= tol and all(M2[t, t] >= -tol for t in range(n_types)):
                break
            if best_viol <= tol:
                # Only IR issues remain; adjust payments
                for t in range(n_types):
                    utt = float(np.dot(V[t], allocations[t])) - payments[t]
                    if utt < -tol:
                        payments[t] = max(0.0, utt + tol)
                break
            continue

        t, tp = best_t, best_tp
        gap = best_viol + 1e-12

        # Try increasing payments[tp]
        new_pay = payments[tp] + gap
        if new_pay >= 0:
            # Check IR for tp
            u_tp_tp = float(np.dot(V[tp], allocations[tp])) - new_pay
            if u_tp_tp >= -tol:
                payments[tp] = new_pay
                continue
            # Increasing payment breaks IR. Try reducing allocation instead.

        # Reduce allocation of good with highest V[tp, g]
        g_best = int(np.argmax(V[tp]))
        old_x = allocations[tp, g_best]
        allocations[tp, g_best] = max(0.0, old_x - gap)
        # If allocation can't be reduced enough, also reduce payment slightly
        if allocations[tp, g_best] == 0.0 and old_x < gap:
            # Need to reduce payment too, but that worsens IR. 
            # Instead, just clamp and continue
            pass

        # Also ensure payments non-negative
        payments = np.maximum(payments, 0.0)
        allocations = np.clip(allocations, 0.0, 1.0)

    return allocations, payments


def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = t0 + 82.0

    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)

    probs = _parse_probs(distribution_spec, n_types)
    V = np.array([[t[0], t[1]] for t in types], dtype=float)

    # Solve primal LP
    result = _solve_lp(probs, V, n_types)

    if result is not None:
        allocations, payments = result
        # Snap allocations
        snap_tol = 1e-5
        allocations = np.where(allocations < snap_tol, 0.0, allocations)
        allocations = np.where(allocations > 1.0 - snap_tol, 1.0, allocations)
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)

        # Verify
        max_viol, revenue = _verify(allocations, payments, V, probs, n_types)

        if max_viol > 1e-8:
            # Repair
            allocations, payments = _repair_mechanism(allocations, payments, V, probs, n_types)
            max_viol, revenue = _verify(allocations, payments, V, probs, n_types)

            if max_viol > 1e-8:
                # Fallback: return zero mechanism
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                return {'allocations': allocations, 'payments': payments}

        # Final safety check
        if time.time() > deadline - 2.0:
            return {'allocations': allocations, 'payments': payments}

        return {'allocations': allocations, 'payments': payments}
    else:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}


# EVOLVE-BLOCK-END
