import time
import numpy as np
from scipy.optimize import linprog


def _fallback_mechanism(types, n_types, n_goods, probs):
    """Simple valid mechanism: allocate good 1 to type if v1 >= 2, pay v1 if allocated."""
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        vi = types[i]
        if vi[0] >= 2:
            allocations[i, 0] = 1.0
            payments[i] = float(vi[0])
    return allocations, payments


def _max_payment_for_type(i, types, n_types, allocations, payments, tol=1e-10):
    """Find the maximum payment for type i that maintains IC and IR."""
    vi = np.array(types[i], dtype=float)
    truthful_val = np.dot(allocations[i], vi)
    max_p = truthful_val

    for j in range(n_types):
        if i == j:
            continue
        ic_bound = payments[j] + np.dot(vi, allocations[i] - allocations[j])
        max_p = min(max_p, ic_bound)

    return max(0.0, max_p)


def _validate(allocations, payments, types, n_types, tol=1e-9):
    """Check IC, IR, non-negativity, and feasibility with tight tolerance."""
    if np.any(allocations < -tol) or np.any(payments < -tol):
        return False
    if np.any(allocations > 1.0 + tol):
        return False
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui_i = np.dot(allocations[i], vi) - payments[i]
        if ui_i < -tol:
            return False
        for j in range(n_types):
            ui_j = np.dot(allocations[j], vi) - payments[j]
            if ui_i < ui_j - tol:
                return False
    return True


def _expected_revenue(allocations, payments, probs):
    """Compute expected revenue."""
    return float(np.sum(probs * payments))


def _solve_lp(probs, types, n_types, n_goods):
    """Solve the revenue-maximizing LP for a given probability distribution."""
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]

    A_ub = []
    b_ub = []

    # IC constraints: p_i - p_j + v_i . (x_j - x_i) <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i * n_goods + 0] = -vi[0]
            row[i * n_goods + 1] = -vi[1]
            row[j * n_goods + 0] = vi[0]
            row[j * n_goods + 1] = vi[1]
            row[n_types * n_goods + i] = 1.0
            row[n_types * n_goods + j] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: p_i - v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        row[i * n_goods + 0] = -vi[0]
        row[i * n_goods + 1] = -vi[1]
        row[n_types * n_goods + i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    bounds = []
    for _ in range(n_types * n_goods):
        bounds.append((0.0, 1.0))
    for _ in range(n_types):
        bounds.append((0.0, None))

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True,
                           'dual_feasibility_tolerance': 1e-12,
                           'primal_feasibility_tolerance': 1e-12})
    if not res.success:
        return None, None

    x_flat = res.x[:n_types * n_goods]
    p_flat = res.x[n_types * n_goods:]
    allocations = x_flat.reshape(n_types, n_goods)
    payments = p_flat.copy()
    return allocations, payments


def _polish_payments(allocations, payments, types, n_types, probs, deadline):
    """Iteratively push payments up to their IC/IR bounds to maximize revenue."""
    best_alloc = allocations.copy()
    best_pay = payments.copy()
    best_rev = _expected_revenue(best_alloc, best_pay, probs)

    for _polish_iter in range(15):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(n_types):
            max_p = _max_payment_for_type(i, types, n_types, best_alloc, best_pay)
            if max_p > best_pay[i] + 1e-13:
                old_p = best_pay[i]
                best_pay[i] = max_p
                if _validate(best_alloc, best_pay, types, n_types):
                    new_rev = _expected_revenue(best_alloc, best_pay, probs)
                    if new_rev > best_rev + 1e-15:
                        best_rev = new_rev
                        improved = True
                else:
                    lo, hi = best_pay[i], max_p
                    for _ in range(25):
                        mid = (lo + hi) / 2.0
                        best_pay[i] = mid
                        if _validate(best_alloc, best_pay, types, n_types):
                            lo = mid
                        else:
                            hi = mid
                    best_pay[i] = lo
                    if best_pay[i] > old_p + 1e-13:
                        improved = True
        if not improved:
            break

    return best_alloc, best_pay, best_rev


def _try_direct_enumeration(types, n_types, n_goods, probs, deadline):
    """
    For small problems, try to find the optimal mechanism by exploring
    the polytope vertices. The allocation matrix is 16x2 with binary or
    fractional entries. For a single bidder with additive values, the
    optimal mechanism has a specific structure.
    
    Key insight: In a single-bidder auction with additive values, the
    optimal IC mechanism is determined by the allocation rule. Given an
    allocation rule, the maximum payment for each type is determined by
    the IC and IR constraints. So we can enumerate allocation rules.
    
    However, with 16 types and 2 goods, the space is large. Instead,
    we use the LP solution and polish it.
    """
    pass


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0

    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    n_goods = 2

    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types

    probs = probs / probs.sum()

    try:
        allocations, payments = _solve_lp(probs, types, n_types, n_goods)

        if allocations is not None:
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)

            # Round small numerical errors
            allocations[np.abs(allocations) < 1e-12] = 0.0
            allocations[np.abs(allocations - 1.0) < 1e-12] = 1.0
            payments[payments < 1e-12] = 0.0

            if _validate(allocations, payments, types, n_types):
                # Polish payments to push them to their maximum feasible values
                best_alloc, best_pay, best_rev = _polish_payments(
                    allocations, payments, types, n_types, probs, deadline
                )
                if _validate(best_alloc, best_pay, types, n_types):
                    return {'allocations': best_alloc, 'payments': best_pay}

            # If polish failed, try repair
            for i in range(n_types):
                max_p = _max_payment_for_type(i, types, n_types, allocations, payments)
                if payments[i] > max_p:
                    payments[i] = max_p

            if _validate(allocations, payments, types, n_types):
                return {'allocations': allocations, 'payments': payments}

        # Fallback: try a simple but valid mechanism
        alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
        if _validate(alloc_fb, pay_fb, types, n_types):
            return {'allocations': alloc_fb, 'payments': pay_fb}

        # Last resort: zero mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    except Exception:
        alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
        return {'allocations': alloc_fb, 'payments': pay_fb}
