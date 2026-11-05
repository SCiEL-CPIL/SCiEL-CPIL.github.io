"""Incentive-compatible finite auction design.
Combines a direct (ironing / envelope) LP over the expected-utility vector
with the robust verify-and-fallback machinery of the seed program."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def _parse_distribution(distribution_spec, num_types, types):
    """Parse distribution spec into a probability vector over the 16 types."""
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        probs = np.ones(num_types) / num_types
    else:
        try:
            arr = np.array(distribution_spec, dtype=float).flatten()
            if len(arr) == num_types:
                total = arr.sum()
                if total > 0:
                    probs = arr / total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except Exception:
            probs = np.ones(num_types) / num_types
    return probs

def _adjacent_pairs(num_types):
    """Adjacent pairs in the 4x4 value grid (right + up)."""
    adj = []
    for v1 in range(4):
        for v2 in range(4):
            i = v1 * 4 + v2
            if v2 < 3:
                adj.append((i, v1 * 4 + (v2 + 1)))
            if v1 < 3:
                adj.append((i, (v1 + 1) * 4 + v2))
    return adj

def _solve_lp(probs, types, num_types, A_ub, b_ub, bounds):
    """Solve the revenue-maximizing LP with several HiGHS configurations."""
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]

    configs = [
        {'method': 'highs', 'options': {'presolve': True,
                                        'primal_feasibility_tolerance': 1e-12,
                                        'dual_feasibility_tolerance': 1e-12}},
        {'method': 'highs', 'options': {'presolve': False,
                                        'primal_feasibility_tolerance': 1e-12,
                                        'dual_feasibility_tolerance': 1e-12}},
        {'method': 'highs', 'options': {'presolve': True}},
        {'method': 'highs'},
    ]
    best_x, best_obj = None, -np.inf
    for cfg in configs:
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, **cfg)
            if res.success:
                obj = -res.fun
                if obj > best_obj:
                    best_obj, best_x = obj, res.x
        except Exception:
            pass
    return best_x

def _build_lp_matrices(types, num_types):
    """Full (x1,x2,t) LP: all-pairs IC + IR, as in the seed program."""
    n_vars = 3 * num_types
    n_constraints = num_types * num_types + num_types
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    row = 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            r = A_ub[row]
            r[i] = -v1_i
            r[num_types + i] = -v2_i
            r[2 * num_types + i] = 1.0
            if i != j:
                r[j] += v1_i
                r[num_types + j] += v2_i
                r[2 * num_types + j] -= 1.0
            row += 1
    for i in range(num_types):
        v1_i, v2_i = types[i]
        r = A_ub[row]
        r[i] = -v1_i
        r[num_types + i] = -v2_i
        r[2 * num_types + i] = 1.0
        row += 1
    return A_ub, b_ub

def _ironing_allocations(u, types, num_types, adj):
    """Recover allocations x_r = ∇u(v_r) from the expected-utility vector u.
    For a product partial order the gradient components are differences along
    the canonical coordinate directions (value step = 1)."""
    x1 = np.zeros(num_types)
    x2 = np.zeros(num_types)
    for (i, j) in adj:
        v1_i, v2_i = types[i]
        v1_j, v2_j = types[j]
        if v2_j == v2_i + 1 and v1_j == v1_i:      # i -> right (v2 + 1)
            x2[i] = u[j] - u[i]
        elif v1_j == v1_i + 1 and v2_j == v2_i:    # i -> up (v1 + 1)
            x1[i] = u[j] - u[i]
    # top / right edges have no forward neighbour -> allocation 0
    x1 = np.clip(x1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    return x1, x2

def _ironing_lp(probs, types, num_types, adj):
    """Direct (ironing / envelope) LP over the expected-utility vector u.
    IC  <=> u is nondecreasing along the product partial order (adjacent).
    IR  <=> u >= 0.  Maximize revenue = sum p_i u_i.  Returns (x1,x2,t)."""
    n = num_types
    # objective: maximize sum p_i u_i  ->  minimize -sum p_i u_i
    c = -probs.copy()
    rows, rhs = [], []
    for (i, j) in adj:                 # u_i - u_j <= 0  (monotonicity)
        r = np.zeros(n)
        r[i] = 1.0
        r[j] = -1.0
        rows.append(r)
        rhs.append(0.0)
    for i in range(n):                 # -u_i <= 0  (IR)
        r = np.zeros(n)
        r[i] = -1.0
        rows.append(r)
        rhs.append(0.0)
    A_ub = np.array(rows)
    b_ub = np.array(rhs)
    bounds = [(0.0, None)] * n
    best_x, best_obj = None, -np.inf
    for cfg in [{'method': 'highs', 'options': {'presolve': True,
                                                 'primal_feasibility_tolerance': 1e-12,
                                                 'dual_feasibility_tolerance': 1e-12}},
                {'method': 'highs', 'options': {'presolve': False,
                                                 'primal_feasibility_tolerance': 1e-12}},
                {'method': 'highs'}]:
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, **cfg)
            if res.success and -res.fun > best_obj:
                best_obj, best_x = -res.fun, res.x
        except Exception:
            pass
    if best_x is None:
        return None
    u = np.clip(best_x, 0.0, None)
    x1, x2 = _ironing_allocations(u, types, num_types, adj)
    t = np.array([types[i][0] * x1[i] + types[i][1] * x2[i] - u[i]
                  for i in range(n)])
    t = np.clip(t, 0.0, None)
    return x1, x2, t

def _verify_mechanism(allocations, payments, types, num_types, tol=1e-8):
    """Verify IC and IR constraints."""
    x1 = allocations[:, 0]
    x2 = allocations[:, 1]
    t = payments
    for i in range(num_types):
        v1_i, v2_i = types[i]
        u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
        if u_ii < -tol:
            return False
        for j in range(num_types):
            u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
            if u_ii < u_ij - tol:
                return False
    return True

def _fix_violations(allocations, payments, types, num_types):
    """Given a fixed allocation, recompute the maximal IC/IR payments."""
    x1 = np.clip(allocations[:, 0], 0, 1).copy()
    x2 = np.clip(allocations[:, 1], 0, 1).copy()
    t = np.clip(payments, 0, None).copy()
    for _ in range(100):
        changed = False
        for i in range(num_types):
            v1_i, v2_i = types[i]
            ir_bound = v1_i * x1[i] + v2_i * x2[i]
            ic_bound = np.inf
            for j in range(num_types):
                bound = (v1_i * x1[i] + v2_i * x2[i]
                         - v1_i * x1[j] - v2_i * x2[j] + t[j])
                if bound < ic_bound:
                    ic_bound = bound
            max_t = max(min(ir_bound, ic_bound), 0.0)
            if t[i] > max_t:
                t[i] = max_t
                changed = True
            elif t[i] < 0:
                t[i] = 0.0
                changed = True
        if not changed:
            break
    return np.column_stack([x1, x2]), t

def _payment_lp(x1, x2, types, num_types, probs, tol=1e-12):
    """Secondary LP: maximize revenue over t with fixed allocation (x1,x2)."""
    n = num_types
    c2 = -probs.copy()
    A_ub2 = np.zeros((n * n + n, n))
    b_ub2 = np.zeros(n * n + n)
    row = 0
    for i in range(n):
        v1_i, v2_i = types[i]
        u_ii = v1_i * x1[i] + v2_i * x2[i]
        for j in range(n):
            u_ij = v1_i * x1[j] + v2_i * x2[j]
            A_ub2[row, i] = 1.0
            if i != j:
                A_ub2[row, j] = -1.0
            b_ub2[row] = u_ii - u_ij
            row += 1
        A_ub2[row, i] = 1.0
        b_ub2[row] = u_ii
        row += 1
    bounds2 = [(0.0, None)] * n
    try:
        res = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs',
                      options={'presolve': True, 'primal_feasibility_tolerance': tol})
        if res.success:
            return np.clip(res.x, 0.0, None)
    except Exception:
        pass
    return None

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before the 90 s kill

    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)

    probs = _parse_distribution(distribution_spec, num_types, types)
    adj = _adjacent_pairs(num_types)

    best_result = None
    best_revenue = -np.inf

    def consider(alloc, pay):
        nonlocal best_result, best_revenue
        if _verify_mechanism(alloc, pay, types, num_types):
            rev = float(np.sum(probs * pay))
            if rev > best_revenue:
                best_revenue = rev
                best_result = {'allocations': alloc, 'payments': pay}
            return True
        return False

    # ---- PRIMARY: direct ironing / envelope LP over u ----
    if time.time() < deadline:
        try:
            out = _ironing_lp(probs, types, num_types, adj)
            if out is not None:
                x1, x2, t = out
                alloc = np.column_stack([x1, x2])
                if consider(alloc, t):
                    return best_result
        except Exception:
            pass

    # ---- SECONDARY: full (x1,x2,t) LP ----
    if time.time() < deadline:
        A_ub, b_ub = _build_lp_matrices(types, num_types)
        bounds = []
        for _ in range(num_types):
            bounds.append((0, 1))
        for _ in range(num_types):
            bounds.append((0, 1))
        for _ in range(num_types):
            bounds.append((0, None))
        x_sol = _solve_lp(probs, types, num_types, A_ub, b_ub, bounds)
        if x_sol is not None:
            x1 = np.clip(x_sol[:num_types], 0, 1)
            x2 = np.clip(x_sol[num_types:2 * num_types], 0, 1)
            t = np.clip(x_sol[2 * num_types:3 * num_types], 0, None)
            alloc = np.column_stack([x1, x2])
            if consider(alloc, t):
                return best_result
            # polish payments with a dedicated LP, then fix + verify
            t2 = _payment_lp(x1, x2, types, num_types, probs)
            if t2 is not None and consider(alloc, t2):
                return best_result
            alloc_f, pay_f = _fix_violations(alloc, t)
            if consider(alloc_f, pay_f):
                return best_result

    # ---- FALLBACK: heuristic mechanisms ----
    if time.time() < deadline:
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x1 = np.full(num_types, alpha)
            x2 = np.full(num_types, alpha)
            t = np.array([types[i][0] * alpha + types[i][1] * alpha
                          for i in range(num_types)])
            alloc = np.column_stack([x1, x2])
            if consider(alloc, t):
                return best_result
        for thresh in range(0, 8):
            x1 = np.array([1.0 if types[i][0] >= thresh else 0.0
                           for i in range(num_types)])
            x2 = np.array([1.0 if types[i][1] >= thresh else 0.0
                           for i in range(num_types)])
            t = np.array([types[i][0] * x1[i] + types[i][1] * x2[i]
                          for i in range(num_types)])
            alloc = np.column_stack([x1, x2])
            if consider(alloc, t):
                return best_result

    if best_result is not None:
        return best_result

    # Final fallback: zero mechanism
    return {'allocations': np.zeros((num_types, 2)),
            'payments': np.zeros(num_types)}
# EVOLVE-BLOCK-END
