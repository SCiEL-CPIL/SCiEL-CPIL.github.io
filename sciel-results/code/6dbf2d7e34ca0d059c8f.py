"""Incentive-compatible finite auction design solver using LP + SLSQP polish."""
import time
import numpy as np
from scipy.optimize import linprog, minimize


def _parse_distribution(distribution_spec):
    """Parse distribution_spec into a 4x4 joint probability matrix."""
    p_vals = None
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) == 4:
                p_vals = np.outer(probs, probs)
            elif len(probs) == 16:
                p_vals = probs.reshape(4, 4)
        elif isinstance(distribution_spec, dict):
            p_vals = np.zeros((4, 4))
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    p_vals[int(k[0]), int(k[1])] = float(v)
            if all(isinstance(k, int) for k in distribution_spec.keys()):
                marg = np.array([distribution_spec.get(i, 0) for i in range(4)])
                if marg.sum() > 0:
                    p_vals = np.outer(marg, marg)
    except Exception:
        pass

    if p_vals is None:
        p_vals = np.ones((4, 4)) / 16.0

    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.size == 4:
        p_vals = np.outer(p_vals, p_vals)
    p_vals = p_vals.reshape(4, 4)
    s = p_vals.sum()
    if s > 0:
        p_vals = p_vals / s
    else:
        p_vals = np.ones((4, 4)) / 16.0
    return p_vals


def _parse_support(support):
    """Parse support into a set of valid (i,j) type tuples."""
    all_types = [(i, j) for i in range(4) for j in range(4)]
    if support is None:
        return set(all_types)
    eff = []
    if isinstance(support, (list, tuple, set, np.ndarray)):
        for r in support:
            try:
                eff.append((int(r[0]), int(r[1])))
            except Exception:
                pass
    valid = set()
    for t in eff:
        if 0 <= t[0] <= 3 and 0 <= t[1] <= 3:
            valid.add(t)
    return valid if valid else set(all_types)


def _fallback():
    """Simple valid fallback: allocate good g if value >= 2, pay 2 per good."""
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if i >= 2:
                alloc[idx, 0] = 1.0
                pay[idx] += 2.0
            if j >= 2:
                alloc[idx, 1] = 1.0
                pay[idx] += 2.0
    return {"allocations": alloc, "payments": pay}


def _revenue(alloc, pay, p_vals):
    return float(np.sum(p_vals * pay))


def _max_regret(alloc, pay, support):
    """Return the maximum regret (max_t max_{t'} (U(t,t') - U(t,t))) and max IR violation."""
    worst = 0.0
    support_list = list(support)
    for t in support_list:
        ti = t[0] * 4 + t[1]
        utt = t[0] * alloc[ti, 0] + t[1] * alloc[ti, 1] - pay[ti]
        for t2 in support_list:
            ti2 = t2[0] * 4 + t2[1]
            utt2 = t[0] * alloc[ti2, 0] + t[1] * alloc[ti2, 1] - pay[ti2]
            if utt2 - utt > worst:
                worst = utt2 - utt
        # IR: utt >= 0
        if -utt > worst:
            worst = -utt
    return worst


def _repair(alloc, pay, support):
    """Clamp tiny numerical violations to guarantee validity. Returns (alloc, pay, ok)."""
    alloc = np.clip(alloc, 0.0, 1.0)
    pay = np.clip(pay, 0.0, 6.0)
    # IR repair: pay[t] <= alloc[t] dot t
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            cap = i * alloc[idx, 0] + j * alloc[idx, 1]
            if pay[idx] > cap + 1e-9:
                pay[idx] = max(0.0, cap)
    # Enforce monotonicity of allocations
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if i < 3:
                idxn = (i + 1) * 4 + j
                for g in range(2):
                    if alloc[idx, g] > alloc[idxn, g]:
                        alloc[idxn, g] = alloc[idx, g]
            if j < 3:
                idxn = i * 4 + (j + 1)
                for g in range(2):
                    if alloc[idx, g] > alloc[idxn, g]:
                        alloc[idxn, g] = alloc[idx, g]
    # Re-apply IR cap after monotonicity
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            cap = i * alloc[idx, 0] + j * alloc[idx, 1]
            if pay[idx] > cap:
                pay[idx] = cap
    # Verify
    worst = _max_regret(alloc, pay, support)
    ok = worst <= 1e-8
    return alloc, pay, ok


def _solve_lp(p_vals, support):
    """Solve the exact LP. Returns (alloc, pay) or None."""
    all_types = [(i, j) for i in range(4) for j in range(4)]
    type_index = {t: i for i, t in enumerate(all_types)}
    n_vars = 48
    support_list = list(support)

    c = np.zeros(n_vars)
    for i in range(16):
        t = all_types[i]
        c[32 + i] = -p_vals[t[0], t[1]]

    A_ub = []
    b_ub = []

    # IR
    for t in support_list:
        ti = type_index[t]
        row = np.zeros(n_vars)
        row[32 + ti] = 1.0
        row[ti * 2 + 0] = -float(t[0])
        row[ti * 2 + 1] = -float(t[1])
        A_ub.append(row)
        b_ub.append(0.0)

    # Full IC
    for t in support_list:
        ti = type_index[t]
        for t2 in support_list:
            if t == t2:
                continue
            ti2 = type_index[t2]
            row = np.zeros(n_vars)
            row[32 + ti] = 1.0
            row[32 + ti2] = -1.0
            row[ti * 2 + 0] = -float(t[0])
            row[ti * 2 + 1] = -float(t[1])
            row[ti2 * 2 + 0] = float(t[0])
            row[ti2 * 2 + 1] = float(t[1])
            A_ub.append(row)
            b_ub.append(0.0)

    # Monotonicity
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if i < 3:
                idxn = (i + 1) * 4 + j
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx * 2 + g] = 1.0
                    row[idxn * 2 + g] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)
            if j < 3:
                idxn = i * 4 + (j + 1)
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx * 2 + g] = 1.0
                    row[idxn * 2 + g] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)
    bounds = [(0.0, 1.0)] * 32 + [(0.0, 6.0)] * 16

    best = None
    best_obj = -np.inf
    for method in ('highs', 'interior-point'):
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and res.x is not None:
                alloc = np.clip(res.x[:32].reshape(16, 2), 0, 1)
                pay = np.clip(res.x[32:], 0, 6)
                obj = _revenue(alloc, pay, p_vals)
                if obj > best_obj:
                    best_obj = obj
                    best = (alloc, pay)
        except Exception:
            pass

    return best


def _slsqp_polish(alloc0, pay0, p_vals, support, deadline):
    """Polish the LP solution using SLSQP. Returns (alloc, pay) or None."""
    all_types = [(i, j) for i in range(4) for j in range(4)]
    support_list = list(support)
    
    def objective(x):
        alloc = np.clip(x[:32].reshape(16, 2), 0, 1)
        pay = np.clip(x[32:], 0, 6)
        return -_revenue(alloc, pay, p_vals)

    constraints = []
    # IR constraints
    for t in support_list:
        ti = t[0] * 4 + t[1]
        def ir_con(x, ti=ti, t=t):
            alloc = x[:32].reshape(16, 2)
            pay = x[32:]
            return t[0] * alloc[ti, 0] + t[1] * alloc[ti, 1] - pay[ti]
        constraints.append({'type': 'ineq', 'fun': ir_con})
    
    # IC constraints
    for t in support_list:
        ti = t[0] * 4 + t[1]
        for t2 in support_list:
            if t == t2:
                continue
            ti2 = t2[0] * 4 + t2[1]
            def ic_con(x, ti=ti, ti2=ti2, t=t):
                alloc = x[:32].reshape(16, 2)
                pay = x[32:]
                utt = t[0] * alloc[ti, 0] + t[1] * alloc[ti, 1] - pay[ti]
                utt2 = t[0] * alloc[ti2, 0] + t[1] * alloc[ti2, 1] - pay[ti2]
                return utt - utt2
            constraints.append({'type': 'ineq', 'fun': ic_con})

    x0 = np.concatenate([alloc0.flatten(), pay0])
    bounds = [(0, 1)] * 32 + [(0, 6)] * 16

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            alloc = np.clip(res.x[:32].reshape(16, 2), 0, 1)
            pay = np.clip(res.x[32:], 0, 6)
            return alloc, pay
    except Exception:
        pass
    return None


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Parse inputs
    p_vals = _parse_distribution(distribution_spec)
    support_set = _parse_support(support)

    # Solve LP
    lp_result = _solve_lp(p_vals, support_set)

    if lp_result is not None:
        alloc, pay = lp_result
        # Try SLSQP polish if time permits
        if time.time() < deadline - 10.0:
            polished = _slsqp_polish(alloc, pay, p_vals, support_set, deadline)
            if polished is not None:
                alloc, pay = polished

        # Repair and verify
        alloc, pay, ok = _repair(alloc, pay, support_set)
        if ok:
            return {"allocations": alloc, "payments": pay}

    # Fallback
    return _fallback()
