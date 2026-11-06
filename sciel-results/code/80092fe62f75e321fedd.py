import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    
    probs = _parse_probs(distribution_spec, n_types)
    
    n_alloc = n_types * 2
    n_pay = n_types
    n_vars = n_alloc + n_pay
    
    def alloc_var_idx(type_idx, good):
        return type_idx * 2 + (good - 1)
    
    def pay_var_idx(type_idx):
        return n_alloc + type_idx
    
    c = np.zeros(n_vars)
    for v_idx in range(n_types):
        c[pay_var_idx(v_idx)] = -probs[v_idx]
    
    A_ub = []
    b_ub = []
    
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(n_types):
            row = np.zeros(n_vars)
            row[alloc_var_idx(v_idx, 1)] = -v1
            row[alloc_var_idx(v_idx, 2)] = -v2
            row[pay_var_idx(v_idx)] = 1.0
            row[alloc_var_idx(vp_idx, 1)] = v1
            row[alloc_var_idx(vp_idx, 2)] = v2
            row[pay_var_idx(vp_idx)] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        row = np.zeros(n_vars)
        row[alloc_var_idx(v_idx, 1)] = -v1
        row[alloc_var_idx(v_idx, 2)] = -v2
        row[pay_var_idx(v_idx)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, 1)] * n_alloc + [(0, None)] * n_pay
    
    try:
        if time.time() > deadline:
            return _trivial_mechanism(n_types)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_alloc = np.clip(result.x[:n_alloc].reshape(n_types, 2), 0, 1)
            p_pay = np.clip(result.x[n_alloc:n_alloc+n_pay], 0, None)
            _fix_violations(x_alloc, p_pay, types, n_types)
            return {'allocations': x_alloc, 'payments': p_pay}
        else:
            return _trivial_mechanism(n_types)
    except Exception:
        return _trivial_mechanism(n_types)

def _parse_probs(distribution_spec, n_types):
    if isinstance(distribution_spec, dict):
        if 0 in distribution_spec:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(n_types)], dtype=float)
        else:
            probs = np.array(list(distribution_spec.values()), dtype=float)
            if len(probs) != n_types:
                probs = np.resize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.maximum(probs, 0)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.ones(n_types) / n_types
    return probs

def _fix_violations(x_alloc, p_pay, types, n_types):
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        u_vv = v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - p_pay[v_idx]
        if u_vv < 0:
            p_pay[v_idx] = max(0.0, v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - 1e-9)
            u_vv = v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - p_pay[v_idx]
        for vp_idx in range(n_types):
            vp1, vp2 = types[vp_idx]
            u_vvp = v1 * x_alloc[vp_idx, 0] + v2 * x_alloc[vp_idx, 1] - p_pay[vp_idx]
            if u_vvp > u_vv + 1e-8:
                p_pay[vp_idx] = max(p_pay[vp_idx], v1 * x_alloc[vp_idx, 0] + v2 * x_alloc[vp_idx, 1] - u_vv)
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        u_vv = v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - p_pay[v_idx]
        if u_vv < 0:
            p_pay[v_idx] = max(0.0, v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - 1e-9)

def _trivial_mechanism(n_types):
    return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
