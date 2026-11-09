# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _clip_and_adjust(domain, values, K):
    """Apply the evaluator's clipping and f(1) adjustment.
    Returns adjusted values array aligned to 1..K (zeros where k not in domain).
    """
    clipped = np.clip(values, -10.0, 10.0)

    f = np.zeros(K + 1, dtype=np.float64)
    for i, k in enumerate(domain):
        if 1 <= k <= K:
            f[k] = clipped[i]

    S = np.sum(clipped / domain.astype(np.float64))
    f[1] = f[1] - S
    return f


def _evaluate(f, K):
    """Compute M(x) for all x in 1..10K and the objective a.
    Returns (feasible, a).
    """
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1)
    k_vals = np.arange(1, K + 1)
    M = np.dot(np.floor(x_vals[:, None].astype(np.float64) / k_vals[None, :].astype(np.float64)), f[1:])
    feasible = bool(np.all(M <= 1.0001))
    if not feasible:
        return False, 0.0
    log_k = np.log(k_vals)
    a = -np.sum(f[1:] * log_k / k_vals)
    return True, float(a)


def _solve_lp_cutting_plane(K, time_limit):
    """Solve LP using cutting-plane: start with subset of constraints, add violated ones.
    Variables: f(1)..f(K). Objective: maximize -sum f(k)*log(k)/k.
    Constraints: M(x) = sum_k f(k)*floor(x/k) <= 1 for x=1..10K.
    Bounds: f(1) free, f(k) in [-10,10] for k>=2.
    """
    t0 = time.time()
    n_vars = K + 1  # f(1) to f(K)
    k_vals = np.arange(1, K + 1, dtype=np.float64)
    x_max = 10 * K
    
    # Objective: minimize sum f(k)*log(k)/k (which is -a)
    c = np.log(k_vals) / k_vals
    
    # Bounds
    bounds = [(None, None)]  # f(1) free
    for k in range(2, K + 1):
        bounds.append((-10.0, 10.0))
    
    # Start with a coarse set of x constraints
    # x values that are multiples of small numbers are most constraining
    initial_x = list(range(1, min(100, x_max + 1)))
    # Add multiples of 2, 3, 5, 7 up to x_max
    for p in [2, 3, 5, 7, 11]:
        for m in range(p, x_max + 1, p):
            if m not in initial_x:
                initial_x.append(m)
    initial_x = sorted(set(initial_x))
    
    # Build initial A_ub
    x_arr = np.array(initial_x, dtype=np.float64)
    A_ub = np.floor(x_arr[:, None] / k_vals[None, :])
    b_ub = np.ones(len(initial_x), dtype=np.float64)
    
    active_x = set(initial_x)
    f_opt = None
    
    for iteration in range(50):
        if time.time() - t0 > time_limit:
            break
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            break
        
        if result.status != 0:
            break
        
        f_opt = result.x
        
        # Check all x in 1..10K for violations
        x_all = np.arange(1, x_max + 1, dtype=np.float64)
        M_all = np.dot(np.floor(x_all[:, None] / k_vals[None, :]), f_opt)
        
        # Find violated constraints (M(x) > 1.0001)
        violated = np.where(M_all > 1.0001)[0]
        if len(violated) == 0:
            break
        
        # Add violated constraints
        new_x = [int(x_all[i]) for i in violated if int(x_all[i]) not in active_x]
        if len(new_x) == 0:
            break
        
        active_x.update(new_x)
        # Rebuild A_ub with all active constraints
        x_active = np.array(sorted(active_x), dtype=np.float64)
        A_ub = np.floor(x_active[:, None] / k_vals[None, :])
        b_ub = np.ones(len(x_active), dtype=np.float64)
    
    if f_opt is None:
        return None
    
    # Build domain and values from f_opt
    # Only include k where f(k) is significantly nonzero, plus k=1
    domain_list = [1]
    values_list = [f_opt[0]]
    for k in range(2, K + 1):
        if abs(f_opt[k]) > 1e-9:
            domain_list.append(k)
            values_list.append(f_opt[k])
    
    domain = np.array(domain_list, dtype=np.int64)
    values = np.array(values_list, dtype=np.float64)
    return domain, values


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    t_start = time.time()
    deadline = t_start + 590.0  # 600s budget with safety margin

    # Seed fallback
    domain_seed = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values_seed = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_domain = domain_seed.copy()
    best_values = values_seed.copy()
    best_score = 1.0

    # Try LP with cutting-plane for various K
    for K in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200]:
        if time.time() > deadline - 30.0:
            break
        time_limit = min(30.0, deadline - time.time() - 5.0)
        if time_limit < 5.0:
            break
        result = _solve_lp_cutting_plane(K, time_limit)
        if result is not None:
            dom, val = result
            f = _clip_and_adjust(dom, val, K)
            feasible, a = _evaluate(f, K)
            if feasible and a > best_score:
                best_score = a
                best_domain = dom.copy()
                best_values = val.copy()

    return best_domain, best_values


# EVOLVE-BLOCK-END
