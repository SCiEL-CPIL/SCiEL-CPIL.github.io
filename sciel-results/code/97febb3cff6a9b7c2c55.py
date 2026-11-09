# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.stats import norm

def _compute_P_fast(support, weights):
    """
    Fast exact computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n).
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()

    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)

    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]

    # count of elements strictly < threshold
    indices = np.searchsorted(sorted_sums, thresholds, side='left')

    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.where(mask_pos, indices - 1, 0)
    safe_indices = np.clip(safe_indices, 0, len(cum_probs) - 1)
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]

    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    return float(P)

def _gaussian_P(a, b):
    """
    Compute P[X1+X2+X3 < 2*X4] where Xi ~ N(a, 1) and X4 ~ N(b, 1).
    Let D = X1+X2+X3 - 2*X4. Then D ~ N(3a - 2b, 6).
    P = Phi((2b - 3a) / sqrt(6)).
    """
    mu = 3*a - 2*b
    sigma = np.sqrt(6.0)
    z = -mu / sigma  # = (2b - 3a) / sqrt(6)
    return float(norm.cdf(z))

def _gaussian_to_discrete(a, b, n_points=8, n_samples=50):
    """
    Convert a 2-point Gaussian mixture approximation to a discrete measure.
    The Gaussian mixture approximates the sum distribution. We discretize the
    underlying point mass locations and assign weights based on the mixture.
    
    We use a 2-component mixture: each Xi is a mixture of two point masses.
    Actually, let's just use the optimal 2-point baseline and perturb it.
    
    Better approach: Use the Gaussian insight to guide the support/weight design.
    For a 2-point measure {0, a} with weight q at 0:
    P[X1+X2+X3 < 2*X4] can be computed exactly.
    We know the optimum is at q=1/4, P=229/256≈0.89453125.
    
    Let's try 3-point measures inspired by Gaussian structure.
    """
    # For a 3-point measure, use points at 0, m, 2m with weights q, 1-2q, q
    # (symmetric around m, similar to a 3-point quadrature)
    # Actually, let's just return the 2-point baseline and let the optimizer refine.
    q = 0.25
    m = 1.0
    s = np.array([0.0, m, 2.0*m], dtype=np.float64)
    w = np.array([q, 1.0-2.0*q, q], dtype=np.float64)
    w = np.maximum(w, 1e-10)
    w = w / w.sum()
    return s, w

def _optimize_weights_for_support(s, z0=None, maxiter=2000):
    """Optimize weights (via softmax) for a fixed support."""
    n = len(s)
    if z0 is None:
        z0 = np.zeros(n)

    def objective_softmax(z):
        exp_z = np.exp(z - np.max(z))
        w = exp_z / exp_z.sum()
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective_softmax, z0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        z_opt = res.x
        exp_z = np.exp(z_opt - np.max(z_opt))
        w_opt = exp_z / exp_z.sum()
        P_opt = -res.fun
        return s, w_opt, P_opt
    except Exception:
        return None, None, 0.0

def _joint_optimize(s_init, w_init, maxiter=2000):
    """Jointly optimize support and weights. Support = cumulative positive steps."""
    n = len(s_init)
    steps = np.diff(s_init)
    if np.any(steps <= 0):
        steps = np.maximum(np.abs(steps), 0.01)
    log_steps = np.log(steps)
    log_w = np.log(np.maximum(w_init, 1e-10))
    params = np.concatenate([log_steps, log_w])

    def objective(params):
        log_s = params[:n - 1]
        log_w = params[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)

    try:
        res = minimize(objective, params, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        log_s = res.x[:n - 1]
        log_w = res.x[n - 1:]
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

def _diff_evolve_support(n_points, bounds_s, bounds_w, maxiter=100):
    """
    Use differential evolution to jointly optimize support and weights.
    """
    # Parameters: log_steps (n-1) + log_w (n)
    # Total: 2n-1 parameters
    n_params = 2 * n_points - 1
    
    def objective(params):
        log_s = params[:n_points - 1]
        log_w = params[n_points - 1:]
        s = np.zeros(n_points)
        for i in range(1, n_points):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        return -_compute_P_fast(s, w)
    
    # Define bounds
    # log_steps in [-5, 5] (steps from ~0.007 to ~148)
    # log_w in [-10, 0] (weights from ~0 to 1)
    bounds = [(-5.0, 5.0)] * (n_points - 1) + [(-10.0, 0.0)] * n_points
    
    try:
        res = differential_evolution(
            objective, bounds, maxiter=maxiter, tol=1e-8, 
            polish=True, seed=42, init='sobol', popsize=15
        )
        log_s = res.x[:n_points - 1]
        log_w = res.x[n_points - 1:]
        s = np.zeros(n_points)
        for i in range(1, n_points):
            s[i] = s[i - 1] + np.exp(log_s[i - 1])
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        P = -res.fun
        return s, w, P
    except Exception:
        return None, None, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a discrete measure mu on [0, inf) maximizing P[X1+X2+X3 < 2*X4].
    Returns (support, weights), 1-D float arrays of equal length n >= 1.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin

    # 1. Exact optimal 2-point baseline: {0, a} with P(X=0)=q=1/4.
    #    P = 1 - q*(1-q)^3, maximized at q=1/4 -> 229/256 ~= 0.89453125.
    #    The value is independent of a (a>0); use a tiny a to stay safely positive.
    q_opt = 0.25
    a_opt = 1e-3
    s_base = np.array([0.0, a_opt], dtype=np.float64)
    w_base = np.array([q_opt, 1.0 - q_opt], dtype=np.float64)
    P_base = _compute_P_fast(s_base, w_base)

    best_P = P_base
    best_support = s_base.copy()
    best_weights = w_base.copy()

    # 2. Exact 2-point verification over a fine grid (cheap, robust).
    for q in np.linspace(0.01, 0.99, 99):
        s2 = np.array([0.0, a_opt], dtype=np.float64)
        w2 = np.array([q, 1.0 - q], dtype=np.float64)
        P2 = _compute_P_fast(s2, w2)
        if P2 > best_P:
            best_P = P2
            best_support = s2.copy()
            best_weights = w2.copy()

    # 3. Time-boxed exploration of larger supports.
    supports_to_try = []
    for n in [3, 4, 5, 6, 8]:
        for base in [1.5, 2.0, 3.0, 5.0, 10.0]:
            s = np.array([base ** i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
    for n in [3, 4, 5, 6, 8]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
    for n in [5, 6, 8]:
        for k in range(2, min(n, 5)):
            s = np.zeros(n)
            s[:k] = np.arange(k, dtype=np.float64)
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
    for n in [4, 5, 6]:
        s = np.array([0.0] + [2.0 ** i for i in range(1, n)], dtype=np.float64)
        supports_to_try.append(s)
    
    # Add symmetric 3-point supports inspired by Gaussian quadrature
    for n in [3, 4, 5]:
        for m in [0.5, 1.0, 2.0, 5.0]:
            s = np.array([0.0, m, 2.0*m] + [m*(2*n - 2 - 2*i) for i in range(n - 3)], dtype=np.float64)
            s = np.sort(np.maximum(s, 0.0))
            supports_to_try.append(s)

    candidates = []
    for s in supports_to_try:
        if time.time() > deadline - 120.0:
            break
        n_s = len(s)
        initial_zs = [
            np.zeros(n_s),
            np.concatenate([np.log([1.5]), np.zeros(n_s - 1)]),
            np.log(np.linspace(1.0, 0.3, n_s)),
        ]
        for z0 in initial_zs:
            if time.time() > deadline - 100.0:
                break
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s, z0, maxiter=1500)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                candidates.append((P_opt, s_opt.copy(), w_opt.copy()))

    if time.time() < deadline - 60.0:
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:5] if len(candidates) >= 5 else candidates
        for P_c, s_c, w_c in top_candidates:
            if time.time() > deadline - 30.0:
                break
            s_j, w_j, P_j = _joint_optimize(s_c, w_c, maxiter=2000)
            if s_j is not None and P_j > best_P:
                best_P = P_j
                best_support = s_j.copy()
                best_weights = w_j.copy()

    # 4. Differential evolution for joint optimization of small supports.
    if time.time() < deadline - 30.0:
        for n in [3, 4, 5]:
            if time.time() > deadline - 20.0:
                break
            s_de, w_de, P_de = _diff_evolve_support(n, None, None, maxiter=80)
            if s_de is not None and P_de > best_P:
                best_P = P_de
                best_support = s_de.copy()
                best_weights = w_de.copy()

    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline - 10.0:
                break
            n = np.random.randint(3, 10)
            scale = np.random.uniform(1.0, 20.0)
            s_raw = np.sort(np.random.uniform(0, scale, n))
            for i in range(1, n):
                if s_raw[i] - s_raw[i - 1] < 0.01:
                    s_raw[i] = s_raw[i - 1] + 0.01
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s_raw, np.zeros(n), maxiter=1000)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()

    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, a_opt])
        best_weights = np.array([q_opt, 1.0 - q_opt])
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w

    return best_support, best_weights

# EVOLVE-BLOCK-END
