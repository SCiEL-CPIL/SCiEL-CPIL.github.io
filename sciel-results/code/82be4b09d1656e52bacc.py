# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

def _differential_evolution_search(n, bounds_s, bounds_w, maxiter=200, popsize=10, seed=42):
    """
    Use differential evolution to jointly optimize support and weights.
    Support is constrained to be non-negative and sorted.
    Weights are constrained to be non-negative.
    """
    # Parameter vector: [s_0, s_1, ..., s_{n-1}, w_0, w_1, ..., w_{n-1}]
    # s_i are the support points, w_i are the weights.
    # We need to ensure s is sorted and non-negative, and w sums to 1 (or positive).
    
    # Strategy:
    # s_0 >= 0
    # s_i >= s_{i-1} + eps for i > 0
    # w_i >= 0
    
    # Let's parameterize s as cumulative steps to ensure sorting and non-negativity.
    # s_0 = step_0
    # s_i = s_{i-1} + step_i
    # All steps > 0.
    # And w via softmax or just raw positive weights.
    
    # Let's use log-softmax for weights to ensure they sum to 1 and are positive.
    # And log-steps for support to ensure positive steps.
    
    # Parameters: [log_step_0, ..., log_step_{n-1}, log_w_0, ..., log_w_{n-1}]
    # Total params: 2n
    
    # Bounds for log_steps: log(1e-3) to log(100) -> [-6.9, 4.6]
    # Bounds for log_w: -20 to 20
    
    lb = np.concatenate([np.full(n, -6.9), np.full(n, -20.0)])
    ub = np.concatenate([np.full(n, 4.6), np.full(n, 20.0)])
    
    def objective(params):
        log_steps = params[:n]
        log_w = params[n:]
        
        steps = np.exp(log_steps)
        s = np.cumsum(steps)
        
        # Normalize weights
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        # Check validity
        if np.any(s < -1e-9) or np.any(w < -1e-9) or w.sum() <= 1e-9:
            return 1.0  # Large penalty
        
        P = _compute_P_fast(s, w)
        return -P
    
    try:
        res = differential_evolution(objective, list(zip(lb, ub)), 
                                     maxiter=maxiter, popsize=popsize, 
                                     seed=seed, tol=1e-8, atol=1e-10,
                                     polish=True, init='sobol')
        
        log_steps = res.x[:n]
        log_w = res.x[n:]
        
        steps = np.exp(log_steps)
        s = np.cumsum(steps)
        
        max_log_w = np.max(log_w)
        exp_w = np.exp(log_w - max_log_w)
        w = exp_w / np.sum(exp_w)
        
        P = -res.fun
        return s, w, P
    except Exception as e:
        # Fallback to Nelder-Mead if DE fails
        return None, None, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin

    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    # 1. Analytical 2-point solution as a strong baseline
    # P[X1+X2+X3 < 2X4] for {0, a} with P(0)=q, P(a)=1-q
    # X4=0 (prob q): 2X4=0. X1+X2+X3 < 0 is impossible (since Xi >= 0). So P=0.
    # X4=a (prob 1-q): 2X4=2a.
    #   X1+X2+X3 < 2a.
    #   Possible sums of X1+X2+X3:
    #   0 (all 0): prob q^3. 0 < 2a is true.
    #   a (one a, two 0s): prob 3*q^2*(1-q). a < 2a is true (if a>0).
    #   2a (two a, one 0): prob 3*q*(1-q)^2. 2a < 2a is false.
    #   3a (three a): prob (1-q)^3. 3a < 2a is false.
    # So P = (1-q) * (q^3 + 3*q^2*(1-q))
    #      = (1-q) * q^2 * (q + 3(1-q))
    #      = (1-q) * q^2 * (3 - 2q)
    # Let f(q) = (1-q)q^2(3-2q) = 3q^2 - 2q^3 - 3q^3 + 2q^4 = 2q^4 - 5q^3 + 3q^2
    # f'(q) = 8q^3 - 15q^2 + 6q = q(8q^2 - 15q + 6)
    # Roots of 8q^2 - 15q + 6 = 0: q = (15 +- sqrt(225 - 192))/16 = (15 +- sqrt(33))/16
    # sqrt(33) ~ 5.744
    # q1 = (15 - 5.744)/16 = 9.256/16 = 0.5785
    # q2 = (15 + 5.744)/16 = 20.744/16 = 1.296 (invalid)
    # So optimal q = (15 - sqrt(33))/16 ~ 0.5785
    # Let's verify this against the previous program's q_opt.
    # Previous program used q_opt = (15.0 - np.sqrt(33.0)) / 16.0.
    # Let's calculate the value.
    q_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s_2pt = np.array([0.0, 1.0])
    w_2pt = np.array([q_opt, 1.0 - q_opt])
    P_2pt = _compute_P_fast(s_2pt, w_2pt)
    if P_2pt > best_P:
        best_P = P_2pt
        best_support = s_2pt.copy()
        best_weights = w_2pt.copy()

    # 2. Differential Evolution for small n (3, 4, 5)
    # This is the "heavy-tailed" search strategy, exploring the joint space of support and weights.
    for n in [3, 4, 5, 6]:
        if time.time() > deadline - 60.0:
            break
        
        # Run DE for a few seconds
        t_start = time.time()
        s_de, w_de, P_de = _differential_evolution_search(n, None, None, maxiter=50, popsize=10, seed=42)
        
        if s_de is not None and P_de > best_P:
            best_P = P_de
            best_support = s_de.copy()
            best_weights = w_de.copy()
        
        # Refine with Nelder-Mead if DE found something good
        if s_de is not None and P_de > best_P * 0.9:
            if time.time() < deadline - 30.0:
                s_ref, w_ref, P_ref = _joint_optimize(s_de, w_de, maxiter=1000)
                if s_ref is not None and P_ref > best_P:
                    best_P = P_ref
                    best_support = s_ref.copy()
                    best_weights = w_ref.copy()

    # 3. Explore diverse support structures with weight optimization
    supports_to_try = []
    
    # Geometric progressions
    for n in [3, 4, 5, 6, 8]:
        for base in [1.5, 2.0, 3.0, 5.0, 10.0]:
            s = np.array([base**i for i in range(n)], dtype=np.float64)
            supports_to_try.append(s)
    
    # Arithmetic progressions
    for n in [3, 4, 5, 6, 8]:
        for step in [0.5, 1.0, 2.0, 5.0]:
            s = np.arange(n, dtype=np.float64) * step
            supports_to_try.append(s)
    
    # Mixed: small cluster + large jumps
    for n in [5, 6, 8]:
        for k in range(2, min(n, 5)):
            s = np.zeros(n)
            s[:k] = np.arange(k, dtype=np.float64)
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
    
    # Exponential-like: 0, 1, 2, 4, 8, ...
    for n in [4, 5, 6]:
        s = np.array([0.0] + [2.0**i for i in range(1, n)], dtype=np.float64)
        supports_to_try.append(s)
    
    # Try each support with weight optimization
    candidates = []
    for s in supports_to_try:
        if time.time() > deadline - 120.0:
            break
        
        # Use multiple initial guesses for weights
        n_s = len(s)
        initial_zs = [
            np.zeros(n_s),  # uniform
            np.concatenate([np.log([1.5]), np.zeros(n_s-1)]),  # biased toward 0
            np.log(np.linspace(1.0, 0.3, n_s)),  # decreasing
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
    
    # 4. Joint optimization on the best candidates from step 2
    if time.time() < deadline - 60.0:
        # Sort candidates by P and take top few
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:5] if len(candidates) >= 5 else candidates
        
        for P_c, s_c, w_c in top_candidates:
            if time.time() > deadline - 30.0:
                break
            
            # Joint optimization starting from the weight-optimized solution
            s_j, w_j, P_j = _joint_optimize(s_c, w_c, maxiter=2000)
            if s_j is not None and P_j > best_P:
                best_P = P_j
                best_support = s_j.copy()
                best_weights = w_j.copy()
    
    # 5. Random support exploration in remaining time
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline - 10.0:
                break
            
            n = np.random.randint(3, 10)
            # Random sorted support with varying scales
            scale = np.random.uniform(1.0, 20.0)
            s_raw = np.sort(np.random.uniform(0, scale, n))
            # Ensure minimum gap
            for i in range(1, n):
                if s_raw[i] - s_raw[i-1] < 0.01:
                    s_raw[i] = s_raw[i-1] + 0.01
            
            # Quick weight optimization
            s_opt, w_opt, P_opt = _optimize_weights_for_support(s_raw, np.zeros(n), maxiter=1000)
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
    
    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
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
