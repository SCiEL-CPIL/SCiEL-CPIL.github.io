import numpy as np
import time
from scipy.optimize import minimize

_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# Precompute bin indices
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
_h0_vals = _x_flat.astype(np.int64)
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

_ln2 = np.log(2.0)


def _entropy_from_probs(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    nz = probs[probs > 1e-15]
    if len(nz) == 0:
        return 0.0
    return -np.sum(nz * np.log2(nz))


def _score(w):
    """Compute D/M ratio from flat weight array w (assumed non-negative, sum 1)."""
    d_probs = np.bincount(_d_vals, weights=w, minlength=_N_D)
    h0_probs = np.bincount(_h0_vals, weights=w, minlength=_N_H0)
    h1_probs = np.bincount(_h1_vals, weights=w, minlength=_N_H1)
    h2_probs = np.bincount(_h2_vals, weights=w, minlength=_N_H2)
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    return D / M


def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return None
    return P / s


def _log_softmax_to_prob(params):
    log_p = params
    max_log_p = np.max(log_p)
    log_probs = log_p - max_log_p
    exp_probs = np.exp(log_probs)
    return exp_probs / np.sum(exp_probs)


def _neg_ratio(params):
    probs = _log_softmax_to_prob(params)
    return -_score(probs)


def _neg_ratio_smooth(params, temperature=0.1):
    """Smooth objective with soft-max for M."""
    probs = _log_softmax_to_prob(params)
    d_probs = np.bincount(_d_vals, weights=probs, minlength=_N_D)
    h0_probs = np.bincount(_h0_vals, weights=probs, minlength=_N_H0)
    h1_probs = np.bincount(_h1_vals, weights=probs, minlength=_N_H1)
    h2_probs = np.bincount(_h2_vals, weights=probs, minlength=_N_H2)
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    # Soft max via log-sum-exp
    hs = np.array([H0, H1, H2])
    scaled = hs / temperature
    max_scaled = np.max(scaled)
    soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
    
    if soft_max <= 1e-9:
        return 0.0
    return -D / soft_max


def _grad_ratio(params, soft_temp=0.0):
    """
    Compute gradient of D/M with respect to log-softmax parameters.
    Returns (value, gradient).
    If soft_temp > 0, uses soft-max for smooth gradients on M.
    """
    probs = _log_softmax_to_prob(params)
    n = len(probs)
    
    d_probs = np.bincount(_d_vals, weights=probs, minlength=_N_D)
    h0_probs = np.bincount(_h0_vals, weights=probs, minlength=_N_H0)
    h1_probs = np.bincount(_h1_vals, weights=probs, minlength=_N_H1)
    h2_probs = np.bincount(_h2_vals, weights=probs, minlength=_N_H2)
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    if soft_temp > 0:
        # Soft max
        hs = np.array([H0, H1, H2])
        scaled = hs / soft_temp
        max_scaled = np.max(scaled)
        M = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * soft_temp
        # Soft max gradient: softmax weights
        softmax_w = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
    else:
        M = max(H0, H1, H2)
        # Hard max: subgradient picks argmax
        hs = np.array([H0, H1, H2])
        max_idx = np.argmax(hs)
        softmax_w = np.zeros(3)
        softmax_w[max_idx] = 1.0
    
    if M <= 1e-9:
        return 0.0, np.zeros(n)
    
    value = D / M
    
    ln2 = _ln2
    
    def _entropy_grad_flat(probs, bin_indices, n_bins):
        """Gradient of entropy w.r.t. each cell probability."""
        bin_probs = np.bincount(bin_indices, weights=probs, minlength=n_bins)
        dH_dq = np.zeros(n_bins)
        nz = bin_probs > 1e-15
        dH_dq[nz] = -np.log2(bin_probs[nz]) - 1.0 / ln2
        return dH_dq[bin_indices]
    
    gD = _entropy_grad_flat(probs, _d_vals, _N_D)
    gH0 = _entropy_grad_flat(probs, _h0_vals, _N_H0)
    gH1 = _entropy_grad_flat(probs, _h1_vals, _N_H1)
    gH2 = _entropy_grad_flat(probs, _h2_vals, _N_H2)
    
    # gM = weighted combination of gradients
    gM = softmax_w[0] * gH0 + softmax_w[1] * gH1 + softmax_w[2] * gH2
    
    d_value_d_probs = (gD * M - value * gM) / M
    
    mean_grad = np.dot(d_value_d_probs, probs)
    grad = probs * (d_value_d_probs - mean_grad)
    
    return value, grad


def _fast_optimize(params, max_iter=5000, time_limit=30.0, soft_temp=0.0):
    """
    Fast gradient-based optimizer using exact analytic gradients.
    Uses Adam with backtracking line search.
    """
    start = time.time()
    n = len(params)
    
    best_params = params.copy()
    value, grad = _grad_ratio(params, soft_temp)
    best_value = value
    
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros(n)
    v = np.zeros(n)
    lr = 0.05
    eps = 1e-8
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        if value > best_value:
            best_value = value
            best_params = params.copy()
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        # Adaptive learning rate: decay over iterations, scale by gradient magnitude
        base_lr = 0.05 * (1.0 / (1.0 + it / 300.0))
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-10:
            lr = min(0.1, max(0.0005, base_lr * (1.0 / (gnorm + 0.1))))
        
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        alpha = 1.0
        c1 = 1e-4
        new_params = params - alpha * step
        new_value, new_grad = _grad_ratio(new_params, soft_temp)
        
        # Backtracking line search (limited iterations to save time)
        for _ in range(15):
            if new_value > value + c1 * alpha * np.dot(grad, step):
                break
            alpha *= 0.5
            if alpha < 1e-6:
                break
            new_params = params - alpha * step
            new_value, new_grad = _grad_ratio(new_params, soft_temp)
        
        params = new_params
        value = new_value
        grad = new_grad
        
        # Early stopping if converged
        if it > 200 and abs(value - best_value) < 1e-10:
            break
    
    return best_params, best_value


def _build_family_A(c, alpha, beta):
    """Family A: uniform over x=y+c (diagonals), weighted by exp(-alpha*x^2) * exp(-beta*y^2)."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            P[y, x] = np.exp(-alpha * x * x - beta * y * y)
    return _normalize(P)


def _build_family_B(c, gamma, delta):
    """Family B: weight cells on x=y+c line, Gaussian in position along diagonal."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            mid = (x + y) / 2.0
            P[y, x] = np.exp(-gamma * (mid - 4.5) ** 2 - delta * (x - y) ** 2)
    return _normalize(P)


def _build_family_C(c, alpha, beta, center_x, center_y, sigma):
    """Family C: 2D Gaussian centered at (center_x, center_y) restricted to x-y=c."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            dx = x - center_x
            dy = y - center_y
            P[y, x] = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    return _normalize(P)


def _build_family_D(c, weights):
    """Family D: arbitrary weights on x=y=c line."""
    P = np.zeros((_N, _N))
    valid = []
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            valid.append((y, x))
    if len(valid) == 0:
        return None
    w = np.array(weights[:len(valid)])
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 1e-15:
        return None
    w = w / s
    for (y, x), val in zip(valid, w):
        P[y, x] = val
    return P


def _refine_scipy(P_init, timeout_remaining):
    """Refine using scipy minimize with aggressive decreasing temperature."""
    if P_init is None:
        return None
    P_clipped = np.clip(P_init, 1e-15, None)
    P_norm = P_clipped / P_clipped.sum()
    log_params = np.log(P_norm.flatten())
    
    best_P = P_init.copy()
    best_score = _score(P_init.flatten())
    
    temps = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.06, 0.04, 0.025, 0.015, 0.01, 0.005]
    start = time.time()
    
    for temp in temps:
        if time.time() - start > timeout_remaining - 2:
            break
        try:
            result = minimize(
                _neg_ratio_smooth,
                log_params,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            opt_params = result.x
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            opt_score = _score(opt_P.flatten())
            
            if opt_score > best_score:
                best_score = opt_score
                best_P = opt_P.copy()
                log_params = np.log(np.clip(opt_P, 1e-15, None).flatten() / np.clip(opt_P, 1e-15, None).sum())
        except Exception:
            pass
    
    return best_P


def _coordinate_descent_polish(P, time_limit=5.0):
    """Systematic coordinate descent: explore each cell with multiple step sizes."""
    if P is None:
        return None
    P = P.copy()
    score = _score(P.flatten())
    start = time.time()
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    while time.time() - start < time_limit:
        improved = False
        for step in step_sizes:
            if time.time() - start > time_limit - 0.1:
                break
            for i in range(_N):
                for j in range(_N):
                    if time.time() - start > time_limit - 0.05:
                        break
                    new_P = P.copy()
                    new_P[i, j] += step
                    new_P = np.clip(new_P, 0, None)
                    s = new_P.sum()
                    if s > 1e-9:
                        new_P = new_P / s
                        new_score = _score(new_P.flatten())
                        if new_score > score:
                            P = new_P
                            score = new_score
                            improved = True
                    new_P = P.copy()
                    new_P[i, j] -= step
                    new_P = np.clip(new_P, 0, None)
                    s = new_P.sum()
                    if s > 1e-9:
                        new_P = new_P / s
                        new_score = _score(new_P.flatten())
                        if new_score > score:
                            P = new_P
                            score = new_score
                            improved = True
        if not improved:
            break
    
    return P


def _hill_climb(P, time_limit=5.0):
    """Local hill climbing refinement with coordinate descent polish."""
    if P is None:
        return None
    P = P.copy()
    score = _score(P.flatten())
    step_size = 0.2
    start = time.time()
    no_improve = 0
    
    while time.time() - start < time_limit * 0.6:
        i, j = np.random.randint(0, _N, 2)
        delta = np.random.uniform(-step_size, step_size)
        new_P = P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        new_score = _score(new_P.flatten())
        
        if new_score > score:
            P = new_P
            score = new_score
            step_size *= 0.98
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 200 or step_size < 0.0005:
                break
    
    remaining = time_limit - (time.time() - start)
    if remaining > 1.0:
        P = _coordinate_descent_polish(P, time_limit=remaining)
    
    return P


def _dc_optimize(P_init, time_limit=30.0):
    """
    DC-style optimization: maximize D subject to H_k <= T for each k in {0,1,2},
    with T starting at the current M and decreasing.
    Uses penalty method with smooth surrogates.
    """
    if P_init is None:
        return None
    P = P_init.copy()
    score = _score(P.flatten())
    start = time.time()
    
    # Get current M and which entropy is the max
    d_probs = np.bincount(_d_vals, weights=P.flatten(), minlength=_N_D)
    h0_probs = np.bincount(_h0_vals, weights=P.flatten(), minlength=_N_H0)
    h1_probs = np.bincount(_h1_vals, weights=P.flatten(), minlength=_N_H1)
    h2_probs = np.bincount(_h2_vals, weights=P.flatten(), minlength=_N_H2)
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return P
    
    # Start with T = M, then decrease
    T = M
    penalty_weight = 100.0
    
    for outer in range(20):
        if time.time() - start > time_limit - 1:
            break
        
        # Decrease T gradually
        T_new = M * (1 - 0.05 * (outer + 1))
        T_new = max(T_new, M * 0.5)  # Don't go below 50% of original M
        
        # Use penalty method: maximize D - penalty * max(0, H_k - T_new)^2
        log_params = np.log(np.clip(P, 1e-15, None).flatten() / np.clip(P, 1e-15, None).sum())
        
        def _penalty_obj(params):
            probs = _log_softmax_to_prob(params)
            d_probs = np.bincount(_d_vals, weights=probs, minlength=_N_D)
            h0_probs = np.bincount(_h0_vals, weights=probs, minlength=_N_H0)
            h1_probs = np.bincount(_h1_vals, weights=probs, minlength=_N_H1)
            h2_probs = np.bincount(_h2_vals, weights=probs, minlength=_N_H2)
            
            D = _entropy_from_probs(d_probs)
            H0 = _entropy_from_probs(h0_probs)
            H1 = _entropy_from_probs(h1_probs)
            H2 = _entropy_from_probs(h2_probs)
            
            # Penalty for exceeding T
            pen = 0.0
            for H in [H0, H1, H2]:
                excess = max(0.0, H - T_new)
                pen += excess * excess
            pen *= penalty_weight
            
            return -(D - pen)
        
        try:
            result = minimize(
                _penalty_obj,
                log_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            opt_params = result.x
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            opt_score = _score(opt_P.flatten())
            
            if opt_score > score:
                score = opt_score
                P = opt_P.copy()
                # Update M
                d_probs = np.bincount(_d_vals, weights=P.flatten(), minlength=_N_D)
                h0_probs = np.bincount(_h0_vals, weights=P.flatten(), minlength=_N_H0)
                h1_probs = np.bincount(_h1_vals, weights=P.flatten(), minlength=_N_H1)
                h2_probs = np.bincount(_h2_vals, weights=P.flatten(), minlength=_N_H2)
                M = max(_entropy_from_probs(h0_probs), _entropy_from_probs(h1_probs), _entropy_from_probs(h2_probs))
        except Exception:
            pass
        
        # Increase penalty weight
        penalty_weight *= 2.0
        if penalty_weight > 1e6:
            penalty_weight = 1e6
    
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    np.random.seed(12345)
    start_time = time.time()
    timeout = 620  # safety margin
    
    best_P = None
    best_score = 0.0
    
    def _update_best(P):
        nonlocal best_P, best_score
        if P is None:
            return
        score = _score(P.flatten())
        if score > best_score:
            best_score = score
            best_P = P.copy()
    
    # Phase 1: Explore structured families (reduced grid for speed)
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for alpha in [0, 0.1, 0.3, 0.8]:
            for beta in [0, 0.1, 0.3, 0.8]:
                P = _build_family_A(c, alpha, beta)
                if P is not None:
                    _update_best(P)
    
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for gamma in [0, 0.2, 0.6]:
            for delta in [0, 0.2, 0.6]:
                P = _build_family_B(c, gamma, delta)
                if P is not None:
                    _update_best(P)
    
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for cx in [3, 5, 7]:
            for cy in [3, 5, 7]:
                for sigma in [1.5, 3.0, 5.0]:
                    P = _build_family_C(c, 0, 0, cx, cy, sigma)
                    if P is not None:
                        _update_best(P)
    
    # Phase 2: Refine best candidates with fast gradient + scipy
    if time.time() - start_time < timeout - 60:
        if best_P is not None:
            P_clipped = np.clip(best_P, 1e-15, None)
            P_norm = P_clipped / P_clipped.sum()
            log_params = np.log(P_norm.flatten())
            
            # Fast gradient optimization first
            opt_params, opt_val = _fast_optimize(log_params, max_iter=5000, time_limit=15.0, soft_temp=0.05)
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            _update_best(opt_P)
            
            # Then scipy refinement
            if time.time() - start_time < timeout - 40:
                refined = _refine_scipy(best_P, timeout - (time.time() - start_time) - 20)
                if refined is not None:
                    _update_best(refined)
    
    # Phase 3: Multi-start with diverse initializations + fast gradient optimization
    if time.time() - start_time < timeout - 30:
        num_starts = 50
        remaining = timeout - (time.time() - start_time) - 15
        per_start = remaining / num_starts
        
        for s in range(num_starts):
            if time.time() - start_time > timeout - 15:
                break
            if per_start < 1.5:
                break
            
            strategy = s % 6
            
            if strategy == 0:
                c = np.random.randint(-9, 10)
                valid_cells = []
                for y in range(_N):
                    x = y + c
                    if 0 <= x < _N:
                        valid_cells.append((y, x))
                if len(valid_cells) == 0:
                    continue
                w = np.random.dirichlet(np.ones(len(valid_cells)))
                P = np.zeros((_N, _N))
                for (y, x), val in zip(valid_cells, w):
                    P[y, x] = val
                    
            elif strategy == 1:
                cx = np.random.randint(0, _N)
                cy = np.random.randint(0, _N)
                size = np.random.randint(2, 5)
                P = np.zeros((_N, _N))
                for y in range(max(0, cy - size), min(_N, cy + size)):
                    for x in range(max(0, cx - size), min(_N, cx + size)):
                        P[y, x] = 1.0
                P = _normalize(P)
                if P is None:
                    continue
                    
            elif strategy == 2:
                cx = np.random.uniform(2, 7)
                cy = np.random.uniform(2, 7)
                sigma = np.random.uniform(1, 3)
                P = np.zeros((_N, _N))
                for y in range(_N):
                    for x in range(_N):
                        P[y, x] = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
                P = _normalize(P)
                if P is None:
                    continue
                    
            elif strategy == 3:
                if best_P is not None:
                    P = best_P.copy()
                    P += np.random.normal(0, 0.05, P.shape)
                    P = _normalize(P)
                    if P is None:
                        continue
                else:
                    P = np.ones((_N, _N)) / (_N * _N)
                    
            elif strategy == 4:
                k = np.random.randint(3, 8)
                cells = np.random.choice(_N * _N, size=k, replace=False)
                P = np.zeros((_N, _N))
                for idx in cells:
                    P[idx // _N, idx % _N] = 1.0
                P = _normalize(P)
                if P is None:
                    continue
                    
            else:
                c = np.random.randint(-9, 10)
                valid_cells = []
                for y in range(_N):
                    x = y + c
                    if 0 <= x < _N:
                        valid_cells.append((y, x))
                if len(valid_cells) == 0:
                    continue
                n_valid = len(valid_cells)
                if n_valid > 0:
                    center = n_valid // 2
                    w = np.array([1.0 / (1 + abs(i - center)) for i in range(n_valid)])
                    w = w / w.sum()
                    P = np.zeros((_N, _N))
                    for (y, x), val in zip(valid_cells, w):
                        P[y, x] = val
                else:
                    continue
            
            P_clipped = np.clip(P, 1e-15, None)
            P_norm = P_clipped / P_clipped.sum()
            log_params = np.log(P_norm.flatten())
            
            # Fast gradient optimization with soft-max smoothing (two-stage temperature)
            opt_params, opt_val = _fast_optimize(log_params, max_iter=4000, time_limit=min(8, per_start * 0.6), soft_temp=0.08)
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            _update_best(opt_P)
            
            # Second stage: lower temperature for finer optimization
            if time.time() - start_time < timeout - 15:
                opt_params2, opt_val2 = _fast_optimize(opt_params, max_iter=3000, time_limit=min(5, per_start * 0.3), soft_temp=0.02)
                opt_P2 = _log_softmax_to_prob(opt_params2).reshape(_N, _N)
                _update_best(opt_P2)
                
                # Follow up with scipy if time permits
                if time.time() - start_time < timeout - 10:
                    try:
                        result = minimize(
                            _neg_ratio_smooth,
                            opt_params2,
                            args=(0.1,),
                            method='L-BFGS-B',
                            options={'maxiter': 600, 'ftol': 1e-14}
                        )
                        sp_P = _log_softmax_to_prob(result.x).reshape(_N, _N)
                        _update_best(sp_P)
                    except Exception:
                        pass
    
    # Phase 4: Additional multi-start focused on best region
    if time.time() - start_time < timeout - 30 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 15
        if remaining > 10:
            num_extra = min(20, int(remaining / 2.0))
            for s in range(num_extra):
                if time.time() - start_time > timeout - 15:
                    break
                # Perturb best solution and optimize
                P = best_P.copy()
                P += np.random.normal(0, 0.08, P.shape)
                P = _normalize(P)
                if P is None:
                    continue
                P_clipped = np.clip(P, 1e-15, None)
                P_norm = P_clipped / P_clipped.sum()
                log_params = np.log(P_norm.flatten())
                opt_params, opt_val = _fast_optimize(log_params, max_iter=3000, time_limit=min(5, remaining / (num_extra - s)), soft_temp=0.03)
                opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
                _update_best(opt_P)
    
    # Phase 5: Final fine-grained hill climbing with coordinate descent polish
    if time.time() - start_time < timeout - 15 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 13
        if remaining > 5:
            final = _hill_climb(best_P, time_limit=remaining)
            if final is not None:
                _update_best(final)
    
    # Phase 6: High-precision gradient polish with multiple temperatures
    if time.time() - start_time < timeout - 8 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 6
        if remaining > 3:
            P_clipped = np.clip(best_P, 1e-15, None)
            P_norm = P_clipped / P_clipped.sum()
            log_params = np.log(P_norm.flatten())
            try:
                # Stage 1: moderate temperature
                opt_params, opt_val = _fast_optimize(log_params, max_iter=10000, time_limit=remaining * 0.3, soft_temp=0.03)
                opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
                _update_best(opt_P)
                
                # Stage 2: low temperature for fine-tuning
                if time.time() - start_time < timeout - 10:
                    opt_params2, opt_val2 = _fast_optimize(opt_params, max_iter=10000, time_limit=remaining * 0.3, soft_temp=0.008)
                    opt_P2 = _log_softmax_to_prob(opt_params2).reshape(_N, _N)
                    _update_best(opt_P2)
                    
                    # Final scipy polish
                    if time.time() - start_time < timeout - 6:
                        try:
                            result = minimize(
                                _neg_ratio_smooth,
                                opt_params2,
                                args=(0.03,),
                                method='L-BFGS-B',
                                options={'maxiter': 3000, 'ftol': 1e-15}
                            )
                            sp_P = _log_softmax_to_prob(result.x).reshape(_N, _N)
                            _update_best(sp_P)
                        except Exception:
                            pass
            except Exception:
                pass
    
    # Phase 7: Final scipy polish with aggressive annealing
    if time.time() - start_time < timeout - 4 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 2
        if remaining > 3:
            try:
                refined = _refine_scipy(best_P, remaining)
                if refined is not None:
                    _update_best(refined)
            except Exception:
                pass
            
            if time.time() - start_time < timeout - 5:
                try:
                    P_perturbed = best_P.copy()
                    P_perturbed += np.random.normal(0, 0.02, P_perturbed.shape)
                    P_perturbed = _normalize(P_perturbed)
                    if P_perturbed is not None:
                        refined2 = _refine_scipy(P_perturbed, remaining * 0.4)
                        if refined2 is not None:
                            _update_best(refined2)
                except Exception:
                    pass
    
    # Phase 8: Final coordinate descent polish with smaller steps
    if time.time() - start_time < timeout - 2 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 1
        if remaining > 1:
            final = _coordinate_descent_polish(best_P, time_limit=remaining * 0.7)
            if final is not None:
                _update_best(final)
            
            if time.time() - start_time < timeout - 1:
                try:
                    # Fine random walk
                    P_fine = best_P.copy()
                    score = _score(P_fine.flatten())
                    for _ in range(100):
                        if time.time() - start_time > timeout - 0.5:
                            break
                        i, j = np.random.randint(0, _N, 2)
                        delta = np.random.uniform(-0.0005, 0.0005)
                        new_P = P_fine.copy()
                        new_P[i, j] += delta
                        new_P = np.clip(new_P, 0, None)
                        s = new_P.sum()
                        if s > 1e-9:
                            new_P = new_P / s
                            new_score = _score(new_P.flatten())
                            if new_score > score:
                                P_fine = new_P
                                score = new_score
                    _update_best(P_fine)
                except Exception:
                    pass
    
    # Ensure valid output
    if best_P is None:
        best_P = np.ones((_N, _N)) / (_N * _N)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N)) / (_N * _N)
    
    return best_P.astype(np.float64)
