import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _log_score(h):
    """Log of the score for smoother optimization."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return -1e10
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max <= 0:
        return -1e10
    return np.log(l2_sq) - np.log(l1) - np.log(autoconv_max)

def _project(h):
    """Project onto nonnegative simplex (sum=1)."""
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
    return h

def _grad_log_score(h):
    """
    Gradient of log(score) w.r.t. h.
    score = l2_sq / (l1 * M) where M = delta * max(conv)
    log(score) = log(l2_sq) - log(l1) - log(M)
    
    d(log l2_sq)/dh_i = 2*h_i / l2_sq
    d(-log l1)/dh_i = -1/l1
    d(-log M)/dh_i = -dM/dh_i / M
    
    M = delta * max_k conv_k
    At the maximizing index k*, dM/dh_i = delta * d(conv_k*)/dh_i = delta * 2*h_{k*-i}
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 <= 0 or l2_sq <= 0:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    k_star = np.argmax(conv)
    M = DELTA * conv[k_star]
    if M <= 0:
        return np.zeros(N)
    
    grad = np.zeros(N)
    # d(log l2_sq)/dh_i = 2*h_i / l2_sq
    grad += 2.0 * h / l2_sq
    # d(-log l1)/dh_i = -1/l1
    grad -= 1.0 / l1
    # d(-log M)/dh_i = -dM/dh_i / M
    # dM/dh_i = DELTA * 2 * h_{k_star - i} (with bounds check)
    for i in range(N):
        j = k_star - i
        if 0 <= j < N:
            dM_di = DELTA * 2.0 * h[j]
        else:
            dM_di = 0.0
        grad[i] -= dM_di / M
    
    return grad

def _grad_log_score_vectorized(h):
    """Vectorized gradient of log(score)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    if l1 <= 0 or l2_sq <= 0:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    k_star = np.argmax(conv)
    M = DELTA * conv[k_star]
    if M <= 0:
        return np.zeros(N)
    
    grad = 2.0 * h / l2_sq - 1.0 / l1
    
    # dM/dh_i = DELTA * 2 * h_{k_star - i} for i where 0 <= k_star-i < N
    # Create a vector v where v[i] = h_{k_star - i} if valid, else 0
    v = np.zeros(N)
    # i ranges 0..N-1, k_star - i ranges k_star..k_star-(N-1)
    # Valid when 0 <= k_star - i < N, i.e., k_star-N+1 <= i <= k_star
    i_min = max(0, k_star - N + 1)
    i_max = min(N - 1, k_star)
    if i_min <= i_max:
        indices = np.arange(i_min, i_max + 1)
        v[indices] = h[k_star - indices]
    
    grad -= (DELTA * 2.0 * v) / M
    return grad

def _projected_grad_ascent(h, max_iter=5000, lr_init=0.01, deadline=None):
    """Projected gradient ascent on log-score with adaptive step size."""
    h = _project(h.copy())
    best_h = h.copy()
    best_val = _log_score(h)
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g = _grad_log_score_vectorized(h)
        
        # Adaptive step size: try to increase, backtrack if needed
        step = lr
        for _ in range(20):
            h_new = _project(h + step * g)
            val_new = _log_score(h_new)
            if val_new > best_val:
                best_h = h_new.copy()
                best_val = val_new
            if val_new > _log_score(h):
                h = h_new
                break
            step *= 0.5
        else:
            # No improvement found, reduce learning rate
            lr *= 0.9
            if lr < 1e-8:
                break
        # Slowly increase learning rate if making progress
        if it % 100 == 0 and lr < lr_init:
            lr = min(lr * 1.1, lr_init)
    
    return best_h, best_val

def _simulated_annealing(h_init, max_iter=2000, T_init=0.1, T_min=1e-6, deadline=None):
    """Simulated annealing on the simplex."""
    h = _project(h_init.copy())
    best_h = h.copy()
    best_val = _log_score(h)
    T = T_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation: add random noise and project
        noise = np.random.normal(0, T, N)
        h_new = _project(h + noise)
        
        val_new = _log_score(h_new)
        delta_val = val_new - best_val
        
        if delta_val > 0 or np.random.random() < np.exp(delta_val / T):
            h = h_new
        
        if val_new > best_val:
            best_val = val_new
            best_h = h_new.copy()
        
        # Cool down
        T = max(T_min, T_init * (T_min / T_init) ** (it / max_iter))
    
    return best_h, best_val

def _refine_with_sqp(h_init, deadline=None):
    """Use SLSQP for final refinement from a good starting point."""
    import scipy.optimize as opt
    
    def objective(x):
        return -_log_score(x)
    
    def objective_grad(x):
        return -_grad_log_score_vectorized(x)
    
    constraint_eq = {
        'type': 'eq',
        'fun': lambda x: np.sum(x) - 1.0,
        'jac': lambda x: np.ones(N)
    }
    
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = opt.minimize(
            objective,
            h_init,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint_eq],
            jac=objective_grad,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        if res.success:
            h_res = _project(res.x)
            val = _log_score(h_res)
            return h_res, val
    except Exception:
        pass
    
    return h_init, _log_score(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    np.random.seed(12345)
    
    best_h = np.ones(N) / N
    best_val = _log_score(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse random
    for trial in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 20)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
    
    # 3. Gaussian bumps
    for c in np.linspace(3, N - 3, 7):
        for w in np.linspace(0.5, N / 3, 7):
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w) ** 2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
    
    # 4. Triangular
    for c in np.linspace(3, N - 3, 7):
        for w in np.linspace(1, N, 7):
            h_t = np.maximum(0, 1 - np.abs(np.arange(N) - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
    
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.dirichlet(np.ones(N) * 2)
        inits.append(h_r)
    
    # 6. Sidon-like: positions with large pairwise differences
    for _ in range(10):
        h_sid = np.zeros(N)
        # Try to pick positions with good separation
        positions = []
        for _ in range(15):
            pos = np.random.randint(0, N)
            if all(abs(pos - p) >= 2 for p in positions):
                positions.append(pos)
        if len(positions) >= 3:
            h_sid[positions] = 1.0
            h_sid /= np.sum(h_sid)
            inits.append(h_sid)
    
    # Phase 1: Projected gradient ascent from each init
    for i, h_init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        h_ref, val = _projected_grad_ascent(h_init, max_iter=3000, lr_init=0.005, deadline=deadline - 30)
        if val > best_val:
            best_val = val
            best_h = h_ref
    
    # Phase 2: Simulated annealing from best and a few others
    sa_inits = [best_h] + inits[:10]
    for h_init in sa_inits:
        if time.time() > deadline - 20:
            break
        h_sa, val = _simulated_annealing(h_init, max_iter=3000, T_init=0.05, T_min=1e-7, deadline=deadline - 20)
        if val > best_val:
            best_val = val
            best_h = h_sa
    
    # Phase 3: Final SLSQP refinement
    if time.time() < deadline - 5:
        h_final, val = _refine_with_sqp(best_h, deadline=deadline - 2)
        if val > best_val:
            best_val = val
            best_h = h_final
    
    # Final projection
    best_h = _project(best_h)
    return list(best_h)
