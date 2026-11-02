import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _ratio(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-14):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution: full correlation
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def _grad_ratio(h):
    """Gradient of R = sum(h^2) / (sum(h) * max_k sum_i h_i h_{i+k})"""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    l2_sq = np.sum(h ** 2)
    
    # Compute full autoconvolution
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    idx_max = np.argmax(c)
    
    # R = l2_sq / (l1 * sup_c)
    # dR/dh_i = [2*h_i * l1 * sup_c - l2_sq * (sum(h) * d(sup_c)/dh_i + sup_c * 1)] / (l1 * sup_c)^2
    # = [2*h_i * sup_c - l2_sq * (d(sup_c)/dh_i + sup_c/l1)] / (l1 * sup_c)^2
    
    # sup_c = c[idx_max] = sum_j h_j * h_{j + (idx_max - N + 1)}
    # Let k = idx_max - N + 1 (the lag)
    k = idx_max - N + 1
    
    # c[k] = sum_j h_j * h_{j+k} where indices are valid
    # d(c[k])/dh_i = h_{i-k} + h_{i+k} (with boundary handling)
    dc_dh = np.zeros(N)
    for i in range(N):
        # h_i contributes to c[k] as h_i * h_{i+k} and h_{i-k} * h_i
        if 0 <= i + k < N:
            dc_dh[i] += h[i + k]
        if 0 <= i - k < N:
            dc_dh[i] += h[i - k]
    
    num = 2 * h * sup_c - l2_sq * (dc_dh + sup_c / l1)
    den = (l1 * sup_c) ** 2
    
    grad = num / den
    return grad

def _optimize_from_start(h0, deadline):
    """Optimize starting from h0 using gradient ascent on log(R) or direct R."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    # Use L-BFGS-B with bounds
    def neg_log_ratio(params):
        x = np.maximum(params, 0.0)
        r = _ratio(x)
        if r < 1e-30:
            return 1e10
        return -np.log(r)
    
    def neg_log_ratio_grad(params):
        x = np.maximum(params, 0.0)
        r = _ratio(x)
        if r < 1e-30:
            return np.zeros_like(params)
        g = _grad_ratio(x)
        # d(-log R)/dh = -g/R
        return -g / r
    
    start_time = time.time()
    
    for attempt in range(3):
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(
                neg_log_ratio,
                h,
                jac=neg_log_ratio_grad,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = np.maximum(res.x, 0.0)
            r_new = _ratio(h_new)
            if r_new > best_r:
                best_r = r_new
                best_h = h_new.copy()
            h = h_new.copy()
        except Exception:
            break
    
    return best_h, best_r

def _coord_descent(h, deadline, max_iters=500):
    """Coordinate descent: optimize each coordinate in turn."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    best_r = _ratio(h)
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() > deadline - 3.0:
            break
        improved = False
        for i in range(N):
            # Optimize h[i] with others fixed
            # R(h) = (sum_j h_j^2) / (sum_j h_j * sup_c)
            # For fixed other coords, this is a 1D problem in h_i
            h_old_i = h[i]
            h_no_i = h.copy()
            h_no_i[i] = 0.0
            
            l1_other = np.sum(h_no_i)
            l2_sq_other = np.sum(h_no_i ** 2)
            
            # c = autoconv of h_no_i
            c_base = np.correlate(h_no_i, h_no_i, mode='full')
            
            # h = h_no_i + h_i * e_i
            # l1 = l1_other + h_i
            # l2_sq = l2_sq_other + h_i^2
            # c[k] = c_base[k] + h_i * (h_no_i[i-k] + h_no_i[i+k]) + h_i^2 * delta_{k,0}
            # Actually c[k] for full correlation:
            # c_full[k] = sum_j h_j h_{j+k-N+1}... let me just compute directly
            
            # For efficiency, compute the effect of h_i on each lag
            # h = h_no_i + x * e_i where x = h_i
            # l1(x) = l1_other + x
            # l2_sq(x) = l2_sq_other + x^2
            # c(x)[k] = c_base[k] + x*(h_no_i[i+k'] + h_no_i[i-k']) for appropriate k'
            # The lag-0 term: c(0) includes x^2
            # Let's just do a small grid search for x in [0, 2*h_old_i + 0.1]
            
            x_vals = np.linspace(0, max(2 * h_old_i + 0.05, 0.05), 30)
            best_x = h_old_i
            best_local_r = -1
            
            for x in x_vals:
                h_test = h_no_i.copy()
                h_test[i] = x
                r_test = _ratio(h_test)
                if r_test > best_local_r:
                    best_local_r = r_test
                    best_x = x
            
            if abs(best_x - h[i]) > 1e-8:
                h[i] = best_x
                improved = True
            r_now = _ratio(h)
            if r_now > best_r:
                best_r = r_now
        
        if not improved:
            break
    
    return h, best_r

def _random_start_with_structure(rng, n_spikes=None):
    """Generate random nonnegative vector with some structure."""
    if n_spikes is None:
        n_spikes = rng.integers(1, N)
    
    h = np.zeros(N)
    indices = rng.choice(N, size=n_spikes, replace=False)
    for idx in indices:
        h[idx] = rng.uniform(0.1, 1.0)
    
    # Normalize so sum is around 1
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h

def _sidon_like_start(rng):
    """Create a start inspired by Sidon sets - sparse with good separation."""
    h = np.zeros(N)
    # Pick a small Sidon-like set
    # For simplicity, use primes or quadratic residues mod N
    n_pts = rng.integers(3, 10)
    indices = sorted(rng.choice(N, size=n_pts, replace=False))
    for idx in indices:
        h[idx] = rng.uniform(0.5, 1.0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_r = _ratio(best_h)
    
    # Generate diverse starting points
    starts = []
    
    # Spike
    h = np.zeros(N); h[0] = 1.0; starts.append(h.copy())
    
    # Two spikes
    h = np.zeros(N); h[0] = 0.5; h[25] = 0.5; starts.append(h.copy())
    
    # Uniform
    h = np.ones(N) / N; starts.append(h.copy())
    
    # Random sparse
    for _ in range(15):
        starts.append(_random_start_with_structure(rng))
    
    # Sidon-like
    for _ in range(10):
        starts.append(_sidon_like_start(rng))
    
    # Random dense
    for _ in range(10):
        h = rng.uniform(0, 1, N)
        h = h / np.sum(h)
        starts.append(h)
    
    # Cosine-based
    for _ in range(5):
        t = np.linspace(0, 2 * np.pi, N)
        h = np.maximum(np.cos(t + rng.uniform(0, 2 * np.pi)), 0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        starts.append(h)
    
    # Exponential decay
    for _ in range(5):
        rate = rng.uniform(0.5, 3.0)
        h = np.exp(-rate * np.arange(N) / N)
        h = h / np.sum(h)
        starts.append(h)
    
    # Triangular
    for _ in range(3):
        h = np.maximum(1 - np.abs(np.arange(N) - N/2) / (N/2), 0)
        h = h / np.sum(h)
        starts.append(h)
    
    # Process each start
    for i, h0 in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        # Gradient optimization
        h_opt, r_opt = _optimize_from_start(h0, deadline)
        if r_opt > best_r:
            best_r = r_opt
            best_h = h_opt.copy()
        
        # Coordinate descent from the gradient result
        h_cd, r_cd = _coord_descent(h_opt, deadline, max_iters=100)
        if r_cd > best_r:
            best_r = r_cd
            best_h = h_cd.copy()
        
        # Also try coordinate descent from original
        if time.time() < deadline - 15.0:
            h_cd2, r_cd2 = _coord_descent(h0, deadline, max_iters=50)
            if r_cd2 > best_r:
                best_r = r_cd2
                best_h = h_cd2.copy()
    
    # Final polish: more coordinate descent iterations on best
    if time.time() < deadline - 5.0:
        h_final, r_final = _coord_descent(best_h, deadline, max_iters=200)
        if r_final > best_r:
            best_r = r_final
            best_h = h_final.copy()
    
    # Ensure nonnegativity and valid
    best_h = np.maximum(best_h, 0.0)
    
    return list(best_h)
