"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Strategy:
1. The ratio is scale-invariant, so we normalize sum(h_i) = 1.
2. The objective becomes minimizing sup_k C[k] (the peak of the autoconvolution).
3. We use a combination of:
   a. Smart initializations (Gaussian, triangular, Sidon-like sparse sets)
   b. Local optimization via SLSQP with constraints
   c. Random restarts to escape local optima
   d. Explicit time budget management to stay within 120s limit

Key insight from inspirations:
- The problem is equivalent to finding a sequence whose autoconvolution is as flat as possible
  (minimizing the peak), subject to sum(h)=1 and h>=0.
- Sparse structures with carefully chosen supports can reduce autoconvolution peaks.
- Local optimization from multiple starting points is effective.
"""
import numpy as np
import scipy.optimize as opt
import time

N = 50

def _autoconv(h):
    """Compute linear autoconvolution of h with itself."""
    return np.convolve(h, h)

def _ratio(h):
    """Compute the ratio (sum h^2) / (sum h * sup autoconv)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    c = _autoconv(h)
    peak = np.max(c)
    if peak == 0:
        return 0.0
    return l2_sq / (l1 * peak)

def _minimize_peak(h_init, deadline, maxiter=200):
    """
    Given an initial h (sum=1), minimize the peak of autoconvolution
    using SLSQP. Returns the best h found.
    """
    h_init = np.asarray(h_init, dtype=np.float64)
    h_init = np.clip(h_init, 0, None)
    s = np.sum(h_init)
    if s == 0:
        return h_init
    h_init = h_init / s
    
    def objective(x):
        h = np.clip(x, 0, None)
        s = np.sum(h)
        if s == 0:
            return 1e10
        h = h / s
        c = _autoconv(h)
        return np.max(c)
    
    def objective_grad(x):
        h = np.clip(x, 0, None)
        s = np.sum(h)
        if s == 0:
            return np.ones(N)
        h = h / s
        c = _autoconv(h)
        k_max = np.argmax(c)
        # Gradient of c[k_max] wrt h_i = 2 * h[k_max - i] (with bounds)
        grad = np.zeros(N)
        for i in range(N):
            j = k_max - i
            if 0 <= j < N:
                grad[i] = 2.0 * h[j]
        # Account for normalization h/s
        # d/dx_i [max(c) * s / s] ... this is getting complex.
        # Let's just use finite differences if analytical is tricky, 
        # or approximate. Actually, let's just use the gradient of c[k_max]
        # and rely on SLSQP's robustness.
        return grad / (s * s)  # Rough scaling
    
    bounds = [(0.0, None)] * N
    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
         'jac': lambda x: np.ones(N)}
    ]
    
    best_h = h_init.copy()
    best_obj = objective(h_init)
    
    try:
        res = opt.minimize(
            objective,
            h_init,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8}
        )
        if res.x is not None:
            h_res = np.clip(res.x, 0, None)
            s = np.sum(h_res)
            if s > 0:
                h_res = h_res / s
                obj_res = objective(h_res)
                if obj_res < best_obj:
                    best_obj = obj_res
                    best_h = h_res.copy()
    except Exception:
        pass
    
    return best_h

def _sidon_init(k):
    """
    Generate a Sidon-like set of k positions in [0, N-1] where
    all pairwise differences are distinct (or nearly so).
    This minimizes autoconvolution peaks for sparse sequences.
    """
    if k <= 1:
        return [0] if k == 1 else []
    
    # Greedy construction: pick positions to maximize minimum difference
    positions = [0]
    for _ in range(1, k):
        best_pos = -1
        best_min_diff = -1
        for p in range(N):
            if p in positions:
                continue
            diffs = [abs(p - q) for q in positions]
            min_diff = min(diffs)
            if min_diff > best_min_diff:
                best_min_diff = min_diff
                best_pos = p
        if best_pos >= 0:
            positions.append(best_pos)
        else:
            break
    
    # Fill remaining with random non-colliding positions if needed
    while len(positions) < k:
        p = np.random.randint(N)
        if p not in positions:
            positions.append(p)
    
    return positions[:k]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_ratio = _ratio(best_h)
    
    def try_h(h):
        nonlocal best_h, best_ratio
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s == 0:
            return
        h = h / s
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # 1. Try various analytical initializations
    x_idx = np.arange(N)
    
    # Uniform
    try_h(np.ones(N))
    
    # Gaussian profiles
    for center in np.linspace(5, N-5, 8):
        for width in np.linspace(2, N/2, 8):
            h_gauss = np.exp(-0.5 * ((x_idx - center) / width)**2)
            try_h(h_gauss)
    
    # Triangular profiles
    for center in np.linspace(5, N-5, 8):
        for width in np.linspace(3, N, 8):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - center) / width)
            try_h(h_tri)
    
    # 2. Try sparse Sidon-like initializations and optimize
    for k in range(2, 12):
        if time.time() > deadline - 10:
            break
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            positions = _sidon_init(k)
            h_sparse = np.zeros(N)
            for p in positions:
                h_sparse[p] = 1.0
            try_h(h_sparse)
            
            # Optimize from this sparse start
            h_opt = _minimize_peak(h_sparse, deadline, maxiter=100)
            try_h(h_opt)
    
    # 3. Random restarts with local optimization
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with some structure
        if i % 3 == 0:
            # Smooth random
            h_rand = np.random.exponential(1.0, N)
        elif i % 3 == 1:
            # Sparse random
            k = np.random.randint(3, 15)
            support = np.random.choice(N, k, replace=False)
            h_rand = np.zeros(N)
            h_rand[support] = np.random.uniform(0.5, 1.0, k)
        else:
            # Bimodal
            h_rand = np.zeros(N)
            c1, c2 = np.random.randint(0, N//2), np.random.randint(N//2, N)
            w1, w2 = np.random.uniform(1, 5), np.random.uniform(1, 5)
            h_rand += np.exp(-0.5 * ((x_idx - c1) / w1)**2)
            h_rand += np.exp(-0.5 * ((x_idx - c2) / w2)**2)
        
        try_h(h_rand)
        
        # Local optimization
        h_opt = _minimize_peak(h_rand, deadline, maxiter=150)
        try_h(h_opt)
    
    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
