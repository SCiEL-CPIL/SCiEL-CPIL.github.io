import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0
TOL = 1e-9 * N  # 1e-7

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(300):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _poisson_solve(sources, n_sources=5):
    """
    Solve discrete Poisson equation: -Laplacian(h) = sources
    with periodic boundary conditions.
    sources: array of (position, amplitude) pairs
    Returns h after Poisson solve.
    """
    # Build the source vector
    f = np.zeros(N)
    for pos, amp in sources:
        idx = int(round(pos)) % N
        f[idx] += amp
    
    # Discrete Laplacian with periodic BC: h[i-1] - 2*h[i] + h[i+1]
    # We want -Laplacian(h) = f, i.e., 2*h[i] - h[i-1] - h[i+1] = f[i]
    # This is a circulant matrix. Solve via FFT.
    # The eigenvalues of the discrete Laplacian are: 2 - 2*cos(2*pi*k/N)
    # So we need to solve: (2 - 2*cos(2*pi*k/N)) * H[k] = F[k]
    F = np.fft.rfft(f)
    k = np.arange(N // 2 + 1)
    eigvals = 2.0 - 2.0 * np.cos(2.0 * np.pi * k / N)
    # Avoid division by zero for k=0 (DC component)
    eigvals[0] = 1.0  # DC component will be handled by projection
    H = F / eigvals
    h = np.fft.irfft(H, n=N)
    
    # Remove mean (DC component) and set to TARGET_SUM/N
    h -= np.mean(h)
    h += TARGET_SUM / N
    
    return h

def _poisson_to_heights(params, n_sources):
    """
    Convert optimization parameters to height sequence via Poisson solve.
    params: [pos1, amp1, pos2, amp2, ..., pos_n, amp_n]
    Returns feasible height sequence.
    """
    sources = []
    for i in range(n_sources):
        pos = params[2*i]
        amp = params[2*i+1]
        sources.append((pos, amp))
    
    h = _poisson_solve(sources, n_sources)
    h = _project_feasible(h)
    return h

def _poisson_objective(params, n_sources, deadline):
    """Objective function for Poisson-based optimization."""
    if time.time() > deadline - 5:
        return 1e10
    h = _poisson_to_heights(params, n_sources)
    return _correlation_max(h)

def _poisson_search(deadline, rng, n_sources=5, n_tries=30):
    """
    Search in Poisson source parameter space.
    Each source has a position (0 to N) and amplitude (-1 to 1).
    """
    best_h = None
    best_val = float('inf')
    
    for t in range(n_tries):
        if time.time() > deadline - 10:
            break
        
        # Random initial parameters
        params = np.zeros(2 * n_sources)
        for i in range(n_sources):
            params[2*i] = rng.uniform(0, N)  # position
            params[2*i+1] = rng.uniform(-0.5, 0.5)  # amplitude
        
        bounds = []
        for i in range(n_sources):
            bounds.append((0, N))  # position
            bounds.append((-1.0, 1.0))  # amplitude
        
        def obj(params):
            return _poisson_objective(params, n_sources, deadline)
        
        try:
            res = minimize(
                obj, params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            h_res = _poisson_to_heights(res.x, n_sources)
            val = _correlation_max(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    return best_h, best_val

def _soft_max_objective(h, beta=100.0):
    """Smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    c_max = np.max(corr)
    if c_max == 0:
        return 0.0
    scaled = beta * (corr - c_max)
    scaled = np.clip(scaled, -100, 0)
    exp_vals = np.exp(scaled)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0
    log_sum = np.log(sum_exp)
    return (c_max + log_sum / beta) / N * 2.0

def _polish_with_slsqp(h0, deadline):
    """Polish a candidate with SLSQP using soft-max objective."""
    bounds = [(0.0, 1.0)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
         'jac': lambda x: np.ones(N)}
    ]

    best_h = _project_feasible(h0)
    best_val = _correlation_max(best_h)

    for beta in [50.0, 100.0, 200.0, 400.0, 800.0]:
        if time.time() > deadline - 5:
            break
        obj = lambda x, b=beta: _soft_max_objective(x, beta=b)
        try:
            res = minimize(
                obj, best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res
        except Exception:
            pass
    return best_h

def _local_gradient_descent(h0, deadline, max_iters=200):
    """Local gradient descent on the soft-max surrogate with adaptive step size."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _correlation_max(h)
    
    beta = 100.0
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Compute gradient of soft-max objective
        corr = np.correlate(h, 1.0 - h, mode='full')
        c_max = np.max(corr)
        if c_max == 0:
            break
        
        scaled = beta * (corr - c_max)
        scaled = np.clip(scaled, -100, 0)
        exp_vals = np.exp(scaled)
        sum_exp = np.sum(exp_vals)
        if sum_exp == 0:
            break
        
        weights = exp_vals / sum_exp
        grad = np.zeros(N)
        
        for idx, l in enumerate(range(-(N-1), N)):
            w = weights[idx]
            if w < 1e-15:
                continue
            
            if l >= 0:
                if N - 1 - l >= 0:
                    grad[:N-l] -= w * h[l:]
                if l <= N - 1:
                    grad[l:] -= w * h[:N-l]
            else:
                m = -l
                if m <= N - 1:
                    grad[:N-m] -= w * h[m:]
                if N - 1 - m >= 0:
                    grad[m:] -= w * h[:N-m]
                    
        grad += 1.0
        grad /= beta
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        step = lr
        for _ in range(20):
            h_try = h - step * grad
            h_try = _project_feasible(h_try)
            val_try = _correlation_max(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
                h = h_try
                break
            step *= 0.5
        
        if step < 1e-8:
            break
        
        if it % 50 == 49:
            beta *= 1.5
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin (150s limit, 5s buffer)

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    rng = np.random.RandomState(42)

    # Phase 1: Poisson-based low-dimensional search (NEW STRUCTURAL APPROACH)
    # Try different numbers of sources
    for n_sources in [3, 5, 7, 10]:
        if time.time() > deadline - 40:
            break
        h_poisson, val_poisson = _poisson_search(deadline, rng, n_sources=n_sources, n_tries=15)
        if h_poisson is not None and val_poisson < best_val - 1e-10:
            best_val = val_poisson
            best_h = h_poisson.copy()
    
    # Phase 2: Polish best candidate with SLSQP
    if time.time() < deadline - 20:
        h_polished = _polish_with_slsqp(best_h, deadline)
        val = _correlation_max(h_polished)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h_polished.copy()
    
    # Phase 3: Local gradient descent
    if time.time() < deadline - 15:
        h_gd = _local_gradient_descent(best_h, deadline, max_iters=100)
        val = _correlation_max(h_gd)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h_gd.copy()
    
    # Phase 4: More Poisson searches with different seeds
    if time.time() < deadline - 10:
        rng2 = np.random.RandomState(123)
        for n_sources in [5, 8]:
            if time.time() > deadline - 10:
                break
            h_poisson, val_poisson = _poisson_search(deadline, rng2, n_sources=n_sources, n_tries=10)
            if h_poisson is not None and val_poisson < best_val - 1e-10:
                best_val = val_poisson
                best_h = h_poisson.copy()
            if time.time() < deadline - 8:
                h_polished = _polish_with_slsqp(best_h, deadline)
                val = _correlation_max(h_polished)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_polished.copy()
    
    # Phase 5: Final SLSQP polish
    if time.time() < deadline - 5:
        h_final = _polish_with_slsqp(best_h, deadline)
        val = _correlation_max(h_final)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_final.copy()
    
    # Final: ensure validity
    best_h = _project_feasible(best_h)
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-10:
        mask = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
        n_free = np.sum(mask)
        if n_free > 0:
            best_h[mask] += diff / n_free
        best_h = np.clip(best_h, 0.0, 1.0)
        best_h = _project_feasible(best_h)

    return best_h
