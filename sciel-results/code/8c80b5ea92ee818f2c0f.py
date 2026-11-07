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

def _soft_max_objective_and_grad(h, beta=100.0):
    """
    Smooth approximation of max correlation with gradient.
    Returns (objective, gradient).
    Objective: (1/beta) * log(sum(exp(beta * C_k)))
    """
    corr = np.correlate(h, 1.0 - h, mode='full')
    c_max = np.max(corr)
    if c_max == 0:
        return 0.0, np.zeros_like(h)
    
    scaled = beta * (corr - c_max)
    scaled = np.clip(scaled, -100, 0)
    exp_vals = np.exp(scaled)
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return 0.0, np.zeros_like(h)
    
    log_sum = np.log(sum_exp)
    obj = (c_max + log_sum / beta)
    
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
    
    return obj, grad

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
        
        obj, grad = _soft_max_objective_and_grad(h, beta=beta)
        
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

def _construct_diverse_seeds(rng):
    """Generate a set of diverse starting points."""
    seeds = []
    x = np.arange(N)

    # Uniform
    seeds.append(np.full(N, 0.5))

    # Alternating low-high
    seeds.append(np.where(x % 2 == 0, 0.2, 0.8).copy())

    # Sine waves of various frequencies
    for freq in range(1, 11):
        for amp in [0.3, 0.45, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            seeds.append(_project_feasible(h))
            h = 0.5 + amp * np.cos(2 * np.pi * freq * x / N)
            seeds.append(_project_feasible(h))

    # Block patterns
    for block_size in [10, 20, 25, 33, 50]:
        h = np.zeros(N)
        positions = np.arange(0, N, block_size * 2)
        for pos in positions:
            end = min(pos + block_size, N)
            h[pos:end] = 1.0
        h = _project_feasible(h)
        seeds.append(h)

    # Random smooth (low-pass filtered)
    for _ in range(15):
        noise = rng.randn(N)
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        h = 0.5 + 0.4 * smooth
        seeds.append(_project_feasible(h))

    # Random binary-like (many 0s and 1s)
    for _ in range(10):
        h = rng.choice([0.0, 1.0], size=N, p=[0.5, 0.5])
        seeds.append(_project_feasible(h))

    # Sawtooth
    for period in [10, 20, 25, 50]:
        sawtooth = np.linspace(0, 1, period, endpoint=False)
        h = np.tile(sawtooth, N // period + 1)[:N]
        seeds.append(_project_feasible(h))

    # Piecewise constant
    for n_segments in [4, 5, 10, 20]:
        seg_len = N // n_segments
        vals = rng.uniform(0.2, 0.8, n_segments)
        h = np.repeat(vals, seg_len + (1 if seg_len * n_segments < N else 0))[:N]
        seeds.append(_project_feasible(h))

    # Low-autocorrelation inspired: use m-sequence-like patterns
    for _ in range(8):
        seq = np.zeros(N)
        state = rng.randint(1, 16)
        for i in range(N):
            seq[i] = 0.5 + 0.4 * ((state & 1) * 2 - 1)
            lsb = state & 1
            msb = (state >> 3) & 1
            state = (state >> 1) | ((lsb ^ msb) << 3)
        seeds.append(_project_feasible(seq))

    # Quadratic phase (chirp-like)
    for a in [0.5, 1.0, 2.0, 3.0]:
        h = 0.5 + 0.4 * np.sin(2 * np.pi * a * x**2 / N)
        seeds.append(_project_feasible(h))

    return seeds

def _parametric_search(deadline, rng):
    """
    Search in a low-dimensional parametric space:
    h(x) = 0.5 + sum_{k=1}^{K} a_k * sin(2*pi*f_k*x/N + phi_k)
    K=3 sinusoids, 9 parameters total.
    """
    x = np.arange(N)
    best_h = None
    best_val = float('inf')
    
    # Parameter bounds
    amp_bounds = [(0.0, 0.49)] * 3
    freq_bounds = [(0.5, 10.0)] * 3
    phase_bounds = [(-np.pi, np.pi)] * 3
    
    def _eval_params(params):
        amps = params[:3]
        freqs = params[3:6]
        phases = params[6:9]
        h = np.full(N, 0.5)
        for k in range(3):
            h += amps[k] * np.sin(2 * np.pi * freqs[k] * x / N + phases[k])
        h = _project_feasible(h)
        val = _correlation_max(h)
        return val, h
    
    # Try several random initializations in parametric space
    n_tries = 20
    for t in range(n_tries):
        if time.time() > deadline - 10:
            break
        
        # Random initial parameters
        amps = rng.uniform(0.1, 0.45, 3)
        freqs = rng.uniform(1.0, 8.0, 3)
        phases = rng.uniform(-np.pi, np.pi, 3)
        x0 = np.concatenate([amps, freqs, phases])
        
        bounds = list(zip(
            [0.0]*3 + [0.5]*3 + [-np.pi]*3,
            [0.49]*3 + [10.0]*3 + [np.pi]*3
        ))
        
        def obj(params, _x=x):
            amps = params[:3]
            freqs = params[3:6]
            phases = params[6:9]
            h = np.full(N, 0.5)
            for k in range(3):
                h += amps[k] * np.sin(2 * np.pi * freqs[k] * _x / N + phases[k])
            h = _project_feasible(h)
            return _correlation_max(h)
        
        try:
            res = minimize(
                obj, x0,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            val, h_res = _eval_params(res.x)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    return best_h, best_val

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
        obj = lambda x, b=beta: _soft_max_objective_and_grad(x, beta=b)[0]
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin (150s limit, 5s buffer)

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    rng = np.random.RandomState(42)

    # Phase 1: Parametric low-dimensional search (9-dim)
    if time.time() < deadline - 30:
        h_param, val_param = _parametric_search(deadline, rng)
        if h_param is not None and val_param < best_val - 1e-10:
            best_val = val_param
            best_h = h_param.copy()

    # Phase 2: Generate diverse seeds and polish with SLSQP
    seeds = _construct_diverse_seeds(rng)

    for h0 in seeds:
        if time.time() > deadline - 25:
            break
        h_polished = _polish_with_slsqp(h0, deadline)
        val = _correlation_max(h_polished)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h_polished.copy()

    # Phase 3: Local gradient descent on best candidate
    if time.time() < deadline - 20:
        h_gd = _local_gradient_descent(best_h, deadline, max_iters=100)
        val = _correlation_max(h_gd)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h_gd.copy()

    # Phase 4: Local search with small perturbations + SLSQP polish
    if time.time() < deadline - 15:
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            scale = 0.05 + 0.1 * (1 - trial / 30.0)
            perturbation = rng.randn(N) * scale
            perturbation = np.convolve(perturbation, np.ones(3)/3.0, mode='same')
            h_try = best_h + perturbation
            h_try = _project_feasible(h_try)
            h_polished = _polish_with_slsqp(h_try, deadline)
            val = _correlation_max(h_polished)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_polished.copy()
            if time.time() < deadline - 10:
                h_gd = _local_gradient_descent(h_try, deadline, max_iters=50)
                val_gd = _correlation_max(h_gd)
                if val_gd < best_val - 1e-10:
                    best_val = val_gd
                    best_h = h_gd.copy()

    # Phase 5: High-precision SLSQP polish
    if time.time() < deadline - 5:
        h_final = _polish_with_slsqp(best_h, deadline)
        val = _correlation_max(h_final)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_final.copy()

    # Phase 6: Final gradient descent
    if time.time() < deadline - 2:
        h_gd_final = _local_gradient_descent(best_h, deadline, max_iters=100)
        val = _correlation_max(h_gd_final)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_gd_final.copy()

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
