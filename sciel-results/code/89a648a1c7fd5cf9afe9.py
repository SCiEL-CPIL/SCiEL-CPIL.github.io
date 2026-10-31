"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _overlap_and_grad_h(h, alpha=200.0):
    """Return overlap vector and smooth-max weights."""
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    w = _smooth_max_grad(overlap, alpha)
    return overlap, w

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1 using bisection."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    # Use a more robust projection: bisection on scaling + clipping
    for _ in range(60):
        s = h.sum()
        if abs(s - target) < 1e-15:
            break
        if s > target:
            # Scale down: h = h * (target/s), then clip, iterate
            h = h * (target / max(s, 1e-15))
            h = np.clip(h, 0.0, 1.0)
        else:
            # Scale up
            h = h * (target / max(s, 1e-15))
            h = np.clip(h, 0.0, 1.0)
    # Final exact fix: distribute residual among non-bound entries
    for _ in range(30):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-14
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 1e-14
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _three_block_init():
    """Three-block initial guess: A, B, A with optimal widths."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    # Try different split points
    best_h = None
    best_score = float('inf')
    for i in range(10, N-10):
        for j in range(i+5, N-5):
            h0 = np.zeros(N)
            h0[:i] = A
            h0[i:j] = B
            h0[j:] = A
            h0 = _project_to_feasible(h0)
            sc = _score(h0)
            if sc < best_score:
                best_score = sc
                best_h = h0.copy()
    return best_h

def _optimize_from(h_start, alpha, maxiter, ftol, deadline):
    """Run SLSQP epigraph optimization from h_start. Returns (h, score) or None."""
    current = _score(h_start)
    def objective(params):
        h = params[:N]
        t = params[N]
        overlap, _ = _overlap_and_grad_h(h, alpha)
        smax = _smooth_max(overlap, alpha)
        return t + 1000.0 * np.maximum(0.0, smax - t)**2

    def constraint_sum(params):
        return np.sum(params[:N]) - N/2.0

    def constraint_epigraph(params):
        h = params[:N]
        t = params[N]
        overlap, _ = _overlap_and_grad_h(h, alpha)
        return t - np.max(overlap)

    x0 = np.concatenate([h_start, [current + 0.05]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_epigraph}
    ]
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds_ext, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        if res.fun < 100.0:
            h_opt = _project_to_feasible(res.x[:N])
            sc = _score(h_opt)
            if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                return h_opt, sc
    except Exception:
        pass
    return None

def _optimize_lbfgsb(h_start, alpha, maxiter, ftol, deadline):
    """Run L-BFGS-B with penalty method. Returns (h, score) or None."""
    current = _score(h_start)
    
    def objective(params):
        h = params
        overlap = np.zeros(N)
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
        smax = _smooth_max(overlap, alpha)
        # Penalty for sum constraint
        sum_pen = 100.0 * (np.sum(h) - N/2.0)**2
        return smax + sum_pen

    x0 = h_start.copy()
    bounds = [(0.0, 1.0)] * N
    try:
        res = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        if res.fun < 100.0:
            h_opt = _project_to_feasible(res.x)
            sc = _score(h_opt)
            if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                return h_opt, sc
    except Exception:
        pass
    return None

def _local_search_polish(h_start, max_iter=5000, deadline=None):
    """Local search with adaptive step sizes."""
    h = h_start.copy()
    best_score = _score(h)
    rng = np.random.default_rng(123)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Adaptive step size
        base_step = 0.03 * (1.0 - iteration / max_iter)**0.5
        step_size = max(0.001, base_step)
        
        # Try different numbers of perturbations
        n_perturb = int(rng.integers(1, 4))
        idxs = rng.choice(N, size=n_perturb, replace=False)
        
        h_trial = h.copy()
        for i in idxs:
            h_trial[i] = np.clip(h_trial[i] + rng.uniform(-step_size, step_size), 0.0, 1.0)
        h_trial = _project_to_feasible(h_trial)
        sc = _score(h_trial)
        
        if sc < best_score - 1e-14:
            best_score = sc
            h = h_trial.copy()
    
    return h, best_score

def _coordinate_descent(h_start, max_iter=1000, deadline=None):
    """Coordinate descent: optimize each coordinate one at a time."""
    h = h_start.copy()
    best_score = _score(h)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Try small perturbations of coordinate i
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _project_to_feasible(h_trial)
                sc = _score(h_trial)
                if sc < best_score - 1e-14:
                    best_score = sc
                    h = h_trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return h, best_score

def construct():
    deadline = time.time() + 130.0  # Safety margin before 150s kill

    # Start with 2-block solution
    h_final = _two_block_init()
    current_score = _score(h_final)
    
    # Phase 1: SLSQP from 2-block, moderate alpha
    res = _optimize_from(h_final, alpha=200.0, maxiter=2000, ftol=1e-14, deadline=deadline)
    if res and res[1] < current_score:
        h_final, current_score = res

    # Phase 2: L-BFGS-B polish
    if time.time() < deadline - 25.0:
        res = _optimize_lbfgsb(h_final, alpha=300.0, maxiter=3000, ftol=1e-16, deadline=deadline)
        if res and res[1] < current_score:
            h_final, current_score = res

    # Phase 3: Three-block initialization
    if time.time() < deadline - 40.0:
        h3 = _three_block_init()
        if h3 is not None:
            sc3 = _score(h3)
            if sc3 < current_score:
                h_final = h3
                current_score = sc3
            # Optimize from 3-block
            res = _optimize_from(h3, alpha=250.0, maxiter=2000, ftol=1e-14, deadline=deadline)
            if res and res[1] < current_score:
                h_final, current_score = res

    # Phase 4: High-precision polish with tighter alpha
    if time.time() < deadline - 20.0:
        res = _optimize_from(h_final, alpha=600.0, maxiter=3000, ftol=1e-16, deadline=deadline)
        if res and res[1] < current_score:
            h_final, current_score = res

    # Phase 5: Diverse structured starts
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(7)
        starts = []
        # Sinusoidal patterns
        for freq in [1, 2, 3, 4, 5]:
            for phase in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3]:
                x = np.arange(N)
                h_init = 0.5 + 0.45 * np.sin(2*np.pi*freq*x/N + phase)
                h_init = _project_to_feasible(h_init)
                starts.append(h_init)
        # Random smooth
        for _ in range(6):
            h_init = np.clip(0.5 + rng.normal(0, 0.15, N), 0.0, 1.0)
            h_init = _project_to_feasible(h_init)
            starts.append(h_init)
        # Alternating-ish
        for amp in [0.25, 0.35, 0.45, 0.55]:
            h_init = np.full(N, 0.5)
            h_init[0::2] = min(1.0, 0.5 + amp)
            h_init[1::2] = max(0.0, 0.5 - amp)
            h_init = _project_to_feasible(h_init)
            starts.append(h_init)
        # Multi-block
        for nblocks in [3, 4, 5]:
            for _ in range(3):
                h_init = np.zeros(N)
                block_width = N // nblocks
                for k in range(nblocks):
                    val = 0.5 + 0.4 * math.sin(math.pi * k / (nblocks - 1))
                    start = k * block_width
                    end = (k+1) * block_width if k < nblocks-1 else N
                    h_init[start:end] = val
                h_init = _project_to_feasible(h_init)
                starts.append(h_init)

        for h_init in starts:
            if time.time() > deadline - 15.0:
                break
            res = _optimize_from(h_init, alpha=300.0, maxiter=1500, ftol=1e-13, deadline=deadline)
            if res and res[1] < current_score:
                h_final, current_score = res

    # Phase 6: Local search refinement
    if time.time() < deadline - 15.0:
        h_polished, sc_polished = _local_search_polish(h_final, max_iter=4000, deadline=deadline - 5.0)
        if sc_polished < current_score:
            h_final, current_score = h_polished

    # Phase 7: Coordinate descent
    if time.time() < deadline - 8.0:
        h_cd, sc_cd = _coordinate_descent(h_final, max_iter=500, deadline=deadline - 3.0)
        if sc_cd < current_score:
            h_final, current_score = h_cd

    # Final validation
    h_final = _project_to_feasible(h_final)
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)

    return h_final
