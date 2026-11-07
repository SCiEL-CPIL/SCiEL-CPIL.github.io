import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _logsumexp(x, beta=1.0):
    """Compute log(sum(exp(x/beta))) * beta, which approximates max(x) as beta->0."""
    x_max = np.max(x)
    if x_max == -np.inf:
        return x_max
    shifted = x - x_max
    return x_max + beta * np.log(np.sum(np.exp(shifted / beta)))

def _smooth_objective(h, beta=0.1):
    """Smooth approximation of the max correlation using log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    ls = _logsumexp(corr, beta)
    return ls / N * 2.0

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(500):
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

def _objective(h):
    """Objective to minimize: the raw value."""
    return _correlation_max(h)

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Phase 1: Generate diverse starting points
    starts = [np.full(N, 0.5)]
    
    # Alternating patterns
    for base in [0.2, 0.3, 0.4, 0.45]:
        alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_feasible(alt))
    
    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]:
        for amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            t = np.arange(N)
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_cos))
    
    # Two-block patterns
    for split in [20, 30, 40, 50, 60, 70, 80]:
        h_block = np.zeros(N)
        h_block[:split] = 1.0
        h_block = _project_feasible(h_block)
        starts.append(h_block)
        h_block2 = np.zeros(N)
        h_block2[split:] = 1.0
        h_block2 = _project_feasible(h_block2)
        starts.append(h_block2)
    
    # Linear ramps
    starts.append(_project_feasible(np.linspace(0.0, 1.0, N)))
    starts.append(_project_feasible(np.linspace(1.0, 0.0, N)))
    
    # Random smooth seeds
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng.seed(seed)
        noise = rng.randn(N)
        kernel = np.ones(7) / 7.0
        smooth = np.convolve(noise, kernel, mode='same')
        h_pert = 0.5 + 0.4 * smooth
        starts.append(_project_feasible(h_pert))
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
    
    # Phase 2: Refine with SLSQP using smooth objective with annealing beta
    beta_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for beta in beta_schedule:
        if time.time() - start_time > DEADLINE - 15:
            break
        for h0 in starts:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                obj_smooth = lambda x, b=beta: _smooth_objective(x, b)
                res = minimize(
                    obj_smooth,
                    h0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
                    starts[0] = best_h.copy()
            except Exception:
                pass
    
    # Phase 3: Differential Evolution with smooth objective
    if time.time() - start_time < DEADLINE - 20:
        try:
            def de_objective(x):
                return _smooth_objective(_project_feasible(x), beta=0.1)
            
            de_bounds = [(0.0, 1.0)] * N
            result = differential_evolution(
                de_objective,
                bounds=de_bounds,
                seed=42,
                maxiter=50,
                tol=1e-8,
                polish=False,
                init='latinhypercube',
                popsize=15
            )
            h_de = _project_feasible(result.x)
            val_de = _correlation_max(h_de)
            if val_de < best_val - 1e-10:
                best_val = val_de
                best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish with exact non-smooth objective
    if time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False}
            )
            h_polish = _project_feasible(res.x)
            val_polish = _correlation_max(h_polish)
            if val_polish < best_val - 1e-12:
                best_val = val_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_feasible(best_h)
    
    return best_h
