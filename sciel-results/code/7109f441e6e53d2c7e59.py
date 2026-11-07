"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0

def _smooth_max_corr(h, beta=80.0):
    """Smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta

def _project_to_feasible(h):
    """Project h onto feasible set: sum=N/2, 0<=h<=1."""
    h = np.array(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _compute_grad_smooth_max(h, beta=80.0):
    """Compute gradient of smooth max correlation w.r.t. h analytically."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    weights = np.exp((corr - m) * beta)
    weights /= np.sum(weights)
    
    # corr[k] = sum_j h[j] * (1 - h[j + k - (N-1)]) for k in 0..2N-2
    # d corr[k] / d h[i] = (1 - h[i + k - (N-1)]) if i + k - (N-1) is valid, 
    #                      plus h[i - (k - (N-1))] if valid, 
    #                      minus h[i]*(1-h[i]) if i == i + k - (N-1) (i.e., k = N-1)
    # Actually let's think more carefully.
    # np.correlate(h, g, mode="full") where g = 1-h
    # corr[k] = sum_j h[j] * g[j + k - (N-1)]  ... this is one convention
    # Actually numpy's correlate: corr[k] = sum_j h[j] * g[j - (k - (N-1))]
    # Let's use the definition: corr[k] = sum_j h[j] * g[j + k - (N-1)]
    # where g = 1 - h.
    # d corr[k] / d h[i] = g[i + k - (N-1)] + h[i] * d g[i + k - (N-1)] / d h[i]
    #                     + sum over j where j + k - (N-1) == i of h[j] * d g[i]/d h[i]
    # d g[m] / d h[i] = -delta_{m,i}
    # So d corr[k] / d h[i] = g[i + k - (N-1)] - h[i] * delta_{i + k - (N-1), i} 
    #                        - sum_j h[j] * delta_{j + k - (N-1), i}
    # The delta_{i+k-(N-1), i} is 1 when k = N-1.
    # The sum term: j + k - (N-1) = i => j = i - k + (N-1).
    # So d corr[k] / d h[i] = (1 - h[i + k - (N-1)]) if 0 <= i + k - (N-1) < N
    #                        - h[i] if k == N-1
    #                        - h[i - k + (N-1)] if 0 <= i - k + (N-1) < N
    
    grad = np.zeros(N)
    for k in range(2 * N - 1):
        w = weights[k]
        for i in range(N):
            idx1 = i + k - (N - 1)
            if 0 <= idx1 < N:
                grad[i] += w * (1 - h[idx1])
            if k == N - 1:
                grad[i] -= w * h[i]
            idx2 = i - k + (N - 1)
            if 0 <= idx2 < N:
                grad[i] -= w * h[idx2]
    return grad

def _local_polish(h, max_iters=500, tol=1e-14, beta=80.0):
    """Local polish using gradient descent on smooth surrogate."""
    h = np.array(h, dtype=float).copy()
    h = _project_to_feasible(h)
    best_val = _correlation_value(h)
    
    for it in range(max_iters):
        grad = _compute_grad_smooth_max(h, beta)
        # Project gradient onto feasible directions (sum = 0, within bounds)
        # Simple approach: subtract mean to make sum zero
        grad -= np.mean(grad)
        
        # Adaptive learning rate
        lr = 0.1 / (np.max(np.abs(grad)) + 1e-8)
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        val = _correlation_value(h_new)
        if val < best_val - tol:
            h = h_new.copy()
            best_val = val
        else:
            # Try smaller step
            lr *= 0.5
            h_new = h - lr * grad
            h_new = _project_to_feasible(h_new)
            val = _correlation_value(h_new)
            if val < best_val - tol:
                h = h_new.copy()
                best_val = val
            else:
                # Try even smaller
                lr *= 0.5
                h_new = h - lr * grad
                h_new = _project_to_feasible(h_new)
                val = _correlation_value(h_new)
                if val < best_val - tol:
                    h = h_new.copy()
                    best_val = val
                else:
                    break
    
    return h, best_val

def _slsqp_optimize(h_init, deadline, beta=80.0):
    """Optimize using SLSQP with smooth objective and analytic gradient."""
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                    'jac': lambda x: np.ones(N)}]
    
    def obj(x):
        return _smooth_max_corr(x, beta)
    
    def obj_grad(x):
        return _compute_grad_smooth_max(x, beta)
    
    try:
        result = minimize(
            obj, h_init, jac=obj_grad, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        if time.time() > deadline:
            return h_init, _correlation_value(h_init)
        
        h_cand = _project_to_feasible(result.x)
        val = _correlation_value(h_cand)
        return h_cand, val
    except Exception:
        return h_init, _correlation_value(h_init)

def _generate_starts(n_starts=8, seed=42):
    """Generate diverse starting points."""
    rng = np.random.RandomState(seed)
    starts = []
    
    # Start 1: uniform 0.5
    starts.append(np.full(N, 0.5))
    
    # Start 2: alternating
    alt = np.ones(N)
    alt[::2] = 0.0
    alt[1::2] = 1.0
    alt = _project_to_feasible(alt)
    starts.append(alt)
    
    # Start 3: sawtooth
    saw = np.linspace(0, 1, N)
    saw = _project_to_feasible(saw)
    starts.append(saw)
    
    # Start 4: inverse sawtooth
    inv_saw = 1 - saw
    inv_saw = _project_to_feasible(inv_saw)
    starts.append(inv_saw)
    
    # Starts 5-8: random
    for _ in range(n_starts - 4):
        h = rng.uniform(0.1, 0.9, size=N)
        h = _project_to_feasible(h)
        starts.append(h)
    
    return starts

def _differential_evolution_search(deadline):
    """Run differential evolution for global search."""
    bounds = [(0.0, 1.0)] * N
    
    def de_obj(x):
        h = _project_to_feasible(x)
        return _smooth_max_corr(h, beta=50.0)
    
    try:
        result = differential_evolution(
            de_obj, bounds, seed=42, maxiter=50, popsize=15,
            tol=1e-8, polish=False
        )
        if time.time() > deadline:
            return None
        h_cand = _project_to_feasible(result.x)
        return h_cand
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    starts = _generate_starts(n_starts=8, seed=42)
    
    best_h = starts[0].copy()
    best_val = _correlation_value(best_h)
    
    # Try differential evolution first if time permits
    if time.time() < deadline - 60.0:
        de_result = _differential_evolution_search(deadline - 50.0)
        if de_result is not None:
            val_de = _correlation_value(de_result)
            if val_de < best_val:
                best_val = val_de
                best_h = de_result.copy()
    
    # SLSQP optimization from multiple starts
    for i, h_init in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        
        h_opt, val_opt = _slsqp_optimize(h_init, deadline - 10.0)
        
        # Local polish
        if time.time() < deadline - 5.0:
            h_pol, val_pol = _local_polish(h_opt, max_iters=300, beta=100.0)
            if val_pol < val_opt:
                h_opt = h_pol
                val_opt = val_pol
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 2.0:
        h_final, val_final = _local_polish(best_h, max_iters=500, tol=1e-15, beta=150.0)
        if val_final < best_val:
            best_h = h_final
            best_val = val_final
    
    # Final validation
    best_h = _project_to_feasible(best_h)
    return best_h
# EVOLVE-BLOCK-END
