import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 100
DEADLINE_MARGIN = 20.0  # seconds before the 150s limit to stop

def objective(h):
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def objective_grad(h):
    """Return (max_corr, gradient) for the max-correlation objective."""
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode='full')
    max_idx = np.argmax(corr)
    max_val = corr[max_idx]
    
    # For full mode, corr[k] = sum_j h[j] * (1-h)[j - (k - (N-1))]
    # Let lag = k - (N-1). Then corr[k] = sum_j h[j] * (1-h)[j - lag]
    # d(corr[k])/dh_i = (1-h)[i-lag] - h[i+lag] (with boundary handling)
    lag = max_idx - (N - 1)
    grad = np.zeros(N)
    
    # Term 1: (1-h)[i-lag] for i where i-lag is in [0, N-1]
    # i ranges from max(0, lag) to min(N-1, N-1+lag)
    i_start = max(0, lag)
    i_end = min(N, N + lag)  # exclusive
    if i_start < i_end:
        indices = np.arange(i_start, i_end)
        grad[indices] += (1.0 - h[indices - lag])
    
    # Term 2: -h[i+lag] for i where i+lag is in [0, N-1]
    # i ranges from max(0, -lag) to min(N-1, N-1-lag)
    i_start2 = max(0, -lag)
    i_end2 = min(N, N - lag)  # exclusive
    if i_start2 < i_end2:
        indices = np.arange(i_start2, i_end2)
        grad[indices] -= h[indices + lag]
    
    return max_val, grad

def project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # Final exact sum correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute to all non-bound elements
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def smooth_objective(h, alpha=50.0):
    """Log-sum-exp approximation of max correlation."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    max_c = np.max(corr)
    if alpha == 0:
        return max_c
    scaled = alpha * (corr - max_c)
    # Use stable log-sum-exp
    max_scaled = np.max(scaled)
    lse = max_c + max_scaled / alpha + np.log(np.sum(np.exp(scaled - max_scaled))) / alpha
    return lse

def smooth_objective_grad(h, alpha=50.0):
    """Gradient of log-sum-exp approximation."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    max_c = np.max(corr)
    scaled = alpha * (corr - max_c)
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of each corr[k] w.r.t. h
    # We need sum_k weights[k] * d(corr[k])/dh
    grad = np.zeros(N)
    for k in range(len(corr)):
        w = weights[k]
        if w < 1e-12:
            continue
        lag = k - (N - 1)
        # d(corr[k])/dh_i = (1-h)[i-lag] - h[i+lag]
        i_start = max(0, lag)
        i_end = min(N, N + lag)
        if i_start < i_end:
            indices = np.arange(i_start, i_end)
            grad[indices] += w * (1.0 - h[indices - lag])
        
        i_start2 = max(0, -lag)
        i_end2 = min(N, N - lag)
        if i_start2 < i_end2:
            indices = np.arange(i_start2, i_end2)
            grad[indices] -= w * h[indices + lag]
    
    return lse_val, grad

def lse_objective(h, alpha=50.0):
    """Log-sum-exp objective value only."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    max_c = np.max(corr)
    scaled = alpha * (corr - max_c)
    max_scaled = np.max(scaled)
    lse = max_c + max_scaled / alpha + np.log(np.sum(np.exp(scaled - max_scaled))) / alpha
    return lse

def lse_objective_grad(h, alpha=50.0):
    """Return (lse_value, gradient) for log-sum-exp objective."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    max_c = np.max(corr)
    scaled = alpha * (corr - max_c)
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    lse_val = max_c + max_scaled / alpha + np.log(np.sum(exp_scaled)) / alpha
    
    grad = np.zeros(N)
    for k in range(len(corr)):
        w = weights[k]
        if w < 1e-14:
            continue
        lag = k - (N - 1)
        i_start = max(0, lag)
        i_end = min(N, N + lag)
        if i_start < i_end:
            indices = np.arange(i_start, i_end)
            grad[indices] += w * (1.0 - h[indices - lag])
        
        i_start2 = max(0, -lag)
        i_end2 = min(N, N - lag)
        if i_start2 < i_end2:
            indices = np.arange(i_start2, i_end2)
            grad[indices] -= w * h[indices + lag]
    
    return lse_val, grad

def coordinate_descent_polish(h, max_iter=5000, tol=1e-14):
    """Local refinement via coordinate descent with exact 1D minimization."""
    h = h.copy()
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Minimize over h[i] given others fixed
            # corr depends on h[i] through terms where h[i] appears
            # We can do a line search: define f(t) = objective(h with h[i]=t)
            # subject to sum constraint: adjust another coordinate or use penalty
            
            # Simpler: just try small perturbations
            best_t = h[i]
            best_val = objective(h)
            
            # Try a few values
            for t in [max(0, h[i] - 0.01), max(0, h[i] - 0.001), h[i], min(1, h[i] + 0.001), min(1, h[i] + 0.01)]:
                h_test = h.copy()
                h_test[i] = t
                # Fix sum
                h_test = project_to_feasible(h_test)
                v = objective(h_test)
                if v < best_val - 1e-15:
                    best_val = v
                    best_t = t
                    h = h_test.copy()
                    improved = True
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    starts = [np.full(N, 0.5)]
    
    # Random perturbations
    for _ in range(8):
        h_pert = np.full(N, 0.5) + rng.randn(N) * 0.15
        starts.append(project_to_feasible(h_pert))
    
    # Sine waves
    for freq in [1, 2, 3, 4]:
        h_sine = 0.5 + 0.45 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(project_to_feasible(h_sine))
    
    # Cosine waves
    for freq in [1, 2, 3]:
        h_cos = 0.5 + 0.45 * np.cos(2 * np.pi * freq * np.arange(N) / N)
        starts.append(project_to_feasible(h_cos))
    
    # Linear ramp
    h_ramp = np.linspace(0.0, 1.0, N)
    starts.append(project_to_feasible(h_ramp))
    
    # Triangle wave
    t = np.linspace(0, 1, N)
    h_tri = 0.5 + 0.5 * np.abs(2 * t - 1) - 0.25
    starts.append(project_to_feasible(h_tri))
    
    # Two-level blocks
    for block_size in [10, 20, 25, 50]:
        h_bin = np.zeros(N)
        n_ones = N // 2
        h_bin[:n_ones] = 1.0
        starts.append(h_bin.copy())
    
    # Alternating
    h_alt = np.where(np.arange(N) % 2 == 0, 0.25, 0.75)
    starts.append(h_alt.copy())
    
    # Sawtooth
    h_saw = np.linspace(0, 1, N)
    h_saw2 = np.linspace(1, 0, N)
    starts.append(project_to_feasible(h_saw))
    starts.append(project_to_feasible(h_saw2))
    
    # Smooth Gaussian-like bumps
    for center in [25, 50, 75]:
        x = np.arange(N)
        h_gauss = 0.5 + 0.4 * np.exp(-((x - center) ** 2) / (2 * 15 ** 2))
        starts.append(project_to_feasible(h_gauss))
    
    # Optimize from each starting point
    alphas = [100.0, 50.0, 20.0, 10.0]
    
    for h0 in starts:
        if time.time() > deadline - 15:
            break
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    lse_objective,
                    h0,
                    args=(alpha,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                
                h_res = project_to_feasible(res.x)
                val = objective(h_res)
                if val < best_val:
                    best_val = val
                    best_h = h_res.copy()
                    
            except Exception:
                pass
    
    # Polish with SLSQP on the best solution found
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lse_objective,
                best_h,
                args=(200.0,),
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            val = objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5:
        best_h = coordinate_descent_polish(best_h, max_iter=2000, tol=1e-14)
        best_h = project_to_feasible(best_h)
        val = objective(best_h)
        if val < best_val:
            best_val = val
    
    # Final high-precision SLSQP
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lse_objective,
                best_h,
                args=(500.0,),
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            h_res = project_to_feasible(res.x)
            val = objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_h = project_to_feasible(best_h)
    
    # Verify constraints
    assert np.all(np.isfinite(best_h)), "Non-finite values"
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "Out of bounds"
    assert abs(np.sum(best_h) - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h
