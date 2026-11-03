import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _linear_cross_correlation(x):
    """Compute all circular cross-correlations of x with (1-x) via FFT.
    Returns array of length N where result[k] = sum_i x[i]*(1-x[(i+k)%N])"""
    omx = 1.0 - x
    # Circular cross-correlation: result[k] = sum_i x[i] * omx[(i+k) % N]
    # Using FFT: ifft(fft(omx) * conj(fft(x))) gives sum_i x[i] * omx[(i-k) mod N]
    # We want sum_i x[i] * omx[(i+k) mod N]
    # By reindexing: sum_i x[i] * omx[(i+k) mod N] = sum_j x[(j-k) mod N] * omx[j]
    # = ifft(fft(x_rev) * fft(omx)) where x_rev is appropriate...
    # Direct approach: 
    # result[k] = sum_i x[i] * omx[(i+k) % N]
    # Let's use: ifft(fft(x) * conj(fft(omx))) gives sum_i x[i] * omx[(i-k) % N]
    # So result[k] = ifft(fft(x) * conj(fft(omx))) with index -k, i.e. result[k] = corr[(-k)%N]
    # where corr = ifft(fft(x) * conj(fft(omx)))
    X = np.fft.fft(x)
    O = np.fft.fft(omx)
    corr = np.fft.ifft(X * np.conj(O)).real
    # corr[m] = sum_i x[i] * omx[(i-m) % N]
    # We want result[k] = sum_i x[i] * omx[(i+k) % N] = corr[(-k) % N]
    result = np.zeros_like(corr)
    for k in range(len(corr)):
        result[k] = corr[(-k) % len(corr)]
    return result

def _max_overlap(x):
    """Compute max over all circular shifts of sum_i x[i]*(1-x[(i+k)%N])"""
    corr = _linear_cross_correlation(x)
    return float(np.max(corr))

def _project_to_feasible(x, N):
    """Project x to [0,1]^N with sum = N/2 using iterative proportional fitting."""
    x = np.clip(x, 0.0, 1.0)
    target = N / 2.0
    for _ in range(500):
        s = x.sum()
        if abs(s - target) < 1e-14:
            break
        need = target - s
        if need > 0:
            room = 1.0 - x
            total_room = room.sum()
            if total_room < 1e-15:
                break
            x += room * (need / total_room)
        else:
            avail = x.copy()
            total_avail = avail.sum()
            if total_avail < 1e-15:
                break
            x -= avail * (-need / total_avail)
        x = np.clip(x, 0.0, 1.0)
    return x

def _epigraph_objective(params, N):
    """Epigraph formulation: minimize t subject to all overlaps <= t.
    params = [x_0, ..., x_{N-1}, t]
    Uses soft constraints: t + sum(max(0, overlap_k - t)^2 * mu)"""
    x = params[:N]
    t = params[N]
    x = np.clip(x, 0.0, 1.0)
    corr = _linear_cross_correlation(x)
    # Soft penalty for violations
    violations = np.maximum(0.0, corr - t)
    penalty = np.sum(violations ** 2)
    return t + 100.0 * penalty

def _epigraph_objective_grad(params, N):
    """Gradient of epigraph objective."""
    x = params[:N]
    t = params[N]
    x = np.clip(x, 0.0, 1.0)
    omx = 1.0 - x
    corr = _linear_cross_correlation(x)
    violations = np.maximum(0.0, corr - t)
    
    # Gradient w.r.t. t: 1 - 200 * sum(violations)
    grad_t = 1.0 - 200.0 * np.sum(violations)
    
    # Gradient w.r.t. x: -200 * sum_k violations[k] * dcorr_k/dx[i]
    # dcorr_k/dx[i] = (1-x[(i+k)%N]) - x[(i-k)%N]
    # Only for k where violations[k] > 0
    active = violations > 0
    if np.any(active):
        # Compute gradient contribution
        # grad_x[i] = -200 * sum_k active[k]*violations[k] * ((1-x[(i+k)%N]) - x[(i-k)%N])
        # = -200 * sum_k active[k]*violations[k]*(1-x[(i+k)%N]) + 200 * sum_k active[k]*violations[k]*x[(i-k)%N]
        w = np.zeros(N)
        w[active] = violations[active]
        # term1[i] = sum_k w[k] * (1-x[(i+k)%N])
        # term2[i] = sum_k w[k] * x[(i-k)%N]
        # Use FFT for efficiency
        w_rev = np.zeros(N)
        for k in range(N):
            w_rev[(-k) % N] = w[k]
        term1 = np.fft.ifft(np.fft.fft(w_rev) * np.fft.fft(x)).real
        term2 = np.fft.ifft(np.fft.fft(w) * np.fft.fft(x)).real
        grad_x = -200.0 * (np.sum(w) - term1 - term2)
    else:
        grad_x = np.zeros(N)
    
    return np.concatenate([grad_x, [grad_t]])

def _objective_lse(x, beta):
    """Log-sum-exp approximation of max overlap."""
    corr = _linear_cross_correlation(x)
    cmax = np.max(corr)
    logsum = cmax + np.log(np.sum(np.exp((corr - cmax) * beta)) / beta)
    return logsum

def _objective_lse_grad(x, beta):
    """Gradient of LSE approximation."""
    corr = _linear_cross_correlation(x)
    cmax = np.max(corr)
    weights = np.exp((corr - cmax) * beta)
    weights /= weights.sum()
    
    # Gradient: sum_k weights[k] * dcorr_k/dx[i]
    # dcorr_k/dx[i] = (1-x[(i+k)%N]) - x[(i-k)%N]
    w_rev = np.zeros(len(x))
    for k in range(len(x)):
        w_rev[(-k) % len(x)] = weights[k]
    term1 = np.fft.ifft(np.fft.fft(w_rev) * np.fft.fft(x)).real
    term2 = np.fft.ifft(np.fft.fft(weights) * np.fft.fft(x)).real
    grad = 1.0 - term1 - term2
    return grad

def _upsample(x, old_N, new_N):
    """Upsample a sequence from old_N to new_N using linear interpolation."""
    t_old = np.linspace(0, 2*np.pi, old_N, endpoint=False)
    t_new = np.linspace(0, 2*np.pi, new_N, endpoint=False)
    x_interp = np.interp(t_new, t_old, x, period=2*np.pi)
    return x_interp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Coarse-to-fine approach
    N_values = [50, 75, 100]
    
    best_h = None
    best_val = float('inf')
    
    current_N = N_values[0]
    current_x = np.full(current_N, 0.5)
    
    for stage_idx, target_N in enumerate(N_values):
        if time_left() < 15:
            break
        
        if stage_idx > 0:
            # Upsample from previous stage
            current_x = _upsample(current_x, current_N, target_N)
            current_N = target_N
            current_x = _project_to_feasible(current_x, target_N)
        
        N = target_N
        bounds = [(0.0, 1.0)] * N
        
        # Phase 1: Differential Evolution for global search (if time permits)
        if time_left() > 40:
            def de_objective(x):
                x = _project_to_feasible(x, N)
                return _max_overlap(x)
            
            try:
                de_result = differential_evolution(
                    de_objective,
                    bounds,
                    seed=42 + stage_idx,
                    maxiter=50,
                    popsize=15,
                    tol=1e-8,
                    polish=True,
                    updating='deferred'
                )
                if de_result.fun < best_val:
                    best_val = de_result.fun
                    best_h = _project_to_feasible(de_result.x, N)
                    current_x = best_h.copy()
            except Exception:
                pass
        
        # Phase 2: L-BFGS-B with epigraph formulation (multiple restarts)
        if time_left() > 15:
            np.random.seed(100 + stage_idx)
            
            # Generate diverse initializations
            inits = []
            # From current best
            inits.append(current_x.copy())
            # Uniform with noise
            inits.append(np.full(N, 0.5) + np.random.randn(N) * 0.1)
            # Sinusoidal patterns
            t = np.arange(N)
            for freq in [1, 2, 3]:
                inits.append(0.5 + 0.3 * np.sin(2*np.pi*freq*t/N))
                inits.append(0.5 + 0.3 * np.cos(2*np.pi*freq*t/N))
            # Two-level
            for frac in [0.3, 0.5, 0.7]:
                init = np.full(N, 0.5)
                init[:int(N*frac)] = 0.8
                init[int(N*frac):] = 0.2
                inits.append(init)
            # Random
            for _ in range(3):
                inits.append(np.random.uniform(0.1, 0.9, N))
            
            for init in inits:
                if time_left() < 10:
                    break
                
                init = _project_to_feasible(init, N)
                
                # Epigraph optimization
                try:
                    epigraph_init = np.concatenate([init, [best_val if best_val < float('inf') else 0.5]])
                    epigraph_bounds = bounds + [(0.0, 1.5)]
                    
                    result = minimize(
                        _epigraph_objective,
                        epigraph_init,
                        args=(N,),
                        jac=_epigraph_objective_grad,
                        method='L-BFGS-B',
                        bounds=epigraph_bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                 'maxfun': 1000, 'disp': False}
                    )
                    x_opt = _project_to_feasible(result.x[:N], N)
                    val = _max_overlap(x_opt)
                    if val < best_val - 1e-12:
                        best_val = val
                        best_h = x_opt.copy()
                        current_x = x_opt.copy()
                except Exception:
                    pass
                
                # LSE optimization as backup
                if time_left() < 10:
                    break
                try:
                    for beta in [10.0, 30.0]:
                        if time_left() < 8:
                            break
                        result = minimize(
                            _objective_lse,
                            init.copy(),
                            args=(beta,),
                            jac=_objective_lse_grad,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                     'maxfun': 500, 'disp': False}
                        )
                        x_opt = _project_to_feasible(result.x, N)
                        val = _max_overlap(x_opt)
                        if val < best_val - 1e-12:
                            best_val = val
                            best_h = x_opt.copy()
                            current_x = x_opt.copy()
                except Exception:
                    pass
        
        # Phase 3: Fine polish with subgradient steps
        if time_left() > 5 and best_h is not None:
            x = best_h.copy()
            N = len(x)
            for lr in [0.01, 0.005, 0.001]:
                if time_left() < 3:
                    break
                for _ in range(50):
                    if time_left() < 2:
                        break
                    corr = _linear_cross_correlation(x)
                    kmax = np.argmax(corr)
                    omx = 1.0 - x
                    sg = omx.copy() - np.roll(x, kmax)
                    x_new = x - lr * sg
                    x_new = _project_to_feasible(x_new, N)
                    val_new = _max_overlap(x_new)
                    if val_new < _max_overlap(x) - 1e-15:
                        x = x_new
            val = _max_overlap(x)
            if val < best_val - 1e-12:
                best_val = val
                best_h = x.copy()
    
    # Final fallback
    if best_h is None:
        best_h = np.full(100, 0.5)
    else:
        best_h = _project_to_feasible(best_h, len(best_h))
    
    return best_h
