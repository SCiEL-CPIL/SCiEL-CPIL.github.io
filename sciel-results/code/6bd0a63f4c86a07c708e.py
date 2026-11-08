"""
Optimization of the second autocorrelation inequality constant C2.
Strategy: Parametric low-rank representation using Discrete Cosine Basis.
Instead of optimizing 50 independent heights, we optimize a small set of
DCT coefficients (amplitudes and frequencies) to generate smooth height profiles.
This reduces dimensionality, enforces global smoothness, and avoids local spikes.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 145.0     # safety margin before 150s kill
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _generate_h_from_params(params):
    """
    Generate height profile from parametric representation.
    params: [a0, a1, a2, f1, f2, f3, a3]
    h[i] = a0 + a1*cos(2*pi*f1*i/N + phase1) + ... 
    We use a sum of cosines with random phases fixed per restart to keep it differentiable.
    Actually, to keep it simple and differentiable, we use:
    h[i] = a0 + sum_k a_k * cos(2*pi*f_k*i/N + phi_k)
    where phi_k are fixed constants for a given restart.
    """
    a0 = params[0]
    # We define up to 3 cosine terms
    # params structure: [a0, a1, a2, a3, f1, f2, f3]
    # Phases are fixed to 0 for simplicity in this first pass, or we can include them.
    # Let's include phases in the param vector to allow more flexibility, but that increases dim.
    # Dim 7 is small. Let's try [a0, a1, a2, a3, f1, f2, f3] with fixed phases 0, pi/2, pi.
    i = np.arange(N)
    h = np.full(N, a0)
    
    # Term 1
    h += params[1] * np.cos(2 * np.pi * params[4] * i / N)
    # Term 2
    h += params[2] * np.cos(2 * np.pi * params[5] * i / N + np.pi / 2)
    # Term 3
    h += params[3] * np.cos(2 * np.pi * params[6] * i / N + np.pi)
    
    return h

def _objective(params):
    """
    Objective function for minimization.
    Returns negative of the value (since we want to maximize).
    """
    h = _generate_h_from_params(params)
    # Ensure non-negative
    if np.any(h < 0):
        # Penalize negative heights heavily
        return 1e6 + np.sum(h[h < 0]**2)
    
    v = compute_value(h)
    return -v

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Initial seed
    track(np.ones(N))
    
    # Parametric bounds
    # a0: [0, 5]
    # a1, a2, a3: [-5, 5]
    # f1, f2, f3: [0.1, 10] (frequencies)
    bounds = [
        (0.0, 5.0),   # a0
        (-5.0, 5.0),  # a1
        (-5.0, 5.0),  # a2
        (-5.0, 5.0),  # a3
        (0.1, 10.0),  # f1
        (0.1, 10.0),  # f2
        (0.1, 10.0),  # f3
    ]
    
    # Phase 1: Global search using Differential Evolution in param space
    # DE is robust and works well in low dimensions.
    if time.time() < deadline - 60.0:
        try:
            de_res = opt.differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=50,
                polish=False, # We will polish manually
                init='sobol'
            )
            if de_res.fun < 0:
                h_de = _generate_h_from_params(de_res.x)
                track(h_de)
        except Exception:
            pass
    
    # Phase 2: Local refinement from best params and random restarts in param space
    # We will perform many short L-BFGS-B runs from different starting points in param space.
    
    # Get current best params if available, else random
    # Since DE doesn't return params directly if it failed or if we didn't track, 
    # let's restart from best_h by inverting? No, just use random starts in param space.
    
    # Let's collect a set of good param candidates.
    param_starts = []
    
    # Add the DE result if it was good
    if time.time() < deadline - 60.0: # This check is redundant if we just ran DE, but safe
         pass # We already tracked it.
    
    # Generate diverse random starts in param space
    for _ in range(100):
        if time.time() > deadline - 20.0:
            break
        
        # Random params
        p = np.array([
            RNG.uniform(0, 2),   # a0
            RNG.uniform(-2, 2),  # a1
            RNG.uniform(-2, 2),  # a2
            RNG.uniform(-2, 2),  # a3
            RNG.uniform(0.5, 5), # f1
            RNG.uniform(0.5, 5), # f2
            RNG.uniform(0.5, 5), # f3
        ])
        param_starts.append(p)
        
        # Evaluate immediately to track
        h_p = _generate_h_from_params(p)
        track(h_p)
        
        # Local refine this param start
        if time.time() > deadline - 10.0:
            continue
            
        try:
            res = opt.minimize(
                _objective,
                p,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < 0:
                h_res = _generate_h_from_params(res.x)
                track(h_res)
        except Exception:
            pass
    
    # Phase 3: Polish the best h found so far using direct optimization on h?
    # The task allows optimizing h directly. The parametric approach found a good region.
    # Let's do a final direct L-BFGS-B on h to see if we can squeeze more.
    if time.time() < deadline - 5.0:
        try:
            # Use a smoothed objective for h to help convergence?
            # Or just direct compute_value with negation.
            # compute_value is not differentiable at 0 due to max(0,h) and abs.
            # But L-BFGS-B can handle it if we stay in positive region.
            
            # Let's use the smoothed version for h as well, just in case.
            def _smooth_h_objective(h, beta=100.0):
                h = np.maximum(0.0, h)
                g = np.convolve(h, h)
                m = len(g)
                y = _build_y(g)
                a = y[:m + 1]
                b = y[1:m + 2]
                l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
                norm_1 = np.sum(g) / (m + 1)
                g_max = np.max(g)
                if g_max <= 0:
                    return 0.0
                lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
                if norm_1 <= 0 or lse <= 0:
                    return 0.0
                return -l2 / (norm_1 * lse)
            
            res_h = opt.minimize(
                lambda x: _smooth_h_objective(x, 100.0),
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_h.fun < 0:
                track(res_h.x)
        except Exception:
            pass
            
    # Phase 4: Nelder-Mead on h for final polish
    if time.time() < deadline - 2.0:
        try:
            def _smooth_h_objective_nm(h, beta=200.0):
                h = np.maximum(0.0, h)
                g = np.convolve(h, h)
                m = len(g)
                y = _build_y(g)
                a = y[:m + 1]
                b = y[1:m + 2]
                l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
                norm_1 = np.sum(g) / (m + 1)
                g_max = np.max(g)
                if g_max <= 0:
                    return 0.0
                lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
                if norm_1 <= 0 or lse <= 0:
                    return 0.0
                return -l2 / (norm_1 * lse)
            
            nm_res = opt.minimize(
                lambda x: _smooth_h_objective_nm(x, 200.0),
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if nm_res.fun < 0:
                track(nm_res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
