"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def _value_from_h(h):
    """Compute the objective value from step heights h."""
    h = np.asarray(h, dtype=np.float64)
    # Ensure nonnegativity
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2), so spacing is 1.0/(m+1)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y has length m+2, indices 0..m+1
    # The sum is over i=0 to m, so pairs (y[i], y[i+1]) for i=0..m
    # This covers y[0]..y[m+1]
    
    # Vectorized computation
    # term_i = y[i]^2 + y[i]*y[i+1] + y[i+1]^2
    # We can compute this for all i=0..m
    
    y1 = y[:-1]  # y[0]..y[m]
    y2 = y[1:]   # y[1]..y[m+1]
    
    terms = y1**2 + y1 * y2 + y2**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _objective(h):
    """Objective function for minimization (negative of value)."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    return -_value_from_h(h)

def construct():
    """Construct 50 nonnegative step heights to maximize the value."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Seed: 50 equal heights
    seed_h = np.ones(N)
    seed_value = _value_from_h(seed_h)
    
    best_h = seed_h.copy()
    best_value = seed_value
    
    # Strategy: Use differential evolution with a smooth surrogate, then polish
    # with Nelder-Mead on the exact objective.
    
    # First, try a few different initial patterns to get good starting points
    # Pattern 1: Uniform (seed)
    # Pattern 2: Triangular (peak in middle)
    # Pattern 3: Power-law decay from center
    # Pattern 4: Sparse spikes
    
    def generate_initial_patterns():
        patterns = []
        # Uniform
        patterns.append(np.ones(N))
        
        # Triangular
        t = np.linspace(-1, 1, N)
        tri = np.maximum(0, 1 - np.abs(t))
        patterns.append(tri)
        
        # Power-law decay from center
        center = N // 2
        dist = np.abs(np.arange(N) - center)
        power = np.maximum(0.1, 1.0 / (1 + dist * 0.5))
        patterns.append(power)
        
        # Gaussian-like
        x = np.linspace(-2, 2, N)
        gauss = np.exp(-x**2)
        patterns.append(gauss)
        
        # Sparse: few high peaks
        sparse = np.zeros(N)
        sparse[N//4] = 1.0
        sparse[3*N//4] = 1.0
        patterns.append(sparse)
        
        # Bimodal
        bimodal = np.zeros(N)
        bimodal[N//4 - 1:N//4 + 2] = 1.0
        bimodal[3*N//4 - 1:3*N//4 + 2] = 1.0
        patterns.append(bimodal)
        
        return patterns
    
    # Evaluate initial patterns
    initial_patterns = generate_initial_patterns()
    for pat in initial_patterns:
        val = _value_from_h(pat)
        if val > best_value:
            best_value = val
            best_h = pat.copy()
    
    # Use differential evolution for global search
    # Bounds: heights should be nonnegative. Scale them so that max is around 1.
    # Actually, the objective is scale-invariant? Let's check.
    # If we scale h by c, then g scales by c^2, y scales by c^2.
    # l2 scales by c^4, norm_1 scales by c^2, norm_inf scales by c^2.
    # So value = l2/(norm_1*norm_inf) scales by c^4/(c^2*c^2) = 1.
    # The value is scale-invariant! So we can constrain, e.g., max(h) = 1 or sum(h) = 1.
    # Let's use a soft constraint or just let DE handle it with bounds [0, 5].
    
    bounds = [(0.0, 5.0)] * N
    
    # Run differential evolution with a time budget
    de_start = time.time()
    de_budget = 60.0  # seconds for DE
    
    try:
        result_de = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=False,  # We'll do our own polish
            callback=lambda x, convergence: (
                True if time.time() - de_start > de_budget else False
            )
        )
        
        if result_de.success or result_de.fun < -best_value:
            de_h = np.maximum(result_de.x, 0.0)
            de_val = _value_from_h(de_h)
            if de_val > best_value:
                best_value = de_val
                best_h = de_h.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20.0:
        return list(best_h)
    
    # Polish with Nelder-Mead from the best point found
    nm_start = time.time()
    nm_budget = deadline - 10.0 - nm_start
    
    try:
        result_nm = minimize(
            _objective,
            best_h,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                     'adaptive': True}
        )
        
        if result_nm.fun < -best_value:
            nm_h = np.maximum(result_nm.x, 0.0)
            nm_val = _value_from_h(nm_h)
            if nm_val > best_value:
                best_value = nm_val
                best_h = nm_h.copy()
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 5.0:
        return list(best_h)
    
    # Try a few more polish rounds from slightly perturbed best_h
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb best_h slightly
        perturbed = best_h + np.random.normal(0, 0.01 * max(1.0, np.max(best_h)), N)
        perturbed = np.maximum(perturbed, 0.0)
        
        try:
            result_polish = minimize(
                _objective,
                perturbed,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                         'adaptive': True}
            )
            
            if result_polish.fun < -best_value:
                polish_h = np.maximum(result_polish.x, 0.0)
                polish_val = _value_from_h(polish_h)
                if polish_val > best_value:
                    best_value = polish_val
                    best_h = polish_h.copy()
        except Exception:
            pass
    
    return list(best_h)
