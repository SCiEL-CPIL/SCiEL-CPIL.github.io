"""Model-and-optimize: Multi-objective Pareto search + soft-max L-BFGS + vectorized polish."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 142  # Safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 5)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective_only(x):
    """Compute objective value only (fast, no gradient)."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    return m / (s * s)

def _sidelobe_energy(x):
    """Compute L2-norm squared of autoconvolution sidelobes (excluding DC)."""
    ac = _autoconv_fft(x)
    # Sidelobes: exclude the central peak region
    # The main lobe is around the center of the autoconvolution
    # For zero-centered signals, the peak is at index N-1
    # We exclude a window around the peak
    center = N - 1
    half_width = 10
    mask = np.ones(len(ac), dtype=bool)
    mask[max(0, center-half_width):min(len(ac), center+half_width+1)] = False
    return np.sum(ac[mask] ** 2)

def _multi_objective(x):
    """Compute both objectives: peak ratio and sidelobe energy."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, 1e10
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    peak = np.max(abs_ac)
    center = N - 1
    half_width = 10
    mask = np.ones(len(ac), dtype=bool)
    mask[max(0, center-half_width):min(len(ac), center+half_width+1)] = False
    sidelobe = np.sum(ac[mask] ** 2)
    return peak / (s * s), sidelobe / (s * s)

def _soft_max_and_grad(x, beta):
    """Compute soft-max surrogate for max|ac| and its gradient using FFT."""
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    
    if sum_exps == 0:
        return m, np.zeros(N)
        
    log_sum = np.log(sum_exps)
    soft_val = m + log_sum / beta
    
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs
    
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2
        
    G_f = np.fft.rfft(grad_ac, n=L_corr)
    X_f = np.fft.rfft(x, n=L_corr)
    
    w = grad_ac
    w_rev = w[::-1]
    
    W_f = np.fft.rfft(w, n=L_corr)
    W_rev_f = np.fft.rfft(w_rev, n=L_corr)
    
    corr1 = np.fft.irfft(W_f * np.conj(X_f), n=L_corr)
    corr2 = np.fft.irfft(W_rev_f * np.conj(X_f), n=L_corr)
    
    grad_x = corr1[:N] + corr2[:N]
    
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    """Compute objective and gradient for peak objective."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad

def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase/amplitude manipulation."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1
    freqs = np.fft.rfftfreq(L_fft)
    
    if mode == 0:
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 2:
        mag = 1.0 + 0.5 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    else:
        mag = np.ones(n_freq)
        
    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0
    
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    
    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]
    
    s = np.sum(x)
    if s < 0:
        x = -x
        
    return x

def _pareto_dominates(a, b):
    """Check if point a dominates point b (both objectives to minimize)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])

def _nsga2_select(pop, objectives, pop_size, elite_size):
    """Simple NSGA-II style selection for multi-objective."""
    # Combine population with objectives
    combined = list(zip(pop, objectives))
    
    # Non-dominated sorting
    fronts = []
    remaining = list(range(len(combined)))
    
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and _pareto_dominates(objectives[j], objectives[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        
        if not front:
            break
        
        fronts.append(front)
        for i in front:
            remaining.remove(i)
    
    # Select from fronts
    selected = []
    for front in fronts:
        for i in front:
            if len(selected) < elite_size:
                selected.append(combined[i])
    
    return selected

def _evolutionary_phase(x0, rng, max_gens=50):
    """Run a simple multi-objective evolutionary search."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    # Initialize population
    pop_size = 30
    elite_size = 15
    pop = [x0.copy()]
    for i in range(pop_size - 1):
        x = x0 + rng.normal(0, 0.1, size=N)
        s = np.sum(x)
        if s < 0:
            x = -x
        pop.append(x)
    
    # Compute objectives
    objectives = [_multi_objective(x) for x in pop]
    
    best_x = x0.copy()
    best_val = _objective_only(x0)
    best_side = _sidelobe_energy(x0)
    
    for gen in range(max_gens):
        if _check_deadline():
            break
            
        # Select elite
        elites = _nsga2_select(pop, objectives, pop_size, elite_size)
        
        # Create offspring
        new_pop = [e[0].copy() for e in elites]
        
        while len(new_pop) < pop_size:
            # Tournament selection
            candidates = rng.choice(len(pop), size=3, replace=False)
            # Pick the one with best combined objective
            best_idx = min(candidates, key=lambda i: objectives[i][0] + objectives[i][1])
            parent = pop[best_idx].copy()
            
            # Mutation
            if rng.random() < 0.8:
                # Gaussian mutation
                parent = parent + rng.normal(0, 0.05, size=N)
            else:
                # Coordinate mutation
                idx = rng.randint(N)
                parent[idx] += rng.normal(0, 0.1)
            
            # Repair: ensure sum is positive
            s = np.sum(parent)
            if s < 0:
                parent = -parent
            
            new_pop.append(parent)
        
        # Evaluate offspring
        new_objectives = [_multi_objective(x) for x in new_pop]
        
        # Update best
        for x, obj in zip(new_pop, new_objectives):
            val = obj[0]
            side = obj[1]
            if val < best_val:
                best_val = val
                best_x = x.copy()
                best_side = side
        
        # Replace population
        pop = new_pop
        objectives = new_objectives
    
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    
    if _check_deadline():
        return best_x
        
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral Initialization
    for mode in range(3):
        if _check_deadline():
            break
        x0 = _spectral_init(rng, mode)
        val = _objective_only(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
    if _check_deadline():
        return best_x
        
    # Phase 2: Multi-objective evolutionary search (NSGA-II style)
    x_evo, val_evo = _evolutionary_phase(best_x, rng, max_gens=30)
    if val_evo < best_val:
        best_val = val_evo
        best_x = x_evo.copy()
        
    if _check_deadline():
        return best_x
        
    # Phase 3: L-BFGS with annealed beta
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if _check_deadline():
            break
        try:
            def obj_and_grad_func(x):
                return _objective_and_grad(x, beta)
                
            result = minimize(obj_and_grad_func, best_x, method='L-BFGS-B',
                              jac=True, options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass
            
    if _check_deadline():
        return best_x
        
    # Phase 4: Vectorized Coordinate Descent Polish
    step_size = 0.05
    n_steps = 0
    max_steps = 3000
    no_improve_count = 0
    max_no_improve = 300
    
    while n_steps < max_steps and no_improve_count < max_no_improve:
        if _check_deadline():
            break
            
        improved = False
        
        n_coords = 25
        indices = rng.choice(N, size=n_coords, replace=False)
        
        for i in indices:
            for delta in [step_size, -step_size, 0.5*step_size, -0.5*step_size]:
                x_trial = best_x.copy()
                x_trial[i] += delta
                val = _objective_only(x_trial)
                if val < best_val:
                    best_val = val
                    best_x = x_trial
                    improved = True
                    break
        
        if improved:
            no_improve_count = 0
            step_size = min(0.2, step_size * 1.05)
        else:
            no_improve_count += 1
            step_size *= 0.95
            
        n_steps += 1
        
    return best_x
# EVOLVE-BLOCK-END
