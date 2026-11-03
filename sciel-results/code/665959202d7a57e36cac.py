import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 600
DEADLINE = 145.0

def compute_score(f):
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    n_fft = 2 * N + 1
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(beta * shifted))) / beta

def fft_autoconv(f):
    n_fft = 2 * N + 1
    F = np.fft.rfft(f, n=n_fft)
    return np.fft.irfft(F * F, n=n_fft)

def objective_1d(h, idx, h_copy):
    """Objective for 1D minimization: vary h[idx] while keeping others fixed."""
    def obj(t):
        if t < 0:
            return 1e10
        h_new = h_copy.copy()
        h_new[idx] = t
        s = np.sum(h_new)
        if s < 1e-15:
            return 1e10
        h_norm = h_new / s
        conv = fft_autoconv(h_norm)
        return np.max(conv)
    return obj

def parametric_to_step(params, family='poly'):
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    if family == 'poly':
        a, b, c, d, e, f_ = params
        f = a + b*x**2 + c*x**4 + d*x**6 + e*x**8 + f_*x**10
    elif family == 'cos':
        a, b, c, d, e, f_ = params
        f = a + b*np.cos(2*np.pi*x) + c*np.cos(4*np.pi*x) + d*np.cos(6*np.pi*x) + e*np.cos(8*np.pi*x) + f_*np.cos(10*np.pi*x)
    elif family == 'gauss':
        a, b, c, d, e, f_ = params
        f = a * np.exp(-b*x**2) + c * np.exp(-d*x**2)
        f = f + e * x**2 * np.exp(-f_ * x**2)
    return np.maximum(f, 0.0)

def spectral_init(rng, n_harmonics=8):
    """Initialize in spectral domain: random cosine coefficients."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    coeffs = rng.normal(0, 0.1, size=n_harmonics)
    coeffs[0] = 1.0  # DC term
    f = np.ones(N)
    for k in range(1, n_harmonics):
        f += coeffs[k] * np.cos(2 * np.pi * k * x)
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    else:
        f = np.ones(N) / N
    return f

def objective(params, beta=50.0, family='poly'):
    f = parametric_to_step(params, family)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    f = f / s
    n_fft = 2 * N + 1
    F = np.fft.rfft(f, n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft)
    return smooth_max(conv, beta)

def exact_coordinate_descent(h, start_time, max_time, max_passes=50):
    """Exact 1D coordinate descent with symmetry enforcement."""
    current_h = h.copy()
    s = np.sum(current_h)
    if s < 1e-12:
        return current_h, compute_score(current_h)
    current_h = current_h / s
    
    # Enforce symmetry
    half = N // 2
    current_h[:half] = (current_h[:half] + current_h[half:][::-1]) / 2
    current_h[half:] = current_h[:half][::-1]
    if N % 2 == 1:
        current_h[half] = current_h[half-1]
    s = np.sum(current_h)
    if s > 1e-12:
        current_h = current_h / s
    
    current_val = compute_score(current_h)
    best_h = current_h.copy()
    best_val = current_val
    
    rng = np.random.RandomState(42)
    h_copy = current_h.copy()
    
    for pass_num in range(max_passes):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        indices = rng.permutation(half)
        
        for i in indices:
            if time.time() - start_time > max_time:
                break
            
            # Exact 1D minimization for coordinate i
            obj = objective_1d(current_h, i, h_copy)
            
            # Bracket around current value
            x0 = current_h[i]
            if x0 < 1e-10:
                x0 = 1e-8
            
            # Try to find a better value
            try:
                res = minimize_scalar(obj, bounds=(0, x0 * 3 + 0.01), method='bounded',
                                       options={'xatol': 1e-8, 'maxiter': 100})
                if res.success and res.fun < current_val - 1e-12:
                    current_h[i] = res.x
                    current_h[N-1-i] = res.x
                    s = np.sum(current_h)
                    if s > 1e-12:
                        current_h = current_h / s
                    current_val = compute_score(current_h)
                    h_copy = current_h.copy()
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
            except:
                pass
        
        # Also try pair adjustments: move mass between coordinates
        if time.time() - start_time > max_time:
            break
        
        for _ in range(half // 2):
            if time.time() - start_time > max_time:
                break
            
            i = rng.randint(0, half)
            j = rng.randint(0, half)
            if i == j:
                continue
            
            # 2D subproblem: vary h[i] and h[j] keeping sum constant
            total = current_h[i] + current_h[j]
            if total < 1e-12:
                continue
            
            def obj_2d(t):
                t = np.clip(t, 0, total)
                h_new = current_h.copy()
                h_new[i] = t
                h_new[N-1-i] = t
                h_new[j] = total - t
                h_new[N-1-j] = total - t
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    return 1e10
                h_norm = h_new / s_new
                conv = fft_autoconv(h_norm)
                return np.max(conv)
            
            try:
                res = minimize_scalar(obj_2d, bounds=(0, total), method='bounded',
                                       options={'xatol': 1e-8, 'maxiter': 100})
                if res.success and res.fun < current_val - 1e-12:
                    current_h[i] = res.x
                    current_h[N-1-i] = res.x
                    current_h[j] = total - res.x
                    current_h[N-1-j] = total - res.x
                    s = np.sum(current_h)
                    if s > 1e-12:
                        current_h = current_h / s
                    current_val = compute_score(current_h)
                    h_copy = current_h.copy()
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
            except:
                pass
        
        if not improved:
            break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    rng = np.random.RandomState(123)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Spectral initializations
    for nh in [4, 6, 8, 10]:
        if time.time() - start_time > DEADLINE - 60:
            break
        for _ in range(3):
            f = spectral_init(rng, n_harmonics=nh)
            candidates.append(f)
    
    # Parametric initializations
    families = ['poly', 'cos', 'gauss']
    for family in families:
        if time.time() - start_time > DEADLINE - 50:
            break
        if family == 'poly':
            guesses = [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.5, 0.1, 0.0, 0.0, 0.0],
                [1.0, -0.8, 0.05, 0.0, 0.0, 0.0],
            ]
        elif family == 'cos':
            guesses = [
                [1.0, -0.5, 0.2, -0.05, 0.01, 0.0],
                [1.0, -0.3, 0.1, -0.02, 0.0, 0.0],
                [1.0, -0.8, 0.3, -0.08, 0.02, 0.0],
            ]
        elif family == 'gauss':
            guesses = [
                [1.0, 10.0, 0.0, 5.0, 0.0, 20.0],
                [1.0, 5.0, 0.5, 2.0, 0.1, 10.0],
                [0.8, 8.0, 0.3, 3.0, 0.05, 15.0],
            ]
        for guess in guesses:
            f = parametric_to_step(guess, family)
            s = np.sum(f)
            if s > 1e-12:
                f = f / s
                candidates.append(f)
    
    # Phase 2: Quick parametric optimization on best candidates
    best_params = None
    best_obj = np.inf
    best_family = 'poly'
    
    if time.time() - start_time < DEADLINE - 40:
        for family in ['cos', 'poly']:
            if time.time() - start_time > DEADLINE - 35:
                break
            beta = 50.0
            if family == 'cos':
                x0 = [1.0, -0.5, 0.2, -0.05, 0.01, 0.0]
            else:
                x0 = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
            try:
                result = minimize(objective, x0, args=(beta, family), method='Nelder-Mead',
                                  options={'maxiter': 1500, 'adaptive': True})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
                    best_family = family
            except:
                pass
    
    if best_params is not None:
        f = parametric_to_step(best_params, best_family)
        s = np.sum(f)
        if s > 1e-12:
            f = f / s
            score = compute_score(f)
            if score < best_score:
                best_score = score
                best_f = f
                candidates.insert(0, f)
    
    # Phase 3: Exact coordinate descent on multiple starting points
    # Use remaining time for refinement
    remaining = DEADLINE - time.time() - 2
    
    # Sort candidates by score and refine top ones
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0])
    
    n_refine = min(5, len(scored_candidates))
    time_per_refine = remaining / n_refine
    
    for i in range(n_refine):
        if time.time() - start_time > DEADLINE - 1:
            break
        _, f_init = scored_candidates[i]
        max_time_for_this = min(time_per_refine, DEADLINE - time.time() - 0.5)
        if max_time_for_this < 1:
            break
        
        refined_f, refined_score = exact_coordinate_descent(
            f_init, start_time, start_time + max_time_for_this, max_passes=30)
        
        if refined_score < best_score:
            best_score = refined_score
            best_f = refined_f
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]
