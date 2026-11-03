import numpy as np
from scipy.optimize import minimize
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
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def compute_score_fft(f):
    f = np.asarray(f, dtype=np.float64)
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

def local_refinement_symmetric(h, start_time, max_time):
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    s = np.sum(current_h)
    if s < 1e-12:
        return best_h, best_val
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
    
    # Adaptive step sizes
    deltas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, -0.001, -0.002, -0.005, -0.01, -0.02, -0.05]
    
    while time.time() - start_time < max_time:
        improved = False
        indices = rng.permutation(half)
        for i in indices:
            if time.time() - start_time > max_time:
                break
            # Try moving mass from i to a random other position
            for delta in deltas:
                h_new = current_h.copy()
                h_new[i] = max(0, h_new[i] + delta)
                h_new[N-1-i] = h_new[i]
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new / s_new
                val_new = compute_score(h_new)
                if val_new < current_val - 1e-13:
                    current_h = h_new
                    current_val = val_new
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                    break
            # Also try pair swaps
            if time.time() - start_time > max_time:
                break
            j = rng.randint(0, half)
            if j == i:
                continue
            for swap_frac in [0.05, 0.1, 0.2, 0.3, 0.5]:
                h_new = current_h.copy()
                amt = swap_frac * min(current_h[i], current_h[j])
                h_new[i] -= amt
                h_new[N-1-i] -= amt
                h_new[j] += amt
                h_new[N-1-j] += amt
                h_new = np.maximum(h_new, 0)
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new / s_new
                val_new = compute_score(h_new)
                if val_new < current_val - 1e-13:
                    current_h = h_new
                    current_val = val_new
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                    break
        if not improved:
            # Try a few random perturbations
            for _ in range(10):
                if time.time() - start_time > max_time:
                    break
                h_new = current_h.copy()
                pert_idx = rng.randint(0, half, size=5)
                for idx in pert_idx:
                    h_new[idx] = max(0, h_new[idx] * (1 + rng.uniform(-0.1, 0.1)))
                    h_new[N-1-idx] = h_new[idx]
                s_new = np.sum(h_new)
                if s_new < 1e-12:
                    continue
                h_new = h_new / s_new
                val_new = compute_score(h_new)
                if val_new < current_val - 1e-13:
                    current_h = h_new
                    current_val = val_new
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()
                    break
            if not improved:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    best_f = np.ones(N) / N
    best_score = compute_score(best_f)
    
    # Try multiple parametric families
    families = ['poly', 'cos', 'gauss']
    best_params = None
    best_obj = np.inf
    best_family = 'poly'
    
    rng = np.random.RandomState(123)
    
    for family in families:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        initial_guesses = []
        if family == 'poly':
            initial_guesses = [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                [1.0, -1.5, 0.1, 0.0, 0.0, 0.0],
                [1.0, -0.8, 0.05, 0.0, 0.0, 0.0],
                [1.0, -1.2, 0.08, 0.0, 0.0, 0.0],
                [1.0, -2.0, 0.3, -0.02, 0.0, 0.0],
            ]
        elif family == 'cos':
            initial_guesses = [
                [1.0, -0.5, 0.2, -0.05, 0.01, 0.0],
                [1.0, -0.3, 0.1, -0.02, 0.0, 0.0],
                [1.0, -0.8, 0.3, -0.08, 0.02, 0.0],
            ]
        elif family == 'gauss':
            initial_guesses = [
                [1.0, 10.0, 0.0, 5.0, 0.0, 20.0],
                [1.0, 5.0, 0.5, 2.0, 0.1, 10.0],
                [0.8, 8.0, 0.3, 3.0, 0.05, 15.0],
            ]
        
        for beta in [30.0, 60.0]:
            if time.time() - start_time > DEADLINE - 25:
                break
            for guess in initial_guesses:
                if time.time() - start_time > DEADLINE - 20:
                    break
                x0 = np.array(guess) + rng.uniform(-0.05, 0.05, size=6)
                try:
                    result = minimize(objective, x0, args=(beta, family), method='Nelder-Mead',
                                      options={'maxiter': 3000, 'adaptive': True})
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                        best_family = family
                except:
                    pass
            
            # Random restarts
            for _ in range(3):
                if time.time() - start_time > DEADLINE - 15:
                    break
                if family == 'poly':
                    x0 = [rng.uniform(0.5, 1.5), rng.uniform(-3, 0), rng.uniform(0, 0.5),
                          rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(0, 0.05)]
                elif family == 'cos':
                    x0 = [rng.uniform(0.5, 1.5), rng.uniform(-1.5, 0), rng.uniform(0, 0.5),
                          rng.uniform(-0.2, 0.2), rng.uniform(-0.05, 0.05), rng.uniform(0, 0.05)]
                elif family == 'gauss':
                    x0 = [rng.uniform(0.5, 1.5), rng.uniform(2, 20), rng.uniform(0, 1),
                          rng.uniform(1, 10), rng.uniform(0, 0.5), rng.uniform(5, 30)]
                try:
                    result = minimize(objective, x0, args=(beta, family), method='Nelder-Mead',
                                      options={'maxiter': 2000, 'adaptive': True})
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
    
    # Local refinement
    if time.time() - start_time < DEADLINE - 3:
        refined_f, refined_score = local_refinement_symmetric(best_f, start_time, DEADLINE - 1)
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]
