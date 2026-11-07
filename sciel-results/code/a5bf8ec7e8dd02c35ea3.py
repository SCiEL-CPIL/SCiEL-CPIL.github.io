import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    F = 4096.0 * c3
    E = -122880.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-9:
            r = root.real
            if r > 0:
                eps = 1e-7
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    seed_c0, seed_c1, seed_c2 = -113.0/100.0, 1.0/25.0, 1.0/3240.0
    seed_val = compute_value([seed_c0, seed_c1, seed_c2])
    best_x = np.array([seed_c0, seed_c1, seed_c2])
    best_val = seed_val
    
    # Multi-start local optimization with diverse initial points
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = [np.array([seed_c0, seed_c1, seed_c2])]
    
    # Add random starts in various ranges
    for scale in [0.1, 0.5, 1.0, 5.0, 20.0]:
        for _ in range(5):
            pt = rng.uniform(-scale, scale, size=3)
            starts.append(pt)
    
    # Add structured starts targeting small roots
    # Try to make A/B small and positive
    for c3_val in [1e-6, -1e-6, 1e-5, -1e-5, 1e-4, -1e-4]:
        for c2_val in [1e-4, -1e-4, 1e-3, -1e-3]:
            for c1_val in [1e-3, -1e-3, 1e-2, -1e-2]:
                c0_val = -c1_val * 12.0 - c2_val * 1680.0 - c3_val * 4084080.0
                starts.append(np.array([c0_val, c1_val, c2_val]))
    
    def obj(x):
        if time.time() > deadline:
            return 1e9
        try:
            v = compute_value(x)
            return v
        except:
            return 1e9
    
    for x0 in starts:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # If time remains, do a differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-100, 100), (-10, 10), (-10, 10)]
            de_res = differential_evolution(
                obj, bounds, seed=42, maxiter=50, popsize=15,
                tol=1e-10, polish=False,
                callback=lambda x, convergence: (time.time() > deadline)
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except:
            pass
    
    # Final local polish
    if time.time() < deadline - 10.0:
        try:
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return np.array(best_x)
