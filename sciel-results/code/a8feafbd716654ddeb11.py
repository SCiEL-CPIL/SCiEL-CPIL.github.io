import numpy as np
from scipy.optimize import minimize
import time

N = 400

def compute_value(h):
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try several analytical patterns first
    patterns = []
    
    # Pattern 1: ones (baseline)
    patterns.append(('ones', np.ones(N)))
    
    # Pattern 2: linear ramp
    patterns.append(('ramp', np.linspace(1, 2, N)))
    
    # Pattern 3: quadratic
    x = np.linspace(0, 1, N)
    patterns.append(('quad', 1 + x))
    patterns.append(('quad2', 1 + x*x))
    
    # Pattern 4: low-frequency cosine modes with DC offset
    for k in [1, 2, 3, 4, 5]:
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h = 1.0 + a * np.cos(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'cos_{k}_{a}', h))
        for a in [-0.1, -0.3, -0.5, -0.7, -0.9]:
            h = 1.0 + a * np.cos(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'cos_{k}_{a}', h))
    
    # Pattern 5: sine modes
    for k in [1, 2, 3, 4, 5]:
        for a in [0.3, 0.5, 0.7]:
            h = 1.0 + a * np.sin(2.0 * np.pi * k * np.arange(N) / N)
            patterns.append((f'sin_{k}_{a}', h))
    
    # Pattern 6: block patterns
    for block_frac in [0.1, 0.2, 0.3, 0.5, 0.7]:
        block_len = int(N * block_frac)
        h = np.zeros(N)
        h[:block_len] = 1.0
        patterns.append((f'block_{block_frac}', h))
    
    # Pattern 7: two-level with gap
    for gap_frac in [0.1, 0.2, 0.3]:
        gap_len = int(N * gap_frac)
        h = np.ones(N)
        h[N//2 - gap_len//2:N//2 + gap_len//2] = -0.5
        patterns.append((f'gap_{gap_frac}', h))
    
    # Evaluate all patterns
    for name, h in patterns:
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now do local optimization from the best pattern
    if time.time() < deadline:
        # Use Nelder-Mead on a reduced parameterization
        # Parameterize as: h = 1 + sum of a few cosine modes
        def objective(params):
            if time.time() > deadline:
                return best_val
            a0 = params[0]  # DC offset
            a1 = params[1]  # cos mode 1 amplitude
            a2 = params[2]  # sin mode 1 amplitude
            a3 = params[3]  # cos mode 2 amplitude
            a4 = params[4]  # sin mode 2 amplitude
            i = np.arange(N)
            h = a0 + a1 * np.cos(2*np.pi*i/N) + a2 * np.sin(2*np.pi*i/N) + \
                a3 * np.cos(4*np.pi*i/N) + a4 * np.sin(4*np.pi*i/N)
            return compute_value(h)
        
        # Start from best pattern converted to coefficients
        # Fit the best_h to the parameterization
        i = np.arange(N)
        A = np.column_stack([
            np.ones(N),
            np.cos(2*np.pi*i/N),
            np.sin(2*np.pi*i/N),
            np.cos(4*np.pi*i/N),
            np.sin(4*np.pi*i/N)
        ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            x0 = coeffs
        except:
            x0 = np.array([1.0, 0.1, 0.0, 0.05, 0.0])
        
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 5000})
        
        if result.fun < best_val:
            best_val = result.fun
            a0, a1, a2, a3, a4 = result.x
            best_h = a0 + a1 * np.cos(2*np.pi*i/N) + a2 * np.sin(2*np.pi*i/N) + \
                     a3 * np.cos(4*np.pi*i/N) + a4 * np.sin(4*np.pi*i/N)
    
    # Try direct optimization on a subset of free parameters if time allows
    if time.time() < deadline - 20:
        # Use COBYLA on a reduced set: optimize a few "knobs"
        # Parameterize as piecewise constant with few segments
        def obj2(params):
            if time.time() > deadline:
                return best_val
            # params: [val1, val2, val3, val4, val5] for 5 equal segments
            h = np.repeat(params, N // 5)
            if len(h) < N:
                h = np.concatenate([h, np.full(N - len(h), params[-1])])
            return compute_value(h)
        
        # Start from best_h binned into 5 segments
        seg_size = N // 5
        init_params = np.array([np.mean(best_h[j*seg_size:(j+1)*seg_size]) for j in range(5)])
        
        result2 = minimize(obj2, init_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 10000})
        
        if result2.fun < best_val:
            best_val = result2.fun
            seg_size = N // 5
            best_h = np.repeat(result2.x, seg_size)
            if len(best_h) < N:
                best_h = np.concatenate([best_h, np.full(N - len(best_h), result2.x[-1])])
    
    # Final check
    final_val = compute_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
