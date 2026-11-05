import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    # Seed value
    seed_val = 2.0
    
    # Best result found
    best_h = np.ones(N)
    best_val = seed_val
    
    # Soft max temperature
    T = 0.05
    
    def objective(h):
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        c_max = np.max(conv)
        exp_conv = np.exp((conv - c_max) / T)
        soft_m = T * np.log(np.sum(exp_conv)) + c_max
        
        val = abs(2 * N * soft_m / (s * s))
        
        # Scale invariance penalty
        norm = np.linalg.norm(h)
        penalty = 0.01 * (norm - 1.0)**2
        
        return val + penalty

    def get_val(h):
        s = np.sum(h)
        if s == 0:
            return np.inf
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))

    # Try multiple initializations
    inits = []
    
    # 1. Seed
    inits.append(np.ones(N))
    
    # 2. Random normal, normalized sum to 1
    rng = np.random.default_rng(42)
    h_rand = rng.normal(size=N)
    h_rand = h_rand / np.sum(h_rand) * N # Sum approx N
    inits.append(h_rand)
    
    # 3. Sine wave with DC
    x = np.arange(N)
    h_sin = 1.0 + 0.5 * np.sin(2 * np.pi * x / N)
    inits.append(h_sin)
    
    # 4. Alternating with large DC
    h_alt = np.ones(N)
    h_alt[1::2] = 0.5
    inits.append(h_alt)
    
    bounds = [(-10.0, 10.0)] * N
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5})
            
            val_res = get_val(res.x)
            
            if val_res < best_val:
                best_val = val_res
                best_h = res.x
        except Exception:
            pass
            
    return best_h
