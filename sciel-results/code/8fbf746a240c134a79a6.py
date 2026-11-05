import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    seed_val = 2.0
    best_h = np.ones(N)
    best_val = seed_val
    
    # We optimize the peak autocorrelation under the constraint sum(h) = N.
    # Since the value is scale invariant, fixing sum(h)=N is valid.
    # Value = 2 * N * max(conv(h, h)) / N^2 = 2 * max(conv(h, h)) / N
    # Wait, if sum(h) = N, denominator is N^2.
    # value = 2 * N * max_conv / N^2 = 2 * max_conv / N.
    # So minimizing max_conv is equivalent.
    
    def project(h):
        # Project h onto the hyperplane sum(h) = N
        # h_proj = h - (sum(h) - N) / N
        return h - (np.sum(h) - N) / N

    def objective(h):
        h_p = project(h)
        conv = np.convolve(h_p, h_p)
        # We want to minimize the maximum value of the autocorrelation.
        # Note: The task defines value = abs(2 * N * max(conv) / sum^2).
        # If we fix sum=N, value = 2 * max(conv) / N.
        # So we just need to minimize max(conv).
        return np.max(conv)

    # Generate diverse initializations
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Seed
    inits.append(np.ones(N))
    
    # 2. Random Gaussian
    for _ in range(5):
        h_rand = rng.normal(size=N)
        inits.append(h_rand)
        
    # 3. Sine waves with DC
    for k in range(1, 5):
        x = np.arange(N)
        h_sin = np.sin(2 * np.pi * k * x / N)
        # Add DC to make it look like a step function? 
        # Actually, just the sine is a good start for NM to find side-lobe reductions.
        inits.append(h_sin)
        
    # 4. Box functions (positive center, negative edges)
    for width in [N//4, N//2, 3*N//4]:
        h_box = np.ones(N)
        h_box[:width//2] = -1
        h_box[N-width//2:] = -1
        inits.append(h_box)
        
    # 5. Random signs (Barker-like)
    for _ in range(5):
        h_sign = rng.choice([-1, 1], size=N)
        inits.append(h_sign)
        
    # 6. Linear ramp
    inits.append(np.linspace(-1, 1, N))

    best_obj = np.inf
    best_h_unconstrained = None
    
    for x0 in inits:
        try:
            # Nelder-Mead is derivative-free and handles non-smooth max well
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            h_opt = project(res.x)
            val = 2 * np.max(np.convolve(h_opt, h_opt)) / N
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
                
        except Exception:
            pass
            
    return best_h
