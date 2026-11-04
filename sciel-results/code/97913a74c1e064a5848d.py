"""Combined program for the third autocorrelation inequality task.
Uses L-BFGS-B optimization with diverse initializations and Nelder-Mead polishing.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _value(h):
    """Compute the evaluator's value: abs(2*N*max(convolve(h,h))/sum(h)^2)."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial best is the seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Helper to update best
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        v = _value(h_candidate)
        if v < best_val:
            best_val = v
            best_h = h_candidate.copy()
    
    # List of initializations to try
    inits = []
    
    # 1. Seed
    inits.append(np.ones(N))
    
    # 2. Structured Patterns
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    inits.append(h_alt)
    
    # Block (half pos, half neg)
    h_block = np.ones(N)
    h_block[N//2:] = -1.0
    inits.append(h_block)
    
    # Two blocks (middle neg)
    h_2block = np.ones(N)
    h_2block[N//4:3*N//4] = -1.0
    inits.append(h_2block)
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, N)
    inits.append(np.cos(x))
    inits.append(np.sin(x))
    
    # 3. Random Initializations
    np.random.seed(42)
    for i in range(10):
        # Mix of Gaussian and Uniform
        if i % 2 == 0:
            h_rand = np.random.randn(N)
        else:
            h_rand = np.random.uniform(-1, 1, N)
        # Ensure sum is not zero
        if abs(np.sum(h_rand)) < 1e-5:
            h_rand[0] += 1.0
        inits.append(h_rand)
        
    # 4. Sparse Initializations
    for i in range(5):
        h_sparse = np.zeros(N)
        num_non_zero = N // 4
        idx = np.random.choice(N, num_non_zero, replace=False)
        h_sparse[idx] = np.random.choice([-1, 1], num_non_zero)
        # Ensure sum is not zero
        if abs(np.sum(h_sparse)) < 1e-5:
            h_sparse[idx[0]] += 1.0
        inits.append(h_sparse)

    # Optimization Loop
    # We will run L-BFGS-B on each initialization
    # Time budget per init: (deadline - current_time) / remaining_inits
    
    num_inits = len(inits)
    elapsed = time.time() - start_time
    time_per_init = (deadline - elapsed - 10.0) / max(1, num_inits) # 10s buffer for final checks
    
    for i, h_init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        # Update best with init itself
        try_update(h_init)
        
        # Run L-BFGS-B
        def objective(x):
            return _value(x)
            
        try:
            # L-BFGS-B with numerical gradient
            # We limit iterations to ensure we don't spend too long on one
            max_iter = max(10, int(time_per_init * 10)) # Rough heuristic
            res = minimize(objective, h_init, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'gtol': 1e-6, 'disp': False})
            if time.time() > deadline - 15.0:
                break
            try_update(res.x)
            
            # Polish with Nelder-Mead if time permits
            if time.time() < deadline - 20.0:
                # Nelder-Mead is local and fast for fine-tuning
                # Reduce maxiter significantly
                nm_max_iter = max(10, int((deadline - time.time() - 10.0) / max(1, num_inits - i - 1) * 5))
                res_nm = minimize(objective, res.x, method='Nelder-Mead',
                                  options={'maxiter': nm_max_iter, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
                try_update(res_nm.x)
                
        except Exception:
            pass

    # Final safety check
    if time.time() > deadline - 2.0:
        return best_h if np.all(np.isfinite(best_h)) else np.ones(N)
    
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
