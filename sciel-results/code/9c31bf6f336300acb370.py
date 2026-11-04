"""Parametric family search: optimize shape parameters of structured step functions."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy: Search over parametric families (Gaussian bumps, two-level blocks,
    multi-level structures) to find shapes where autocorrelation peak is small
    relative to squared sum. Then polish with local optimization.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def update_best(h):
        nonlocal best_value, best_h
        val, valid = evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()
        return val, valid
    
    # Parametric family 1: Gaussian bump
    # h[i] = a * exp(-((i-center)/sigma)^2) + b
    # Normalize so sum(h) = 1
    def make_gaussian(params):
        a, center, sigma, b = params
        idx = np.arange(N)
        h = a * np.exp(-((idx - center) / sigma) ** 2) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s * 1.0  # scale so sum = 1
        # Actually we want sum(h)=1, so just normalize
        # But the objective is scale-invariant, so any scale works
        # Let's keep sum(h)=1 for numerical stability
        return h / s
    
    # Parametric family 2: Two-block structure
    # h[i] = a for i in [0, k), b for i in [k, N)
    def make_two_block(params):
        a, b, k_frac = params
        k = int(k_frac * N)
        h = np.zeros(N)
        h[:k] = a
        h[k:] = b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s
    
    # Parametric family 3: Three-block (positive, negative, positive)
    def make_three_block(params):
        a, b, c, k1, k2 = params
        k1 = int(k1 * N)
        k2 = int(k2 * N)
        h = np.zeros(N)
        h[:k1] = a
        h[k1:k2] = b
        h[k2:] = c
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s
    
    # Parametric family 4: Cosine with negative baseline
    def make_cosine(params):
        a, freq, phase, b = params
        t = np.linspace(0, 1, N, endpoint=False)
        h = a * np.cos(2 * np.pi * freq * t + phase) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s
    
    # Parametric family 5: Triangle bump
    def make_triangle(params):
        a, center, width, b = params
        idx = np.arange(N)
        h = a * np.maximum(0, 1 - np.abs(idx - center) / width) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s
    
    # Parametric family 6: Multi-level with random pattern
    def make_multilevel(params, rng):
        a, b, n_blocks = params
        n_blocks = int(n_blocks)
        block_sizes = np.ones(n_blocks)
        h = np.zeros(N)
        pos = 0
        for i in range(n_blocks):
            size = max(1, int(block_sizes[i]))
            val = a if i % 2 == 0 else b
            h[pos:pos+size] = val
            pos += size
            if pos >= N:
                break
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s
    
    # Search over parametric families
    rng = np.random.RandomState(123)
    
    # Family 1: Gaussian bumps
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 5.0)
        center = rng.uniform(0.2 * N, 0.8 * N)
        sigma = rng.uniform(N/10, N/3)
        b = rng.uniform(-0.5, 0.5)
        h = make_gaussian([a, center, sigma, b])
        if h is not None:
            update_best(h)
    
    # Family 2: Two-block
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        k_frac = rng.uniform(0.1, 0.9)
        h = make_two_block([a, b, k_frac])
        if h is not None:
            update_best(h)
    
    # Family 3: Three-block
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        c = rng.uniform(0.5, 3.0)
        k1 = rng.uniform(0.1, 0.4)
        k2 = rng.uniform(0.5, 0.9)
        h = make_three_block([a, b, c, k1, k2])
        if h is not None:
            update_best(h)
    
    # Family 4: Cosine
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 3.0)
        freq = rng.choice([1, 2, 3, 4, 5, 7, 10])
        phase = rng.uniform(0, 2 * np.pi)
        b = rng.uniform(-1.0, 1.0)
        h = make_cosine([a, freq, phase, b])
        if h is not None:
            update_best(h)
    
    # Family 5: Triangle
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 5.0)
        center = rng.uniform(0.2 * N, 0.8 * N)
        width = rng.uniform(N/5, N/2)
        b = rng.uniform(-0.5, 0.5)
        h = make_triangle([a, center, width, b])
        if h is not None:
            update_best(h)
    
    # Family 6: Multi-level
    for trial in range(200):
        if time.time() - start_time > DEADLINE:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        n_blocks = rng.randint(4, 20)
        h = make_multilevel([a, b, n_blocks], rng)
        if h is not None:
            update_best(h)
    
    # Now polish the best result with local optimization
    if time.time() - start_time < DEADLINE - 10:
        h_curr = best_h.copy()
        
        # Use L-BFGS on the exact objective with smooth max surrogate
        def obj_and_grad(h, beta):
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10, np.zeros_like(h)
            
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            shifted = beta * (conv - max_conv)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            
            if sum_exp == 0:
                soft_max = max_conv
                weights = np.zeros_like(conv)
                weights[np.argmax(conv)] = 1.0
            else:
                soft_max = max_conv + np.log(sum_exp) / beta
                weights = exp_shifted / sum_exp
            
            f = 2.0 * N * soft_max / (s * s)
            
            # Gradient
            grad_soft_max = np.zeros(N)
            for k in range(N):
                grad_soft_max[k] = 2.0 * np.dot(weights[k:k+N], h)
            
            grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
            
            return f, grad_f
        
        for beta in [50, 200, 500, 1000]:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            try:
                result = minimize(
                    lambda h, b=beta: obj_and_grad(h, b)[0],
                    h_curr,
                    jac=lambda h, b=beta: obj_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < 1e10:
                    val, valid = evaluate(result.x)
                    if valid and val < best_value:
                        best_value = val
                        best_h = result.x.copy()
                        h_curr = result.x.copy()
                        
            except Exception:
                pass
        
        # Final coordinate descent polish
        if time.time() - start_time < DEADLINE - 2:
            h_curr = best_h.copy()
            lr = 0.05
            for iteration in range(50):
                if time.time() - start_time > DEADLINE - 2:
                    break
                improved = False
                for coord in rng.permutation(N):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    for delta in [lr, -lr, lr*0.5, -lr*0.5]:
                        h_try = h_curr.copy()
                        h_try[coord] += delta
                        val_try, valid = evaluate(h_try)
                        if valid and val_try < best_value:
                            best_value = val_try
                            best_h = h_try.copy()
                            h_curr = h_try
                            improved = True
                            break
                if not improved:
                    lr *= 0.5
                if lr < 1e-6:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
