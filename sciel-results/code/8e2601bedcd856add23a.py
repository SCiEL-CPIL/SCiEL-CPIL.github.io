import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 135.0

def _autocorr_fft(x):
    """Compute aperiodic autocorrelation using FFT."""
    n = len(x)
    n_fft = 2 * n
    X = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(X * np.conj(X), n=n_fft)
    return acf[:n]

def _soft_max_sidelobe(acf, beta):
    """Compute soft max of absolute sidelobes."""
    if len(acf) <= 1:
        return 0.0
    sidelobes = np.abs(acf[1:])
    max_val = np.max(sidelobes)
    if max_val == 0:
        return 0.0
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _objective_and_grad(x, beta):
    """Compute soft-max objective and gradient."""
    acf = _autocorr_fft(x)
    obj = _soft_max_sidelobe(acf, beta)
    
    if len(acf) <= 1:
        return obj, np.zeros_like(x)
    
    sidelobes_abs = np.abs(acf[1:])
    max_val = np.max(sidelobes_abs)
    if max_val == 0:
        return obj, np.zeros_like(x)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    n = len(x)
    grad = np.zeros(n)
    
    # Gradient of sum_k w_k * |R(k)|
    # d/dx_j |R(k)| = sign(R(k)) * (x_{j-k} + x_{j+k})
    for k in range(1, n):
        w = weights[k-1]
        if w < 1e-10:
            continue
        sign_k = np.sign(acf[k])
        if sign_k == 0:
            continue
        # Contribution to grad
        # R(k) = sum_{i=0}^{n-1-k} x_i x_{i+k}
        # d R(k) / d x_j = x_{j-k} (if j>=k) + x_{j+k} (if j+k<n)
        # Vectorized update:
        # grad[0:n-k] += w * sign_k * x[k:n]  (from x_{j+k} term where i=j)
        # grad[k:n]   += w * sign_k * x[0:n-k] (from x_{j-k} term where i+k=j)
        
        grad[:n-k] += w * sign_k * x[k:]
        grad[k:]   += w * sign_k * x[:n-k]
        
    return obj, grad

def _local_search(f, max_iter=5000):
    """Local search via bit flips with efficient delta updates."""
    n = len(f)
    f = f.copy()
    acf = _autocorr_fft(f)
    max_sidelobe = np.max(np.abs(acf[1:]))
    
    # Precompute indices for fast access
    # We iterate through positions and check if flipping improves max sidelobe
    # To be faster, we can check only the most critical lags first, but full scan is safer.
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for j in range(n):
            # Calculate delta in acf if we flip f[j]
            # f_new[j] = -f[j]
            # Delta R(k) = -2 * f[j] * (f[j-k] + f[j+k])
            # We need to update acf and check if new max is lower
            
            # Optimization: Only consider flipping if it might reduce the current max.
            # A simple heuristic: try flipping bits that contribute most to the max lags.
            # However, a full scan is O(N^2) per iteration if we recompute max.
            # With delta update, we can update acf in O(N) and find max in O(N).
            # Total per flip: O(N). Total per pass: O(N^2) = 160,000 ops.
            # 5000 iters * 160,000 = 800M ops. Might be too slow in Python.
            
            # Let's limit local search iterations or use a smarter selection.
            # We'll just do a limited number of passes.
            
            # Compute deltas
            # d_acf[k] for k=1..n-1
            # d_acf[k] = -2 * f[j] * ( (f[j-k] if j>=k else 0) + (f[j+k] if j+k<n else 0) )
            
            # This vectorized approach is tricky for arbitrary j.
            # Let's stick to a simpler, slightly slower local search if needed, 
            # or just do a few passes of random flips + gradient step.
            
            # Actually, for N=400, a pure Python loop of 400 flips per pass, 
            # each taking O(N) to update acf and O(N) to find max, is 400*800 = 320k ops per pass.
            # 100 passes = 32M ops. This is fine.
            
            # Update acf for flip at j
            d_acf = np.zeros(n)
            # Indices for x[j-k]
            # k ranges 1 to n-1
            # For a fixed j, the terms involving f[j] are:
            # R(k) includes f[j]*f[j+k] if j+k < n
            # R(k) includes f[j-k]*f[j] if j-k >= 0
            
            # Let's just recompute acf for simplicity if N is small? No, N=400 is borderline.
            # But _autocorr_fft is O(N log N). 400 * 10 = 4000 ops.
            # 400 flips * 4000 ops = 1.6M ops per pass.
            # 100 passes = 160M ops. This is acceptable in Python if optimized.
            
            # Let's try a standard local search:
            # Try flipping each bit, check if max sidelobe decreases.
            # To speed up, we can use the fact that we only care about the max.
            
            # We will implement a simple version:
            # For each j, flip, compute new max, if better, keep, else revert.
            
            old_val = f[j]
            f[j] = -old_val
            new_acf = _autocorr_fft(f)
            new_max = np.max(np.abs(new_acf[1:]))
            
            if new_max < max_sidelobe:
                max_sidelobe = new_max
                acf = new_acf
                improved = True
            else:
                f[j] = old_val
                
    return f, max_sidelobe

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Start with random continuous values
    x = np.random.randn(N)
    
    best_f = None
    best_obj = float('inf')
    
    # Phase 1: Continuous Gradient Descent
    # Anneal beta from low to high
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    lrs = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    for i, beta in enumerate(betas):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        lr = lrs[i]
        # Run gradient descent steps
        for step in range(200):
            if time.time() - start_time > DEADLINE - 30:
                break
                
            obj, grad = _objective_and_grad(x, beta)
            
            # Normalize gradient to prevent explosion
            g_norm = np.linalg.norm(grad)
            if g_norm > 0:
                grad = grad / g_norm
            
            x = x - lr * grad
            
            # Optional: Project to [-1, 1] to keep values bounded
            x = np.clip(x, -1.0, 1.0)
            
            # Check binary projection periodically
            if step % 20 == 0:
                f_bin = np.sign(x)
                # Ensure no zeros
                f_bin[f_bin == 0] = 1.0
                obj_bin, _ = _objective_and_grad(f_bin, beta)
                if obj_bin < best_obj:
                    best_obj = obj_bin
                    best_f = f_bin.copy()
                    
        # Project at end of beta stage
        f_bin = np.sign(x)
        f_bin[f_bin == 0] = 1.0
        obj_bin, _ = _objective_and_grad(f_bin, beta)
        if obj_bin < best_obj:
            best_obj = obj_bin
            best_f = f_bin.copy()
            
    if best_f is None:
        best_f = np.sign(x)
        best_f[best_f == 0] = 1.0
        
    # Phase 2: Local Search
    if time.time() - start_time < DEADLINE - 10:
        # Run local search on the best candidate
        f_ls, obj_ls = _local_search(best_f, max_iter=50)
        if obj_ls < best_obj:
            best_f = f_ls
            best_obj = obj_ls
            
        # Try a few random restarts in local search if time permits
        # But local search is the bottleneck. Let's just do one good pass.
        
    return best_f
# EVOLVE-BLOCK-END
