"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
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
    
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x, beta=80.0):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = np.convolve(h, h)
        max_c = np.max(conv)
        shifted = conv - max_c
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        # Use FFT for efficiency
        # We need cross-correlation of weights with h
        # grad_lse[i] = 2 * (weights * h)[i] where * is cross-correlation
        # Actually: sum_m weights[m+i] * h[m] for m=0..N-1
        # This is the cross-correlation of weights and h at lag i
        
        # Pad weights to length 2N-1 (already is)
        # We want for each i in [0,N-1]: sum_{m=0}^{N-1} weights[m+i] * h[m]
        # Let's compute this as a sliding dot product
        # Using FFT: if we define a = weights, b = h reversed
        # Actually, let's just do it with a loop since N=400 is small
        # But let's try to vectorize
        
        # For each i, we need dot(weights[i:i+N], h)
        # This is like a correlation
        # Let's use np.correlate
        # np.correlate(weights, h, mode='valid') gives sum over valid overlaps
        # But we need sum_{m=0}^{N-1} weights[m+i] * h[m] for i=0..N-1
        # This is equivalent to correlating weights with h
        
        # Actually: let a = weights (length 2N-1), b = h (length N)
        # We want c[i] = sum_{m=0}^{N-1} a[m+i] * b[m] for i=0..N-1
        # This is the cross-correlation of a and b, but shifted
        # np.correlate(a, b, 'full') gives the full cross-correlation
        # The result has length (2N-1) + N - 1 = 3N-2
        # c[i] corresponds to a specific index in the full correlation
        
        # Let me just compute it directly - N=400, so 400 dots of length 400 = 160K ops, fast
        grad_lse = np.zeros(N)
        for i in range(N):
            grad_lse[i] = 2.0 * np.dot(weights[i:i+N], h)
        
        lse = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        f = 2.0 * N * lse / (s * s)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad
    
    rng = np.random.RandomState(123)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(20):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(20):
        h = np.zeros(N)
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.2 * rng.normal(0, 1, N))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [10, 20, 30, 50, 75, 100, 150, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump with opposite signs
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.5 * np.ones(N)
            initial_points.append(h_off)
            # Two bumps same sign
            h_two = np.exp(-((i_arr - center)**2) / (2 * width**2)) + np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_two)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 8, 12]:
        for _ in range(8):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [10, 20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(5):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)
    
    # Smooth random via Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    for _ in range(15):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    # Sort initial points by their initial value to prioritize good starts
    init_values = []
    for init_h in initial_points:
        val_init, valid = evaluate(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 60 initial points for optimization
    top_inits = [h for _, h in init_values[:60]]
    
    # Also include some random ones for diversity
    for h in initial_points[60:80]:
        top_inits.append(h)
    
    for init_h in top_inits:
        if time.time() - start_time > DEADLINE:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize with different betas
        for beta in [50.0, 80.0, 120.0]:
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                result = minimize(
                    lambda x: smooth_objective(x, beta),
                    init_h,
                    jac=lambda x: smooth_grad(x, beta),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-6}
                )
                
                h_opt = result.x
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt.copy()
                    if best_value < 0.9:
                        break
            except Exception:
                pass
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 10:
        h_curr = best_h.copy()
        lr = 0.1
        improved = True
        max_rounds = 10
        for round_num in range(max_rounds):
            if not improved:
                lr *= 0.6
                if lr < 0.001:
                    break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 5:
                    break
                for delta in [lr, -lr, lr*0.5, -lr*0.5, lr*0.25, -lr*0.25, lr*2, -lr*2]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
    
    # --- Final: try a few more random perturbations of the best solution ---
    if time.time() - start_time < DEADLINE - 3:
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 3:
                break
            h_perturb = best_h + rng.normal(0, 0.05, N)
            val_p, valid_p = evaluate(h_perturb)
            if valid_p and val_p < best_value:
                best_value = val_p
                best_h = h_perturb.copy()
    
    return best_h
# EVOLVE-BLOCK-END
