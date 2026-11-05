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
    
    Strategy:
    1. Generate diverse initial points (random, structured, bump profiles).
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B with FFT-accelerated analytic gradient.
    3. Final exact evaluation and polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    # FFT size for fast convolution/correlation
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    fft_size = max(fft_size, 1024)
    
    def fast_convolve(h):
        """Compute full autoconvolution using FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2*N-1]
    
    def fast_grad_lse(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        using FFT-based cross-correlation for O(N log N) speed.
        weights has length 2N-1, h has length N.
        
        We want: g[i] = sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        
        This is a valid-mode correlation. Using FFT:
        Pad weights and h to size >= len(weights)+len(h)-1 = 3N-2
        corr_full = ifft(fft(weights_padded) * conj(fft(h_padded)))
        The valid correlation is at indices [N-1 : 2N-2] of corr_full
        """
        # Size for FFT: need at least len(weights) + len(h) - 1 = (2N-1) + N - 1 = 3N-2
        fft_size_corr = 1
        while fft_size_corr < (2*N - 1) + N - 1:
            fft_size_corr *= 2
        
        W = np.fft.rfft(weights, n=fft_size_corr)
        H = np.fft.rfft(h, n=fft_size_corr)
        
        # Cross-correlation: ifft(fft(a) * conj(fft(b))) gives correlation
        # corr[k] = sum_n a[n+k] * b[n] (for the valid part)
        corr_full = np.fft.irfft(W * np.conj(H), n=fft_size_corr)
        
        # The valid correlation starts at index N-1 and has N elements
        # Actually, let me verify the indexing:
        # np.correlate(a, b, mode='valid') where len(a) >= len(b):
        # result[k] = sum_{m=0}^{len(b)-1} a[m+k] * b[m]
        # For our case: a=weights (len 2N-1), b=h (len N)
        # result[k] = sum_{m=0}^{N-1} weights[m+k] * h[m] for k=0..N-1
        #
        # FFT correlation: ifft(fft(a) * conj(fft(b))) gives:
        # c[n] = sum_m a[m] * b[m-n] (circular, but with zero-padding it's linear)
        # Actually the standard formula: corr = ifft(conj(fft(a)) * fft(b))
        # Let me use the standard approach:
        # cross_corr = ifft(fft(a) * conj(fft(b)))
        # This gives: cross_corr[k] = sum_n a[n] * conj(b[n-k])
        # For real signals: cross_corr[k] = sum_n a[n] * b[n-k]
        # 
        # We want: g[i] = sum_m weights[i+m] * h[m]
        # Let's substitute: let j = i+m, so m = j-i
        # g[i] = sum_{j=i}^{i+N-1} weights[j] * h[j-i]
        # This is the cross-correlation of weights and h at lag i
        
        # Using: ifft(fft(weights) * conj(fft(h)))
        # Result at index k: sum_n weights[n] * h[n-k] (with appropriate padding)
        # We want sum_n weights[n] * h[n-(N-1-i)] ... this is getting confusing.
        
        # Let me just use the direct FFT formula that's known to work:
        # For cross-correlation where we want sum_{m=0}^{M-1} a[m+k] * b[m]:
        # Pad a to size P, pad b to size P (P >= len(a)+len(b)-1)
        # result = ifft(fft(a) * conj(fft(b)))
        # The valid part is at indices [len(b)-1 : len(b)-1+M] where M = len(a)-len(b)+1
        
        # Actually, let me just test with a simple case and use the correct formula.
        # The safest approach: use the known result that
        # np.correlate(a, b, 'valid') = ifft(fft(a_padded) * conj(fft(b_padded)))[len(b)-1 : len(b)-1+len(a)-len(b)+1]
        
        # Let's compute it properly:
        # a = weights, b = h
        # len(a) = 2N-1, len(b) = N
        # valid output length = len(a) - len(b) + 1 = N
        # 
        # The FFT-based cross-correlation:
        # C = ifft(fft(a) * conj(fft(b)))
        # C[k] = sum_{n=0}^{P-1} a[n] * b[(n-k) mod P]
        # With zero-padding, for k in range, this equals the linear cross-correlation.
        # 
        # The valid correlation is:
        # g[i] = sum_{m=0}^{N-1} a[i+m] * b[m]
        # = sum_{m=0}^{N-1} a[m+(N-1-i)] * b[m]  ... no
        # 
        # Let me just use a different approach: reverse h and do convolution
        # g[i] = sum_m weights[i+m] * h[m] = convolve(weights, h_reversed)[i + N - 1]
        # where h_reversed = h[::-1]
        
        h_rev = h[::-1]
        W = np.fft.rfft(weights, n=fft_size_corr)
        Hrev = np.fft.rfft(h_rev, n=fft_size_corr)
        conv_full = np.fft.irfft(W * Hrev, n=fft_size_corr)
        
        # conv_full[k] = sum_n weights[n] * h_rev[k-n] = sum_n weights[n] * h[N-1-(k-n)]
        # = sum_n weights[n] * h[N-1-k+n]
        # We want: g[i] = sum_m weights[i+m] * h[m]
        # Let n = i+m, so m = n-i:
        # g[i] = sum_{n=i}^{i+N-1} weights[n] * h[n-i]
        # From conv: conv_full[k] = sum_n weights[n] * h[N-1-k+n]
        # Set N-1-k+n = n-i => N-1-k = -i => k = N-1+i
        # So g[i] = conv_full[N-1+i]
        
        g = conv_full[N-1 : 2*N-1]
        return 2.0 * g
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    def smooth_objective(x):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        max_c = np.max(conv)
        beta = 50.0
        shifted = conv - max_c
        # Numerically stable log-sum-exp
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val
    
    def smooth_grad(x):
        """Analytic gradient of the smooth surrogate, FFT-accelerated."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        beta = 50.0
        shifted = conv - max_c
        exp_vals = np.exp(beta * shifted)
        weights = exp_vals / np.sum(exp_vals)
        
        # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        grad_lse = fast_grad_lse(weights, h)
        
        # lse value
        lse = max_c + np.log(np.sum(exp_vals)) / beta
        f = 2.0 * N * lse / (s * s)
        
        # df_dh[i] = 2*N*grad_lse[i]/s^2 - 4*N*lse/s^3
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return grad
    
    rng = np.random.RandomState(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))
    
    # Parameterized bump profiles (shown to work well)
    i_arr = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            # Double bump
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
            # Bump with offset
            h_off = np.exp(-((i_arr - center)**2) / (2 * width**2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)
    
    # Random Fourier synthesis with sine waves (spectral construction)
    for n_comp in [3, 5, 7, 10, 15]:
        for _ in range(8):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.3, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.sin(2 * np.pi * f * t + p)
            # Add small DC offset to ensure non-zero sum
            if abs(np.sum(h_fourier)) < 0.1:
                h_fourier += 0.05
            initial_points.append(h_fourier)
    
    # Mixed sine/cosine spectral constructions
    for n_comp in [5, 10, 15]:
        for _ in range(5):
            freqs = rng.randint(1, 50, n_comp)
            h_spec = np.zeros(N)
            for f in freqs:
                a_s = rng.uniform(0.2, 1.5)
                a_c = rng.uniform(0.2, 1.5)
                p_s = rng.uniform(0, 2 * np.pi)
                p_c = rng.uniform(0, 2 * np.pi)
                h_spec += a_s * np.sin(2 * np.pi * f * t + p_s)
                h_spec += a_c * np.cos(2 * np.pi * f * t + p_c)
            if abs(np.sum(h_spec)) < 0.1:
                h_spec += 0.03
            initial_points.append(h_spec)
    
    # Anti-symmetric patterns (zero-sum by construction)
    for width in [20, 40, 60, 80, 100]:
        h_anti = np.zeros(N)
        h_anti[:N//2] = 1.0
        h_anti[N//2:] = -1.0
        # Smooth transition
        from scipy.ndimage import gaussian_filter1d
        h_smooth = gaussian_filter1d(h_anti, sigma=width/4.0)
        initial_points.append(h_smooth)
        initial_points.append(-h_smooth)
    
    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)
        
        # Try to optimize with more iterations now that gradient is fast
        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-6}
            )
            
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        
        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()
    
    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.1
        deltas = [lr, -lr, lr*0.5, -lr*0.5, lr*0.25, -lr*0.25, lr*0.125, -lr*0.125]
        improved = True
        polish_iter = 0
        while improved and time.time() - start_time < DEADLINE - 2 and polish_iter < 20:
            improved = False
            polish_iter += 1
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in deltas:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.6
                deltas = [lr, -lr, lr*0.5, -lr*0.5, lr*0.25, -lr*0.25, lr*0.125, -lr*0.125]
                if lr < 1e-5:
                    break
    
    return best_h
# EVOLVE-BLOCK-END
