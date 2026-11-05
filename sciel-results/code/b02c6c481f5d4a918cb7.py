"""Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
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

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        fft_size_cc = 1
        while fft_size_cc < N + 2 * N - 1:
            fft_size_cc *= 2
            
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        
        # Extract the relevant part
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        
        return grad_lse

    def smooth_objective_and_grad(x, beta=80.0):
        """Compute both objective and gradient in one pass to avoid redundant FFTs."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
        
        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(exp_terms)
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)
        
        # Gradient
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f, grad

    def smooth_objective(x, beta=80.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=80.0):
        """Analytic gradient of the smooth surrogate."""
        _, grad = smooth_objective_and_grad(x, beta)
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
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
            h_off = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10]:
        for _ in range(5):
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
    for _ in range(15):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            continue

    # --- Polish phase with higher beta ---
    if time.time() - start_time < DEADLINE:
        for beta in [100, 150]:
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                result = minimize(
                    lambda x: smooth_objective(x, beta=beta),
                    best_h,
                    jac=lambda x: smooth_grad(x, beta=beta),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
                )
                
                val, valid = evaluate(result.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = result.x.copy()
            except Exception:
                continue

    # --- Final exact gradient polish ---
    if time.time() - start_time < DEADLINE:
        def exact_obj_grad(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10, np.zeros_like(h)
            
            conv = fast_convolve(h)
            max_val = np.max(conv)
            idx = np.argmax(conv)
            
            raw = 2.0 * N * max_val / (s * s)
            val = abs(raw)
            
            dh_conv = np.zeros(N)
            km = idx - np.arange(N)
            mask = (km >= 0) & (km < N)
            dh_conv[mask] = h[km[mask]] + h[mask]
            
            grad = 2.0 * N * (dh_conv * s - 2.0 * max_val) / (s ** 3)
            
            if raw < 0:
                grad = -grad
            
            return val, grad

        try:
            result = minimize(
                exact_obj_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            
            val, valid = evaluate(result.x)
            if valid and val < best_value:
                best_value = val
                best_h = result.x.copy()
        except Exception:
            pass

    # Verify result
    s = np.sum(best_h)
    if abs(s) < 1e-12:
        return np.ones(N)
    
    conv = fast_convolve(best_h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    
    if val >= 2.0:
        return np.ones(N)
    
    return best_h
