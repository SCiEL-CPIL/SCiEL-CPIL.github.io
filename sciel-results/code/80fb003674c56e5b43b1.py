"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 142  # 8s safety margin before 150s kill

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
    fft_size = 1  # smallest power of 2 >= 2*N-1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        # We want: for each i in [0, N-1], sum_{m=0}^{N-1} weights[i+m] * h[m]
        # This is the cross-correlation of weights and h at specific lags.
        # cross_corr[k] = sum_j weights[j+k] * h[j] for k = -(N-1), ..., N-1
        # We need k = 0, 1, ..., N-1
        # Using FFT: cross_corr = ifft(fft(weights_padded) * conj(fft(h_padded)))
        # But we need to be careful with the indexing.
        
        # Alternative: pad weights to length 2*N-1, h to length 2*N-1
        # Then the linear cross-correlation at lag i is:
        # sum_{m=0}^{N-1} weights[i+m] * h[m]
        
        # Let's use the convolution approach:
        # Define w_rev[j] = weights[2*N-2-j] for j=0..2*N-2
        # Then conv(w_rev, h)[i + N - 1] = sum_{m=0}^{N-1} w_rev[i+N-1-m] * h[m]
        # = sum_{m=0}^{N-1} weights[2*N-2-(i+N-1-m)] * h[m]
        # = sum_{m=0}^{N-1} weights[N-1-i+m] * h[m]
        
        # That's not quite what we want. Let me think more carefully.
        
        # We want: G[i] = sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        # This is a "sliding dot product" where the window over weights slides.
        
        # Using FFT: Let's compute the full cross-correlation.
        # cross_corr_full = ifft(fft(weights, n=4*N) * conj(fft(h, n=4*N)))
        # Then cross_corr_full[k] = sum_j weights[j] * conj(h[j-k]) for appropriate k
        # For real signals: sum_j weights[j] * h[j-k]
        
        # Actually, let's just use the direct matrix approach but with a smarter construction.
        # For N=400, building the 400x400 matrix takes ~1ms, which is acceptable.
        # The FFT approach has overhead from padding and the inverse FFT.
        
        # Let me try a different FFT approach:
        # G[i] = sum_{m=0}^{N-1} weights[i+m] * h[m]
        # = sum_{m=0}^{N-1} weights[i+m] * h[m] * 1_{0<=m<N}
        
        # If we define W[k] = weights[k] for k=0..2N-2, 0 otherwise
        # And H[k] = h[k] for k=0..N-1, 0 otherwise
        # Then the convolution (W * H)[i] = sum_k W[k] * H[i-k] = sum_k weights[k] * h[i-k]
        # We want sum_m weights[i+m] * h[m]
        # Let j = i+m, so m = j-i: sum_j weights[j] * h[j-i] for j=i..i+N-1
        # = sum_j weights[j] * h[j-i] * 1_{0<=j-i<N}
        
        # This is the cross-correlation at lag i of weights and h.
        # cross_corr[i] = sum_j weights[j+i] * h[j]
        
        # Using FFT: cross_corr = ifft(fft(weights_padded, n) * conj(fft(h_padded, n)))
        # where n >= len(weights) + len(h) - 1 = 3N-2
        # Actually for cross-correlation: ifft(fft(a) * conj(fft(b))) gives
        # sum_k a[k+n-j] * b[j] ... this is getting confusing.
        
        # Let me just use the matrix approach - it's simple and fast enough for N=400.
        idx = np.arange(N)
        M = weights[idx[:, None] + idx[None, :]]  # M[i,m] = weights[i+m]
        return M @ h

    def smooth_objective_and_grad(x, beta=60.0):
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
        grad_lse = 2.0 * fast_cross_corr(weights, h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return f, grad

    def smooth_objective(x, beta=60.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=60.0):
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

        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Final high-precision polish using L-BFGS-B with tighter tolerance ---
    if time.time() - start_time < DEADLINE - 3:
        try:
            result = minimize(
                smooth_objective,
                best_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # --- Perturbation phase with re-optimization ---
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 2:
                break
            h_pert = best_h + 0.05 * rng.normal(0, 1, N)
            val_pert, valid = evaluate(h_pert)
            if valid and val_pert < best_value:
                best_value = val_pert
                best_h = h_pert
            else:
                try:
                    result = minimize(
                        smooth_objective,
                        h_pert,
                        jac=smooth_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-7}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
