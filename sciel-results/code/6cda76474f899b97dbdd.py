"""
Optimized autocorrelation inequality: minimize |2*N*max(h*h)/sum(h)^2|
Strategy: Direct time-domain optimization with FFT-accelerated convolution,
log-sum-exp surrogate for max, spectral-null initial points, and coordinate polish.
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
    DEADLINE = 140  # 10s safety margin before 150s kill

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

    # --- Smooth surrogate with log-sum-exp for max ---
    def smooth_objective(x, beta=80.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        # Use log-sum-exp on conv (signed) to approximate max(conv)
        max_c = np.max(conv)
        shifted = beta * (conv - max_c)
        exp_vals = np.exp(shifted)
        log_sum_exp = max_c + np.log(np.sum(exp_vals)) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    def smooth_grad(x, beta=80.0):
        """Analytic gradient of the smooth surrogate."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)

        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = beta * (conv - max_c)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp

        # Gradient of log-sum-exp conv w.r.t. h
        # conv_k = sum_m h_m h_{k-m}
        # d(conv_k)/dh_i = h_{k-i} + h_{k-i} if valid = 2*h_{k-i}
        # grad_lse[i] = sum_k weights[k] * 2 * h[k-i] (if 0<=k-i<N)
        # This is a valid-mode correlation of (2*weights) with h

        w2 = 2.0 * weights

        # Size for FFT: need at least (2N-1) + N - 1 = 3N-2
        fft_size_corr = 1
        while fft_size_corr < (2*N - 1) + N - 1:
            fft_size_corr *= 2

        h_rev = h[::-1]
        W = np.fft.rfft(w2, n=fft_size_corr)
        Hrev = np.fft.rfft(h_rev, n=fft_size_corr)
        conv_full = np.fft.irfft(W * Hrev, n=fft_size_corr)

        # g[i] = conv_full[N-1+i]
        g = conv_full[N-1 : 2*N-1]
        grad_lse = g

        # lse value
        lse = max_c + np.log(sum_exp) / beta
        f = 2.0 * N * lse / (s * s)

        # df_dh[i] = 2*N*grad_lse[i]/s^2 - 4*N*lse/s^3
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return grad

    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, N, endpoint=False)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Spectral-null initial points: DC-removed random vectors
    for _ in range(30):
        h = rng.normal(0, 1, N)
        h -= np.mean(h)  # Remove DC component
        if abs(np.sum(h)) < 0.1:
            h += 0.05
        initial_points.append(h)

    # Fourier-domain nulls: zero out low frequencies
    for _ in range(20):
        H = np.fft.rfft(rng.normal(0, 1, N))
        H[0:5] = 0  # Null out lowest frequencies
        h = np.fft.irfft(H, n=N)
        if abs(np.sum(h)) < 0.1:
            h += 0.05
        initial_points.append(h)

    # Random uniform
    for _ in range(20):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(20):
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
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10, 15, 20]:
        for _ in range(10):
            freqs = rng.randint(1, 60, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.3, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.sin(2 * np.pi * f * t + p)
            if abs(np.sum(h_fourier)) < 0.1:
                h_fourier += 0.05
            initial_points.append(h_fourier)

    # Mixed sine/cosine spectral constructions
    for n_comp in [5, 10, 15, 20]:
        for _ in range(8):
            freqs = rng.randint(1, 80, n_comp)
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

    # Anti-symmetric patterns
    from scipy.ndimage import gaussian_filter1d
    for width in [20, 40, 60, 80, 100]:
        h_anti = np.zeros(N)
        h_anti[:N//2] = 1.0
        h_anti[N//2:] = -1.0
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

        # Try different beta values for sharper/smoother surrogate
        for beta in [50.0, 80.0, 150.0]:
            if time.time() - start_time > DEADLINE:
                break

            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    init_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-6}
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
        while improved and time.time() - start_time < DEADLINE - 2 and polish_iter < 15:
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
