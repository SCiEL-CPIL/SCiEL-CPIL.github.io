"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Uses FFT-based gradient computation with a smooth log-sum-exp surrogate.
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
    DEADLINE = 143  # 7s safety margin before 150s kill

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

    # Precompute FFT sizes
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m]
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        fft_size_cc = 1
        while fft_size_cc < N + 2 * N - 1:
            fft_size_cc *= 2
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        return grad_lse

    def smooth_objective_and_grad(x, beta=100.0):
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
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        return f, grad

    def smooth_objective(x, beta=100.0):
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=100.0):
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    rng = np.random.RandomState(123)

    # Generate diverse initial points
    initial_points = []
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(12):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(12):
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
            h_double = h_bump - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
            h_off = h_bump + 0.3 * np.ones(N)
            initial_points.append(h_off)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10]:
        for _ in range(4):
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
        for _ in range(4):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)

    # Smooth random via Gaussian filter
    for _ in range(12):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # Sort by initial value
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # Optimize from each initial point
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
                options={'maxiter': 250, 'maxfun': 500, 'gtol': 1e-6}
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

    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                smooth_objective,
                best_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Perturbation phase
    if time.time() - start_time < DEADLINE - 3:
        for _ in range(6):
            if time.time() - start_time > DEADLINE - 3:
                break
            h_pert = best_h + 0.05 * rng.normal(0, 1, N)
            val_pert, valid = evaluate(h_pert)
            if valid and val_pert < best_value:
                best_value = val_pert
                best_h = h_pert.copy()
            else:
                try:
                    result = minimize(
                        smooth_objective,
                        h_pert,
                        jac=smooth_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-7}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
