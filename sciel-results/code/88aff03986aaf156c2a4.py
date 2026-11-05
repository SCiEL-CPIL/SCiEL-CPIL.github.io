"""
Combined approach: complex exponential parameterization with inverse-FFT flat-spectrum mask,
refined by L-BFGS-B on a smooth log-sum-exp surrogate with FFT-accelerated gradients.
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
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # --- Fast convolution via FFT ---
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute for each i in [0, N-1]: sum_{m=0}^{N-1} weights[i+m] * h[m]
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        fft_size_cc = 1
        while fft_size_cc < N + 2 * N - 1:
            fft_size_cc *= 2
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        return conv_result[N - 1:2 * N - 1]

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = fast_convolve(h)
        max_c = np.max(conv)
        val = abs(2.0 * N * max_c / (s * s))
        return val

    def smooth_objective_and_grad(x, beta=60.0):
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
        grad_lse = 2.0 * fast_cross_corr(weights, h)

        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return f, grad

    def smooth_objective(x, beta=60.0):
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=60.0):
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    # --- Complex exponential parameterization with flat-spectrum mask ---
    def complex_exp_to_h(theta, null_depth=1.0, curvature=0.0):
        """
        Convert parameters to step heights using complex exponential with
        a flat-spectrum mask that has a deep null at zero frequency.
        
        theta: array of phase values
        null_depth: depth of the null at DC (0 to 1, 1 = full null)
        curvature: quadratic phase curvature parameter
        """
        k = np.arange(N)
        # Create phase with optional curvature
        phi = theta + curvature * k * k
        
        # Create the complex exponential sequence
        z = np.exp(1j * phi)
        
        # Take real part
        h = np.real(z)
        
        # Apply flat-spectrum mask via inverse FFT
        # Compute FFT of h
        H = np.fft.fft(h)
        
        # Apply mask: reduce DC component based on null_depth
        # The mask is flat (all ones) except at zero frequency where it's (1 - null_depth)
        mask = np.ones(N)
        mask[0] = 1.0 - null_depth
        
        # Apply mask in frequency domain
        H_masked = H * mask
        
        # Inverse FFT
        h_masked = np.real(np.fft.ifft(H_masked))
        
        return h_masked

    # --- Generate diverse initial points ---
    rng = np.random.default_rng(42)
    initial_points = []

    # Seed: flat sequence
    initial_points.append(np.ones(N).astype(float))

    # Complex exponential with various parameters
    for null_depth in [0.3, 0.5, 0.7, 0.9, 0.95]:
        for curvature in [-0.01, -0.005, 0.0, 0.005, 0.01]:
            for _ in range(3):
                theta = rng.uniform(0, 2 * np.pi, N)
                h = complex_exp_to_h(theta, null_depth, curvature)
                if abs(np.sum(h)) > 1e-10:
                    initial_points.append(h)

    # Random uniform
    for _ in range(10):
        h = rng.uniform(-1, 1, N)
        if abs(np.sum(h)) > 1e-10:
            initial_points.append(h)

    # Bimodal with random spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        if abs(np.sum(h)) > 1e-10:
            initial_points.append(h)

    # Low-frequency cosines with DC offset
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        for dc in [-2, -1, 0, 1, 2]:
            h = dc + np.cos(2 * np.pi * freq * t)
            if abs(np.sum(h)) > 1e-10:
                initial_points.append(h)

    # Gaussian bumps
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [20, 40, 60, 80, 120]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            for dc in [-1, 0, 1]:
                h = dc + h_bump
                if abs(np.sum(h)) > 1e-10:
                    initial_points.append(h)
                h_neg = dc - h_bump
                if abs(np.sum(h_neg)) > 1e-10:
                    initial_points.append(h_neg)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(3):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            if abs(np.sum(h_fourier)) > 1e-10:
                initial_points.append(h_fourier)

    # Two-level sequences
    for p in [0.3, 0.5, 0.7]:
        n_pos = int(N * p)
        for a in [-2, -1, 1, 2]:
            for b in [-2, -1, 1, 2]:
                h = np.empty(N)
                h[:n_pos] = a
                h[n_pos:] = b
                if abs(np.sum(h)) > 1e-10:
                    initial_points.append(h)

    # Sort by initial value (best first)
    def sort_key(h):
        return evaluate(h)

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # Evaluate seed
    seed_val = evaluate(best_h)
    best_value = seed_val

    for init_h in initial_points:
        if elapsed() > DEADLINE:
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
                options={'maxiter': 200, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            if elapsed() > DEADLINE:
                break

            h_opt = result.x
            val_opt = evaluate(h_opt)
            
            if val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue

    # Final exact evaluation
    final_val = evaluate(best_h)
    if final_val < best_value:
        best_value = final_val

    return best_h
