"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. LP-based initialization: solve a convex relaxation to get a good starting point.
2. Direct L-BFGS-B optimization with analytic gradients of smoothed max surrogate.
3. FFT-based spectral coordinate descent for final refinement.
"""

import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 600
TIME_LIMIT = 145  # safety margin before 150s kill


def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)


def compute_grad(f, beta=20.0):
    """
    Gradient of J(f) = smooth_max(autoconv(f)) / (sum(f))^2.
    """
    f = np.asarray(f, dtype=np.float64)
    s = np.sum(f)
    if s < 1e-15:
        return np.zeros_like(f)

    conv = np.convolve(f, f, mode='full')
    conv_max = np.max(conv)
    if conv_max < 1e-15:
        return np.zeros_like(f)

    # Smoothed max using log-sum-exp
    shifted = conv - conv_max
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp
    K = conv_max + (1.0 / beta) * np.log(sum_exp)

    # Gradient of K w.r.t. f[j]:
    # d(conv[m])/d(f[j]) = f[m-j] for valid indices
    # grad_K[j] = 2 * sum_m w[m] * f[m-j]
    # This is cross-correlation of w with f, truncated to length N.
    n_fft = 1 << (2 * N - 1).bit_length()
    w_hat = np.fft.rfft(w, n=n_fft)
    f_hat = np.fft.rfft(f, n=n_fft)
    grad_K = 2.0 * np.fft.irfft(np.conj(w_hat) * f_hat, n=n_fft)[:N]

    # J = K / s^2
    grad_J = grad_K / (s * s) - (2.0 * K / (s * s * s)) * np.ones(N)

    return grad_J


def lp_initialize():
    """
    Solve an LP to minimize the maximum autoconvolution value.
    Variables: f[0..N-1] >= 0, t >= 0
    Constraints: f[j] >= 0, sum(f) = N, and for all lags m: conv[m] <= t
    Since conv[m] = sum_j f[j]*f[m-j], this is quadratic, not linear.
    
    Instead, use a simpler approach: start with a smooth Gaussian-like function
    and refine it.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    
    # Start with a broad Gaussian
    f = np.exp(-8 * x ** 2)
    f = f / np.sum(f) * N
    return f


def spectral_coordinate_descent(h, start_time, max_time):
    """
    Spectral coordinate descent: work in frequency domain to reduce peak autoconvolution.
    """
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05

    n_fft = 1 << (2 * N - 1).bit_length()

    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)

        # Compute autoconvolution via FFT
        f_hat = np.fft.rfft(current_h, n=n_fft)
        conv = np.fft.irfft(f_hat * f_hat, n=n_fft)[:2 * N - 1]

        max_c = np.max(conv)
        current_val = max_c / (N * N)

        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

        # Find peak lags
        threshold = max_c * 0.95
        peak_mask = (conv >= threshold).astype(np.float64)

        # Sensitivity in frequency domain
        pm_hat = np.fft.rfft(peak_mask, n=n_fft)
        h_hat = np.fft.rfft(current_h, n=n_fft)
        sens = np.fft.irfft(pm_hat * np.conj(h_hat), n=n_fft)[:N]

        sens_mean = np.mean(sens)
        sens_std = np.std(sens) + 1e-12
        sens_norm = (sens - sens_mean) / sens_std

        # Reduce bins with high sensitivity
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)

        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)

        # Evaluate new
        f_hat_new = np.fft.rfft(h_new, n=n_fft)
        conv_new = np.fft.irfft(f_hat_new * f_hat_new, n=n_fft)[:2 * N - 1]
        val_new = np.max(conv_new) / (N * N)

        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.3)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break

    return best_h, best_val


def construct():
    start_time = time.time()
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N

    best_f = np.ones(N)
    best_score = compute_score(best_f)

    rng = np.random.RandomState(42)

    # Phase 1: Generate diverse initial candidates
    candidates = []

    # Gaussian
    f0 = np.exp(-15 * x ** 2)
    f0 = f0 / np.sum(f0)
    candidates.append(f0)

    # Cosine bump families
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        centers = rng.uniform(-0.2, 0.2, size=n_c)
        widths = rng.uniform(0.06, 0.25, size=n_c)
        heights = rng.uniform(0.5, 2.0, size=n_c)

        f_c = np.zeros_like(x)
        for c, w, h in zip(centers, widths, heights):
            dist = np.abs(x - c)
            mask = dist <= w / 2.0
            if np.any(mask):
                f_c[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
        f_c = np.maximum(f_c, 0.0)
        s = np.sum(f_c)
        if s > 1e-12:
            f_c = f_c / s
            candidates.append(f_c)

    # Phase 2: Direct L-BFGS-B optimization on each candidate
    for idx, f_init in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - 30:
            break

        def objective_and_grad(f, _beta=15.0):
            f_clipped = np.maximum(f, 0.0)
            s = np.sum(f_clipped)
            if s < 1e-15:
                return 1e10, np.zeros_like(f)
            conv = np.convolve(f_clipped, f_clipped, mode='full')
            val = np.max(conv) / (s * s)
            grad = compute_grad(f_clipped, beta=_beta)
            return val, grad

        try:
            result = minimize(
                objective_and_grad, f_init, jac=True,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-5, 'maxls': 20}
            )

            f_res = np.maximum(result.x, 0.0)
            s_res = np.sum(f_res)
            if s_res > 1e-12:
                f_res = f_res / s_res
                score_res = compute_score(f_res)
                if score_res < best_score:
                    best_score = score_res
                    best_f = f_res.copy()
        except Exception:
            pass

    # Phase 3: Spectral coordinate descent for final refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = spectral_coordinate_descent(
            best_f, start_time, TIME_LIMIT - 3
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]


if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
