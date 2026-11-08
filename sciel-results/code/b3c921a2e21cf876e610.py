"""Minimize peak autoconvolution of nonnegative step function on 600 intervals.

Strategy: The objective is scale-invariant, so we fix L2 norm = 1 and minimize
the peak of the autoconvolution. The peak autoconvolution is bounded below by
the L2 norm squared (at lag 0). We use a spectral approach combined with
iterative water-filling to push the autoconvolution flat.

The key insight: if the autoconvolution is flat at value c for all lags,
then the Fourier transform of h has |H(f)|^2 = c * (Fourier of rect) which
is a sinc^2 shape. So we want h whose Fourier transform magnitude is
proportional to |sinc|, i.e., h is proportional to |sinc| (with proper
nonnegativity constraints).

Actually, the autoconvolution in frequency domain: if F = FFT(h), then
FFT(autoconv) = |F|^2. We want |F|^2 to be as flat as possible (subject
to the constraint that autoconv is nonnegative and h is nonnegative).

A better approach: use the fact that minimizing max_k ac[k] subject to
||h||_2 = 1 and h >= 0 is equivalent to finding h such that its
autoconvolution is as flat as possible.

We'll use:
1. Start with a good initial guess (e.g., truncated Gaussian or raised cosine)
2. Use projected gradient descent with the subgradient of the max autoconvolution
3. Additionally try a "spectral flattening" approach: iteratively adjust h
   to make |FFT(h)|^2 flatter
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 600
DEADLINE = 140.0  # safety margin before 150s kill


def _autoconvolution_peak(h):
    """Compute the normalized peak autoconvolution score.
    
    Score = max_k (sum_i h[i]*h[i+k]) / (sum_i h[i]^2)
    This is scale-invariant. For uniform h, this equals 1.
    """
    n = len(h)
    # Use FFT for linear autoconvolution
    size = 1
    while size < 2 * n:
        size *= 2
    h_fft = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(h_fft * np.conj(h_fft), n=size)
    peak = np.max(ac[:n])
    total_sq = np.sum(h * h)
    if total_sq < 1e-15:
        return 1e10
    return peak / total_sq


def _autoconvolution_all(h):
    """Compute the full autoconvolution vector."""
    n = len(h)
    size = 1
    while size < 2 * n:
        size *= 2
    h_fft = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(h_fft * np.conj(h_fft), n=size)
    return ac[:n]


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def time_left():
        return DEADLINE - elapsed()

    # Initial candidates
    candidates = []

    # 1. Uniform
    h_uniform = np.ones(N) / np.sqrt(N)
    candidates.append(h_uniform.copy())

    # 2. Raised cosine (Hann window)
    x = np.linspace(0, 1, N, endpoint=False)
    h_hann = np.maximum(0.5 * (1 + np.cos(2 * np.pi * x)), 0)
    h_hann = h_hann / np.sqrt(np.sum(h_hann**2))
    candidates.append(h_hann.copy())

    # 3. Triangular
    x = np.linspace(0, 1, N, endpoint=False)
    h_tri = np.maximum(1 - 2 * np.abs(x - 0.5), 0)
    h_tri = h_tri / np.sqrt(np.sum(h_tri**2))
    candidates.append(h_tri.copy())

    # 4. Sinc-like (absolute sinc, truncated)
    k = np.arange(N)
    center = (N - 1) / 2.0
    h_sinc = np.abs(np.sinc((k - center) / (N * 0.3)))
    h_sinc = np.maximum(h_sinc, 0)
    if np.sum(h_sinc**2) > 1e-15:
        h_sinc = h_sinc / np.sqrt(np.sum(h_sinc**2))
        candidates.append(h_sinc.copy())

    # 5. Gaussian
    x = np.linspace(0, 1, N, endpoint=False)
    h_gauss = np.exp(-20 * (x - 0.5)**2)
    h_gauss = h_gauss / np.sqrt(np.sum(h_gauss**2))
    candidates.append(h_gauss.copy())

    # 6. Flat top with smooth edges
    h_flat = np.ones(N)
    edge = max(1, N // 10)
    ramp = np.linspace(0, 1, edge)
    h_flat[:edge] = ramp
    h_flat[-edge:] = ramp[::-1]
    h_flat = h_flat / np.sqrt(np.sum(h_flat**2))
    candidates.append(h_flat.copy())

    # Evaluate all candidates
    best_h = None
    best_score = 1e10
    for h in candidates:
        score = _autoconvolution_peak(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Gradient-based optimization using subgradient of max autoconvolution
    # We minimize f(h) = max_k ac_k(h) subject to ||h||_2 = 1, h >= 0
    # where ac_k(h) = sum_i h[i]*h[i+k]
    #
    # Subgradient of max_k ac_k with respect to h:
    # d(ac_k)/d(h[j]) = h[j-k] + h[j+k] (with boundary conditions)
    # The subgradient of max is the average of gradients at active indices
    # (or just the gradient at the argmax for simplicity)

    def _project_to_sphere_nonneg(h):
        """Project h onto {x: ||x||_2 = 1, x >= 0}."""
        h = np.maximum(h, 0)
        norm = np.sqrt(np.sum(h**2))
        if norm < 1e-15:
            return np.ones(N) / np.sqrt(N)
        return h / norm

    def _subgradient_step(h, lr):
        """One subgradient step on the sphere."""
        ac = _autoconvolution_all(h)
        # Find active set (top few lags)
        top_k = max(1, min(5, N))
        active = np.argsort(ac)[-top_k:]
        # Average gradient over active set
        grad = np.zeros(N)
        for k in active:
            # d(ac_k)/d(h[j]) = h[j-k] + h[j+k] (with bounds)
            for j in range(N):
                gjk = 0.0
                if j - k >= 0:
                    gjk += h[j - k]
                if j + k < N:
                    gjk += h[j + k]
                grad[j] += gjk
        grad /= top_k
        # Gradient step with projection
        h_new = h - lr * grad
        h_new = _project_to_sphere_nonneg(h_new)
        return h_new, ac

    # Faster subgradient using vectorized operations
    def _subgradient_step_fast(h, lr):
        """One subgradient step using FFT-based computation."""
        ac = _autoconvolution_all(h)
        top_k = max(1, min(10, N))
        active = np.argsort(ac)[-top_k:]

        # Compute gradient: grad[j] = sum_{k in active} (h[j-k] + h[j+k])
        # This is a convolution of h with a kernel that has 1s at positions in active
        kernel = np.zeros(2 * N - 1)
        for k in active:
            kernel[k] += 1  # h[j+k] contribution
            kernel[N - 1 + (N - 1 - k)] += 1  # This is getting complicated

        # Simpler: just loop over active k values
        grad = np.zeros(N)
        for k in active:
            # h[j+k] for j = 0..N-1-k
            grad[:N - k] += h[k:]
            # h[j-k] for j = k..N-1
            grad[k:] += h[:N - k]
        grad /= top_k

        h_new = h - lr * grad
        h_new = _project_to_sphere_nonneg(h_new)
        return h_new, ac

    # Run subgradient optimization
    if time_left() > 20:
        h_opt = best_h.copy()
        # Adaptive learning rate
        lr = 0.1
        best_iter_score = best_score
        best_iter_h = best_h.copy()
        no_improve = 0
        max_no_improve = 200

        while time_left() > 10 and no_improve < max_no_improve:
            h_new, ac = _subgradient_step_fast(h_opt, lr)
            score = np.max(ac) / np.sum(h_opt**2)  # h_opt is already unit norm
            # Actually recompute properly
            score = _autoconvolution_peak(h_new)
            if score < best_iter_score:
                best_iter_score = score
                best_iter_h = h_new.copy()
                h_opt = h_new.copy()
                no_improve = 0
                # Decrease learning rate slightly
                lr *= 0.99
            else:
                no_improve += 1
                lr *= 0.95
                if lr < 0.001:
                    lr = 0.001

        if best_iter_score < best_score:
            best_score = best_iter_score
            best_h = best_iter_h.copy()

    # Spectral flattening approach
    # Idea: iteratively adjust h so that |FFT(h)|^2 is flatter
    # This is related to the "flat spectrum" problem
    if time_left() > 15:
        h_spec = best_h.copy()
        for iteration in range(200):
            if time_left() < 5:
                break
            # Compute current spectrum
            size = 1
            while size < 2 * N:
                size *= 2
            h_fft = np.fft.rfft(h_spec, n=size)
            power = np.abs(h_fft)**2

            # We want power to be as flat as possible
            # Target: constant power (but constrained by nonnegativity of h)
            # Use iterative projection: 
            # 1. In frequency domain: flatten the power spectrum
            # 2. In time domain: enforce nonnegativity and unit norm

            # Flatten: new_power = sqrt(mean(power)) for all frequencies
            # But this doesn't work directly. Instead, use a soft flattening
            mean_power = np.mean(power)
            # Blend toward flat
            alpha = 0.1
            new_power = (1 - alpha) * power + alpha * mean_power
            new_power = np.maximum(new_power, 0)

            # Take square root and inverse FFT (preserve phase)
            phase = np.angle(h_fft)
            new_fft = np.sqrt(new_power) * np.exp(1j * phase)
            h_new = np.fft.irfft(new_fft, n=size)[:N]

            # Enforce nonnegativity and unit norm
            h_new = np.maximum(h_new, 0)
            norm = np.sqrt(np.sum(h_new**2))
            if norm < 1e-15:
                break
            h_new /= norm

            score = _autoconvolution_peak(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()

            h_spec = h_new

    # Try scipy optimization from best point
    if time_left() > 30:
        def objective_smooth(h):
            """Smooth objective using log-sum-exp."""
            h_pos = np.maximum(h, 0)
            norm_sq = np.sum(h_pos**2)
            if norm_sq < 1e-15:
                return 1e10
            ac = _autoconvolution_all(h_pos)
            ac_norm = ac / norm_sq
            # Log-sum-exp for smooth max
            max_val = np.max(ac_norm)
            if max_val < 1e-15:
                return max_val
            temp = 0.01
            shifted = ac_norm - max_val
            log_sum = np.log(np.sum(np.exp(shifted / temp))) * temp + max_val
            return log_sum

        try:
            result = minimize(
                objective_smooth,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            h_opt = np.maximum(result.x, 0)
            norm = np.sqrt(np.sum(h_opt**2))
            if norm > 1e-15:
                h_opt /= norm
                score = _autoconvolution_peak(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Final refinement with smaller steps
    if time_left() > 5:
        h_ref = best_h.copy()
        lr = 0.01
        for _ in range(500):
            if time_left() < 2:
                break
            h_new, ac = _subgradient_step_fast(h_ref, lr)
            score = _autoconvolution_peak(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
                h_ref = h_new.copy()
                lr *= 0.98
            else:
                lr *= 0.95
                if lr < 0.0001:
                    break

    # Return the best heights found
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    return list(best_h)
