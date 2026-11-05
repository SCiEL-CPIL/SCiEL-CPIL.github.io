"""
Autocorrelation inequality optimizer - QP-based approach.

Strategy:
1. Generate diverse initial candidates using parametric families.
2. Formulate the problem as minimizing the squared L2 norm of autoconvolution
   subject to sum(h)=1, which suppresses peak magnitude.
3. Use SLSQP for the constrained optimization.
4. Polish with a smooth-max surrogate for the actual max-based objective.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400


def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def fast_evaluate(h):
        """Evaluate the objective using FFT for speed."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        n_fft = 2 * N
        h_fft = np.fft.rfft(h, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2 * N - 1]
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = fast_evaluate(best_h)
    best_value = val_seed

    def update_best(h):
        nonlocal best_value, best_h
        val, valid = fast_evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()
        return val, valid

    rng = np.random.RandomState(42)

    # Generate diverse initial candidates
    candidates = []

    # Two-block
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        k_frac = rng.uniform(0.1, 0.9)
        k = int(k_frac * N)
        h = np.zeros(N)
        h[:k] = a
        h[k:] = b
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        candidates.append(h)

    # Gaussian bumps
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 5.0)
        center = rng.uniform(0.2 * N, 0.8 * N)
        sigma = rng.uniform(N / 10, N / 3)
        b = rng.uniform(-0.5, 0.5)
        idx = np.arange(N)
        h = a * np.exp(-((idx - center) / sigma) ** 2) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        candidates.append(h)

    # Cosine
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        freq = rng.choice([1, 2, 3, 4, 5, 7, 10])
        phase = rng.uniform(0, 2 * np.pi)
        b = rng.uniform(-1.0, 1.0)
        t = np.linspace(0, 1, N, endpoint=False)
        h = a * np.cos(2 * np.pi * freq * t + phase) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        candidates.append(h)

    # Smooth random
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 60:
            break
        h = rng.randn(N)
        kernel_size = rng.randint(5, 20)
        kernel = np.exp(-0.5 * (np.arange(-kernel_size, kernel_size + 1) / (kernel_size / 3)) ** 2)
        kernel /= np.sum(kernel)
        h = np.convolve(h, kernel, mode='same')
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
        update_best(h)

    # QP-based optimization: minimize ||h*h||_2^2 subject to sum(h)=1
    # This suppresses the overall autoconvolution energy
    def qp_objective(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        h_norm = h / s
        n_fft = 2 * N
        h_fft = np.fft.rfft(h_norm, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2 * N - 1]
        return np.sum(conv ** 2)

    def qp_grad(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)
        h_norm = h / s
        # Gradient of ||h*h||_2^2 with respect to h
        # Using FFT: conv = h*h, d(conv)/d(h[k]) = 2*h[j-k] summed appropriately
        # For efficiency, compute via FFT
        n_fft = 2 * N
        h_fft = np.fft.rfft(h_norm, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2 * N - 1]
        
        # grad[k] = sum_j conv[j] * d(conv[j])/d(h[k])
        # d(conv[j])/d(h[k]) = 2 * h[j-k] if 0 <= j-k < N
        # Vectorized computation
        grad = np.zeros(N)
        for k in range(N):
            # j from max(0,k) to min(2N-1, k+N-1)
            j_start = max(0, k)
            j_end = min(2 * N - 1, k + N - 1)
            if j_start <= j_end:
                indices = np.arange(j_start, j_end + 1)
                h_vals = h_norm[indices - k]
                grad[k] = 2.0 * np.sum(conv[indices] * h_vals)
        # Chain rule for h_norm = h/s
        # d(h_norm)/d(h[k]) = (1/s) * (e_k - h_norm)
        grad = grad / s - np.sum(grad * h_norm) / s * h_norm
        return grad

    if time.time() - start_time < DEADLINE - 20:
        h_curr = best_h.copy()
        
        # Try QP optimization from best candidate
        try:
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0,
                           'jac': lambda h: np.ones(N)}]
            
            result = minimize(
                qp_objective,
                h_curr,
                jac=qp_grad,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            if result.fun < 1e10:
                val, valid = fast_evaluate(result.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = result.x.copy()
                    h_curr = result.x.copy()
        except Exception:
            pass

    # Smooth-max surrogate optimization
    def smooth_objective_and_grad(h, beta):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)

        n_fft = 2 * N
        h_fft = np.fft.rfft(h, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2 * N - 1]

        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)

        if sum_exp == 0:
            soft_max = max_conv
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            soft_max = max_conv + np.log(sum_exp) / beta
            weights = exp_shifted / sum_exp

        f = 2.0 * N * soft_max / (s * s)

        # Compute gradient efficiently using vectorized operations
        grad_soft_max = np.zeros(N)
        for j in range(len(conv)):
            wj = weights[j]
            if wj < 1e-15:
                continue
            # Contribution from h[j-k]: k from max(0, j-N+1) to min(N-1, j)
            k_min1 = max(0, j - N + 1)
            k_max1 = min(N - 1, j)
            if k_min1 <= k_max1:
                indices1 = np.arange(k_min1, k_max1 + 1)
                h_vals1 = h[j - indices1]
                grad_soft_max[indices1] += wj * h_vals1
            
            # Contribution from h[k-j]: k from max(0, j) to min(N-1, j+N-1)
            k_min2 = max(0, j)
            k_max2 = min(N - 1, j + N - 1)
            if k_min2 <= k_max2:
                indices2 = np.arange(k_min2, k_max2 + 1)
                h_vals2 = h[indices2 - j]
                grad_soft_max[indices2] += wj * h_vals2

        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))

        return f, grad_f

    if time.time() - start_time < DEADLINE - 10:
        h_curr = best_h.copy()

        for beta in [100, 300, 500]:
            if time.time() - start_time > DEADLINE - 5:
                break

            try:
                result = minimize(
                    lambda h, b=beta: smooth_objective_and_grad(h, b)[0],
                    h_curr,
                    jac=lambda h, b=beta: smooth_objective_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if result.fun < 1e10:
                    val, valid = fast_evaluate(result.x)
                    if valid and val < best_value:
                        best_value = val
                        best_h = result.x.copy()
                        h_curr = result.x.copy()

            except Exception:
                pass

    return best_h
