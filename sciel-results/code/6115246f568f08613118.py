"""
Optimized step-height construction for autocorrelation inequality.

Strategy:
1. Generate diverse initial candidates using parametric families.
2. Use FFT-based fast evaluation to speed up the search.
3. Polish the best candidate with a smooth-max surrogate and gradient-based optimization.
4. Final coordinate descent for fine-tuning.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400


def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def fast_evaluate(h):
        """Evaluate the objective using FFT for speed."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        # Use FFT to compute convolution
        n_fft = 2 * N
        h_fft = np.fft.rfft(h, n=n_fft)
        conv_fft = h_fft * h_fft
        conv = np.fft.irfft(conv_fft, n=n_fft)[:2 * N - 1]
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    # Evaluate seed
    val_seed, _ = fast_evaluate(best_h)
    best_value = val_seed

    def update_best(h):
        nonlocal best_value, best_h
        val, valid = fast_evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()
        return val, valid

    # Parametric families for initial candidates
    rng = np.random.RandomState(42)

    def make_gaussian(params):
        a, center, sigma, b = params
        idx = np.arange(N)
        h = a * np.exp(-((idx - center) / sigma) ** 2) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    def make_two_block(params):
        a, b, k_frac = params
        k = int(k_frac * N)
        h = np.zeros(N)
        h[:k] = a
        h[k:] = b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    def make_three_block(params):
        a, b, c, k1, k2 = params
        k1 = int(k1 * N)
        k2 = int(k2 * N)
        h = np.zeros(N)
        h[:k1] = a
        h[k1:k2] = b
        h[k2:] = c
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    def make_cosine(params):
        a, freq, phase, b = params
        t = np.linspace(0, 1, N, endpoint=False)
        h = a * np.cos(2 * np.pi * freq * t + phase) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    def make_triangle(params):
        a, center, width, b = params
        idx = np.arange(N)
        h = a * np.maximum(0, 1 - np.abs(idx - center) / width) + b
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    def make_multilevel(params):
        a, b, n_blocks = params
        n_blocks = int(n_blocks)
        h = np.zeros(N)
        block_size = N // n_blocks
        for i in range(n_blocks):
            val = a if i % 2 == 0 else b
            start = i * block_size
            end = min(start + block_size, N)
            h[start:end] = val
        s = np.sum(h)
        if abs(s) < 1e-12:
            return None
        return h / s

    # Generate diverse initial candidates
    candidates = []

    # Gaussian bumps
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 5.0)
        center = rng.uniform(0.2 * N, 0.8 * N)
        sigma = rng.uniform(N / 10, N / 3)
        b = rng.uniform(-0.5, 0.5)
        h = make_gaussian([a, center, sigma, b])
        if h is not None:
            candidates.append(h)

    # Two-block
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        k_frac = rng.uniform(0.1, 0.9)
        h = make_two_block([a, b, k_frac])
        if h is not None:
            candidates.append(h)

    # Three-block
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        c = rng.uniform(0.5, 3.0)
        k1 = rng.uniform(0.1, 0.4)
        k2 = rng.uniform(0.5, 0.9)
        h = make_three_block([a, b, c, k1, k2])
        if h is not None:
            candidates.append(h)

    # Cosine
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        freq = rng.choice([1, 2, 3, 4, 5, 7, 10])
        phase = rng.uniform(0, 2 * np.pi)
        b = rng.uniform(-1.0, 1.0)
        h = make_cosine([a, freq, phase, b])
        if h is not None:
            candidates.append(h)

    # Triangle
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 5.0)
        center = rng.uniform(0.2 * N, 0.8 * N)
        width = rng.uniform(N / 5, N / 2)
        b = rng.uniform(-0.5, 0.5)
        h = make_triangle([a, center, width, b])
        if h is not None:
            candidates.append(h)

    # Multi-level
    for _ in range(300):
        if time.time() - start_time > DEADLINE - 60:
            break
        a = rng.uniform(0.5, 3.0)
        b = rng.uniform(-2.0, 2.0)
        n_blocks = rng.randint(4, 20)
        h = make_multilevel([a, b, n_blocks])
        if h is not None:
            candidates.append(h)

    # Random smooth functions
    for _ in range(200):
        if time.time() - start_time > DEADLINE - 60:
            break
        h = rng.randn(N)
        # Smooth with a Gaussian kernel
        kernel_size = rng.randint(5, 20)
        kernel = np.exp(-0.5 * (np.arange(-kernel_size, kernel_size + 1) / (kernel_size / 3)) ** 2)
        kernel /= np.sum(kernel)
        h = np.convolve(h, kernel, mode='same')
        s = np.sum(h)
        if abs(s) < 1e-12:
            continue
        h = h / s
        candidates.append(h)

    # Evaluate all candidates and keep the best
    for h in candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
        update_best(h)

    # Polish with smooth-max surrogate and gradient-based optimization
    def smooth_objective_and_grad(h, beta):
        """Compute smooth max surrogate and its gradient."""
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)

        # Compute convolution using FFT
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

        # Gradient of soft_max with respect to h
        # d(soft_max)/d(h[k]) = sum_j weights[j] * d(conv[j])/d(h[k])
        # conv[j] = sum_i h[i] * h[j-i]
        # d(conv[j])/d(h[k]) = h[j-k] + h[k-j] (for valid indices)
        grad_soft_max = np.zeros(N)
        for k in range(N):
            # conv[j] = sum_i h[i] h[j-i]
            # d(conv[j])/d(h[k]) = h[j-k] (if j-k in [0,N-1]) + h[k-j] (if k-j in [0,N-1])
            # But more precisely: conv[j] = sum_i h[i] h[j-i]
            # d(conv[j])/d(h[k]) = sum_i [delta(i,k) h[j-i] + h[i] delta(j-i,k)]
            # = h[j-k] + sum_i h[i] delta(j-i,k)
            # = h[j-k] + h[j-k] (if j-k in [0,N-1])
            # Actually: d(conv[j])/d(h[k]) = h[j-k] (if 0 <= j-k < N) + h[k-j] (if 0 <= k-j < N)
            # But conv[j] = sum_i h[i] h[j-i], so:
            # d(conv[j])/d(h[k]) = h[j-k] (when i=k, term is h[k] h[j-k], derivative is h[j-k])
            # + sum_i h[i] * d(h[j-i])/d(h[k]) = sum_i h[i] * delta(j-i, k) = h[j-k] (when j-i=k, i=j-k)
            # So total = 2 * h[j-k] if 0 <= j-k < N, else 0
            # Wait, that's not right either. Let me reconsider.
            # conv[j] = sum_{i=0}^{N-1} h[i] h[j-i]
            # d(conv[j])/d(h[k]) = h[j-k] (from first factor) + h[j-k] (from second factor, when i=j-k)
            # = 2 * h[j-k] if 0 <= j-k < N
            # But this is only valid when both i=k and i=j-k are in range.
            # More carefully:
            # conv[j] = sum_i h[i] h[j-i]
            # d(conv[j])/d(h[k]) = sum_i [delta(i,k) h[j-i] + h[i] delta(j-i, k)]
            # = h[j-k] + sum_i h[i] delta(j-i, k)
            # = h[j-k] + h[j-k] (when j-i=k, i=j-k)
            # = 2 * h[j-k] if 0 <= j-k < N
            # So:
            if 0 <= j - k < N:
                pass  # This loop structure is wrong, let me redo
            # Actually, I need to sum over j:
            # grad_soft_max[k] = sum_j weights[j] * d(conv[j])/d(h[k])
            # = sum_j weights[j] * (h[j-k] if 0 <= j-k < N else 0) + (h[k-j] if 0 <= k-j < N else 0)
            # Let me compute this more efficiently.
        # Redo gradient computation
        grad_soft_max = np.zeros(N)
        for k in range(N):
            # d(conv[j])/d(h[k]) = h[j-k] (if 0 <= j-k < N) + h[k-j] (if 0 <= k-j < N)
            # But h[j-k] and h[k-j] are the same term when j-k = k-j, i.e., j=2k.
            # Let's just compute:
            # For each j, the contribution is weights[j] * (h[j-k] if 0 <= j-k < N else 0)
            # + weights[j] * (h[k-j] if 0 <= k-j < N else 0)
            # Note: h[j-k] is only defined when 0 <= j-k < N, i.e., k <= j < k+N
            # And h[k-j] is only defined when 0 <= k-j < N, i.e., k-N < j <= k
            # So:
            # First term: sum over j from max(0,k) to min(2N-1, k+N-1) of weights[j] * h[j-k]
            # Second term: sum over j from max(0,k-N+1) to min(2N-1, k) of weights[j] * h[k-j]
            # But h[k-j] = h[|k-j|] only if symmetric, which it's not necessarily.
            # Actually h is just an array, so h[k-j] means the element at index k-j.
            # Let me just do a direct loop for clarity (N=400, should be fast enough).
            total = 0.0
            for j in range(len(conv)):
                dk = 0.0
                if 0 <= j - k < N:
                    dk += h[j - k]
                if 0 <= k - j < N:
                    dk += h[k - j]
                total += weights[j] * dk
            grad_soft_max[k] = total

        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))

        return f, grad_f

    if time.time() - start_time < DEADLINE - 10:
        h_curr = best_h.copy()

        for beta in [50, 200, 500, 1000]:
            if time.time() - start_time > DEADLINE - 5:
                break

            try:
                result = minimize(
                    lambda h, b=beta: smooth_objective_and_grad(h, b)[0],
                    h_curr,
                    jac=lambda h, b=beta: smooth_objective_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )

                if result.fun < 1e10:
                    val, valid = fast_evaluate(result.x)
                    if valid and val < best_value:
                        best_value = val
                        best_h = result.x.copy()
                        h_curr = result.x.copy()

            except Exception:
                pass

        # Final coordinate descent polish
        if time.time() - start_time < DEADLINE - 2:
            h_curr = best_h.copy()
            lr = 0.05
            for iteration in range(30):
                if time.time() - start_time > DEADLINE - 2:
                    break
                improved = False
                for coord in rng.permutation(N):
                    if time.time() - start_time > DEADLINE - 2:
                        break
                    for delta in [lr, -lr, lr * 0.5, -lr * 0.5]:
                        h_try = h_curr.copy()
                        h_try[coord] += delta
                        val_try, valid = fast_evaluate(h_try)
                        if valid and val_try < best_value:
                            best_value = val_try
                            best_h = h_try.copy()
                            h_curr = h_try
                            improved = True
                            break
                if not improved:
                    lr *= 0.5
                if lr < 1e-6:
                    break

    return best_h
