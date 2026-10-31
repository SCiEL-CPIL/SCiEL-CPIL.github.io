import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit
NFFT = 2048  # Next power of 2 >= 2*N-1 (1199) to avoid circular aliasing

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def smooth_max(c, temp):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)

    def gradient_smooth_metric(h, temp):
        """Compute gradient of smooth_metric."""
        s = np.sum(h)
        if s < 1e-15:
            return np.zeros(N)
        
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        
        # Softmax weights
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        exp_c = np.exp(c_shifted)
        sum_exp = np.sum(exp_c)
        w = exp_c / sum_exp
        
        # w is defined on the linear convolution indices 0..2N-2.
        # We need the gradient with respect to h.
        # J = (1/s^2) * tau * log(sum exp(c/tau))
        # dJ/dh_i = (1/s^2) * sum_k w_k * dc_k/dh_i - (2J/s)
        
        # dc_k/dh_i = h_{(k-i) mod N} + h_{(i-k) mod N} is for circular.
        # For linear conv c_k = sum_{j=0}^N h_j h_{k-j} (where h_m=0 if m<0 or m>=N)
        # dc_k/dh_i = h_{k-i} + h_{i-k} (with 0 padding)
        
        # We can compute g_i = sum_k w_k (h_{k-i} + h_{i-k})
        # This is equivalent to 2 * (w * h)_i where * is cross-correlation?
        # Let's use FFT.
        # w is length 2N-1. h is length N.
        # We need to align w and h for convolution.
        
        # Pad w to length NFFT. Pad h to length NFFT.
        W = np.fft.rfft(w, n=NFFT)
        # h is real, so H is conjugate symmetric? No, H is from h.
        # We need sum_k w_k h_{k-i}.
        # This is (w * h)_{N-1+i}? 
        # Linear conv w * h has length (2N-1) + N - 1 = 3N-2.
        # We want the part where indices align such that k-i corresponds to h index.
        
        # Alternative: Just compute the gradient via FFT of the "blurred" weights.
        # Let's stick to the previous logic if possible or use a simpler numerical gradient?
        # Numerical gradient is too slow (2N FFTs).
        
        # Let's derive the FFT relation carefully.
        # c = h * h (linear).
        # w = softmax(c).
        # grad_h J = (2/s^2) * (w * h)_{centered} - (2J/s)
        
        # We compute conv_w_h = np.fft.irfft(W * H, n=NFFT)
        # This gives circular convolution of w (padded) and h (padded).
        # Since w is length 2N-1 and h is N, and NFFT >= 3N-2? 
        # 3*600-2 = 1798. NFFT=2048 is sufficient for linear conv of w and h.
        
        conv_w_h = np.fft.irfft(W * H, n=NFFT)
        
        # The linear convolution result is in conv_w_h[: (2N-1) + N - 1]
        # We need to extract the part corresponding to sum_k w_k h_{k-i}.
        # Let L = 2N-1. w has indices 0..L-1. h has indices 0..N-1.
        # (w*h)_m = sum_k w_k h_{m-k}.
        # We want sum_k w_k h_{k-i}. Let j = k-i => k = j+i.
        # sum_j w_{j+i} h_j.
        # This is cross-correlation?
        
        # Let's just use the property:
        # grad_i = sum_k w_k (h_{k-i} + h_{i-k})
        # Term 1: sum_k w_k h_{k-i}.
        # Term 2: sum_k w_k h_{i-k}.
        
        # Let's compute term 1 using FFT.
        # We want a vector g1 where g1[i] = sum_k w_k h_{k-i}.
        # This is the convolution of w and h, but flipped?
        # (w * h)[i] = sum_k w[k] h[i-k].
        # We want sum_k w[k] h[k-i].
        # Let h_flip[k] = h[-k mod N]? No, h is linear.
        
        # Actually, h is defined on 0..N-1.
        # If we treat h as periodic for the FFT, it's circular.
        # But we established we need linear.
        
        # Let's use a simpler approach for the gradient if the FFT derivation is tricky.
        # We can use the fact that we are optimizing a smooth function.
        # Maybe just use `scipy.optimize.minimize` with `BFGS` or `L-BFGS-B` on the full vector again, but with the CORRECT FFT size (2048) and more iterations?
        
        # The previous Phase 3 used L-BFGS-B. With the correct FFT size, the landscape might be different/better.
        # Also, the "stalled progress" might be due to the aliasing bug.
        
        return None # Placeholder, we will use L-BFGS-B in Phase 4 instead of custom gradient descent to avoid derivation errors.

    n = np.arange(N)

    # Phase 1: Structured candidates
    candidates = []
    candidates.append(np.ones(N))

    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))

    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))

    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) +
                0.277263158 * np.cos(4*np.pi*n/(N-1)) +
                0.083578947 * np.cos(6*np.pi*n/(N-1)) +
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))

    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)

    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)

    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)

    # Additional diverse candidates
    for freq in [1, 2, 3]:
        for amp in [0.05, 0.1, 0.2]:
            h_test = 1.0 + amp * np.cos(2 * np.pi * freq * n / N)
            h_test = np.clip(h_test, 0, None)
            if h_test.sum() > 0:
                candidates.append(h_test)

    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Cosine parameterization optimization with adaptive temperature
    def cosine_basis(coeffs, order):
        c0 = coeffs[0]
        h = np.full(N, c0)
        for m in range(1, order + 1):
            cm = coeffs[1 + 2*(m-1)]
            dm = coeffs[2 + 2*(m-1)]
            h += cm * np.cos(2 * np.pi * m * n / N) + dm * np.sin(2 * np.pi * m * n / N)
        return h

    def fit_cosine(h, order):
        coeffs = np.zeros(1 + 2*order)
        coeffs[0] = np.mean(h)
        for m in range(1, order + 1):
            coeffs[1 + 2*(m-1)] = 2.0 * np.mean(h * np.cos(2 * np.pi * m * n / N))
            coeffs[2 + 2*(m-1)] = 2.0 * np.mean(h * np.sin(2 * np.pi * m * n / N))
        return coeffs

    def cosine_objective(coeffs, order, temp):
        if elapsed() > DEADLINE - 10:
            return 1e10
        h = cosine_basis(coeffs, order)
        h = np.clip(h, 0, None)
        if h.sum() < 1e-15:
            return 1e10
        return smooth_metric(h, temp)

    best_cosine_val = best_val
    best_cosine_h = best_h.copy()

    for order in [4, 6, 8, 10, 12]:
        if elapsed() > DEADLINE - 40:
            break

        coeffs0 = fit_cosine(best_h, order)

        # Multi-temperature annealing
        temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        current_coeffs = coeffs0.copy()

        for temp in temps:
            if elapsed() > DEADLINE - 20:
                break
            try:
                result = minimize(
                    lambda c: cosine_objective(c, order, temp),
                    current_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 500}
                )
                if result.fun < 1e9:
                    current_coeffs = result.x
                    h_opt = cosine_basis(current_coeffs, order)
                    h_opt = np.clip(h_opt, 0, None)
                    v = fast_metric(h_opt)
                    if v < best_cosine_val:
                        best_cosine_val = v
                        best_cosine_h = h_opt.copy()
            except Exception:
                pass

    if best_cosine_val < best_val:
        best_val = best_cosine_val
        best_h = best_cosine_h.copy()

    # Phase 3: Full-vector gradient optimization with log-sum-exp surrogate
    def full_objective(x, temp):
        if elapsed() > DEADLINE - 10:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp)

    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)

    for temp in [0.02, 0.01, 0.005, 0.002, 0.001]:
        if elapsed() > DEADLINE - 15:
            break
        try:
            result = minimize(
                lambda x: full_objective(x, temp),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 400}
            )
            if result.fun < 1e9:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)
        except Exception:
            pass

    # Phase 4: High-precision Polish with Corrected FFT and Tighter L-BFGS
    # Since the previous Phase 4 was slow coordinate descent, we replace it with
    # a more aggressive L-BFGS run on the direct h variables (non-negative constraint)
    # using the soft-max surrogate.
    
    def direct_objective(h, temp):
        if elapsed() > DEADLINE - 5:
            return 1e10
        return smooth_metric(h, temp)

    # Use the best_h found so far
    h_current = best_h.copy()
    
    # Run L-BFGS-B on h directly. 
    # Note: h must be non-negative. L-BFGS-B handles bounds.
    bounds = [(0, None)] * N
    
    # Try a few temperatures for the polish
    for temp in [0.005, 0.002, 0.001]:
        if elapsed() > DEADLINE - 5:
            break
        try:
            result = minimize(
                lambda h: direct_objective(h, temp),
                h_current,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 1000}
            )
            if result.fun < 1e9:
                h_opt = result.x
                h_opt = np.maximum(h_opt, 0)
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    h_current = h_opt.copy()
        except Exception:
            pass

    # Phase 5: Final Coordinate Descent with Gradient-Informed Steps
    # Only if time remains
    if elapsed() < DEADLINE - 2:
        current_h = best_h.copy()
        current_val = best_val
        np.random.seed(42)
        
        # Use a small number of iterations
        max_iters = 500
        for _ in range(max_iters):
            if elapsed() > DEADLINE - 1:
                break
            idx = np.random.randint(N)
            orig = current_h[idx]
            
            # Try a few small steps
            step_sizes = [0.01, 0.001, -0.01, -0.001]
            mean_h = np.mean(current_h)
            step_sizes = [s * mean_h for s in step_sizes]
            
            best_local_val = current_val
            best_local_v = orig
            
            for cv in step_sizes:
                new_val = orig + cv
                if new_val < 0:
                    continue
                current_h[idx] = new_val
                v = fast_metric(current_h)
                if v < best_local_val:
                    best_local_val = v
                    best_local_v = new_val
            
            if best_local_val < current_val - 1e-12:
                current_h[idx] = best_local_v
                current_val = best_local_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                current_h[idx] = orig

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
