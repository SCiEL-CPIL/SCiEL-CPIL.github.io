import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit
NFFT = 2048  # next power of 2 >= 2*N-1 for exact linear autoconvolution

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

    def smooth_metric_and_grad(h, temp):
        """Smooth metric and its gradient w.r.t. h."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros_like(h)
        H = np.fft.rfft(h, n=NFFT)
        C = np.fft.irfft(H * H, n=NFFT)
        c = C[:2*N-1]
        # log-sum-exp weights
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        exp_c = np.exp(c_shifted)
        sum_exp = np.sum(exp_c)
        w = exp_c / sum_exp  # softmax weights
        m = smooth_max(c, temp)
        dm_dc = w  # d(smooth_max)/dc = softmax(c/temp)
        # dc/dh: c is linear autoconvolution, c[k] = sum_j h[j]*h[k-j] (with appropriate bounds)
        # dc/dh[i] = sum_k w[k] * dc[k]/dh[i]
        # dc[k]/dh[i] = h[k-i] + h[i-k] (for valid indices)
        # We compute grad via: grad[i] = sum_k w[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
        # This is a correlation-like operation. Use FFT for speed.
        # Actually, dc/dh[i] = (h * w)[i] where * is cross-correlation in the linear sense.
        # Let's use: grad_c = np.fft.irfft(np.fft.rfft(h, n=NFFT) * np.fft.rfft(w, n=NFFT), n=NFFT)[:2*N-1]
        # But we need the full linear correlation. Let me think again.
        # c[k] = sum_j h[j] * h[k-j] for k=0..2N-2
        # dc[k]/dh[i] = h[k-i] + h[i-k] (both terms, with h[x]=0 if x not in [0,N-1])
        # grad[i] = sum_k w[k] * (h[k-i] + h[i-k])
        # = sum_k w[k]*h[k-i] + sum_k w[k]*h[i-k]
        # First term: (w * h) correlation at lag i, second term: (h * w) correlation at lag i
        # Both are the same: cross-correlation of w and h.
        # Use FFT: cross_corr[i] = sum_k w[k] * h[k-i] (with proper bounds)
        # This is the linear cross-correlation. Compute via FFT.
        W = np.fft.rfft(w, n=NFFT)
        Hf = np.fft.rfft(h, n=NFFT)
        # cross-correlation: irfft(conj(W) * Hf) gives sum_k w[k]*h[k+i]... let me be careful.
        # np.fft.irfft(conj(W) * Hf, n=NFFT)[:2*N-1] gives the linear cross-correlation (w ⊛ h)
        # where (w ⊛ h)[i] = sum_k w[k] * h[k-i]... actually let me just use the direct formula.
        # For N=600, 2N-1=1199, direct O(N*(2N-1)) = 600*1199 ≈ 720K ops, which is fine.
        # grad_c[i] = sum_{k=0}^{2N-2} w[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
        # Let's compute this efficiently using the fact that it's a convolution.
        # grad_c = convolve(w, h, mode='full')[N-1:N-1+2*N-1]... hmm.
        # Actually, let me just use the FFT approach properly.
        # The linear autoconvolution c = h ⊛ h (linear convolution).
        # dc/dh[i] is the gradient of c w.r.t. h[i], which for autoconvolution is:
        # dc/dh[i] = 2 * (h ⊛ delta_i) evaluated... no.
        # Let me use: if c = h*h (elementwise in frequency), then dc/dh[i] = 2*conj(H)*dH/dh[i] in freq domain.
        # dH/dh[i] = exp(-2πi*i*k/NFFT) for each frequency bin k.
        # So grad in freq domain = 2*conj(H) * FFT(delta_i) where delta_i is impulse at i.
        # FFT(delta_i) = exp(-2πi*i*k/NFFT) for all k.
        # So grad_freq[k] = 2 * conj(H[k]) * exp(-2πi*i*k/NFFT)
        # grad[i] = IFFT(grad_freq)
        # But we have weights w, so: grad_metric_c[i] = sum_k w[k] * dc[k]/dh[i]
        # This is: grad_c = w ⊛ (dc/dh) ... no, it's the inner product of w with dc/dh.
        # dc/dh is a matrix. Let me think of it differently.
        # grad[i] = sum_k w[k] * dc[k]/dh[i]
        # dc[k]/dh[i] = h[k-i] + h[i-k] (with zero-padding)
        # So grad[i] = sum_k w[k]*h[k-i] + sum_k w[k]*h[i-k]
        # First sum: for fixed i, sum over k of w[k]*h[k-i]. Let j=k-i, sum_j w[j+i]*h[j] = (w_shifted ⊛ h) where w_shifted[j]=w[j+i]
        # This is getting complex. Let me just use the direct O(N^2) computation since N=600.
        # Actually, a cleaner approach: grad_c[i] = 2 * (w ⊛ h)[i] where ⊛ is linear cross-correlation
        # because dc/dh[i] = h ⊛ delta_i + delta_i ⊛ h = 2*(h ⊛ delta_i) for autoconvolution.
        # Wait, c[k] = sum_j h[j]*h[k-j]. dc[k]/dh[i] = h[k-i] + h[i-k] (when both indices valid).
        # So grad[i] = sum_k w[k]*(h[k-i] + h[i-k]) = 2 * sum_k w[k]*h[k-i] (by symmetry of the sum, since sum_k w[k]*h[i-k] = sum_k w[k]*h[k-i] by reindexing k->2N-2-k... not exactly).
        # Let me just compute it directly for correctness. N=600, 2N-1=1199, so 600*1199 ≈ 720K multiply-adds. Fast enough.
        # Build the gradient matrix implicitly.
        # grad[i] = sum_{k=0}^{2N-2} w[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
        # Vectorize: for each i, the valid k range for h[k-i] is k in [i, i+N-1], and for h[i-k] is k in [0, i].
        # Let me use numpy slicing.
        grad_c = np.zeros(N)
        for i in range(N):
            # Term 1: sum_k w[k]*h[k-i] where 0<=k-i<N => k in [i, min(i+N-1, 2N-2)]
            k_start = i
            k_end = min(i + N, 2*N - 1)
            if k_end > k_start:
                grad_c[i] += np.dot(w[k_start:k_end], h[:k_end-k_start])
            # Term 2: sum_k w[k]*h[i-k] where 0<=i-k<N => k in [max(0, i-N+1), min(i, 2N-2)]
            k_start2 = max(0, i - N + 1)
            k_end2 = min(i + 1, 2*N - 1)
            if k_end2 > k_start2:
                grad_c[i] += np.dot(w[k_start2:k_end2], h[i-k_end2+1:i-k_start2+1][::-1])
        # Now grad of metric = grad_c / (s^2) - 2*smooth_max/(s^3) * ones
        grad = grad_c / (s * s) - 2.0 * m / (s * s * s) * np.ones(N)
        return m / (s * s), grad

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

    # Phase 2: Full-vector gradient optimization with log-sum-exp surrogate
    def full_objective_and_grad(x, temp):
        if elapsed() > DEADLINE - 10:
            return 1e10, np.zeros_like(x)
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        val, grad_h = smooth_metric_and_grad(h, temp)
        # Chain rule: dval/dx_i = grad_h[i] * h[i]
        grad_x = grad_h * h
        return val, grad_x

    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)

    for temp in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        if elapsed() > DEADLINE - 30:
            break
        try:
            result = minimize(
                lambda x: full_objective_and_grad(x, temp),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
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

    # Phase 3: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)

    no_improve = 0
    max_no_improve = 5000

    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]

        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig

        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 300 == 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()

    # Phase 4: Fine polish with tiny paired moves
    if elapsed() < DEADLINE - 15:
        step_frac = 0.002
        for _ in range(2000):
            if elapsed() > DEADLINE - 13:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    # Phase 5: Exact-max local polish - refine the peak region directly
    if elapsed() < DEADLINE - 3:
        h_polish = best_h.copy()
        for _ in range(500):
            if elapsed() > DEADLINE - 2:
                break
            # Find where the peak is in the autoconvolution
            H = np.fft.rfft(h_polish, n=NFFT)
            C = np.fft.irfft(H * H, n=NFFT)
            c = C[:2*N-1]
            s = np.sum(h_polish)
            peak_idx = np.argmax(c)
            # Try to reduce the peak by adjusting h values that contribute to c[peak_idx]
            # c[peak_idx] = sum_j h[j]*h[peak_idx-j]
            # Adjust h at positions that are "central" to the peak
            # Try small decreases at high-contributing positions
            for trial in range(10):
                if elapsed() > DEADLINE - 2:
                    break
                # Pick a random position that contributes to the peak
                j = np.random.randint(N)
                k = peak_idx - j
                if 0 <= k < N and j != k:
                    # Try decreasing h[j] slightly
                    for frac in [0.01, 0.02, 0.05]:
                        new_h = h_polish.copy()
                        old_j = new_h[j]
                        new_h[j] = max(0, old_j * (1 - frac))
                        if new_h[j] >= 0:
                            v = fast_metric(new_h)
                            if v < best_val:
                                best_val = v
                                best_h = new_h
                                h_polish = new_h
                                break

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
