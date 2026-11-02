"""Program using direct projected gradient descent for first autocorrelation inequality."""
import time
import numpy as np
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1  # set in construct()

def _fft_convolve(h):
    """FFT-based linear convolution (full mode)."""
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Set FFT size: next power of 2 >= 2N-1
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2
    FFT_N = _FFT_SIZE

    # --- Direct projected gradient descent on h ---
    # Objective: F(h) = max_k conv(h)[k] / (sum(h))^2
    # We use a smooth surrogate: soft_max(conv) / (sum(h))^2
    # where soft_max(conv) = m + log(sum(exp((conv - m) * beta)) / beta)
    #
    # Gradient of soft_max(conv) w.r.t. h[j]:
    #   conv[k] = sum_i h[i] * h[k-i] (linear conv, k=0..2N-2)
    #   d(conv[k])/d(h[j]) = h[k-j] + h[j-k+N-1] ... 
    #   Actually for linear conv: conv[k] = sum_{i=max(0,k-N+1)}^{min(k,N-1)} h[i]*h[k-i]
    #   d(conv[k])/d(h[j]) = h[k-j] if 0<=k-j<N, else 0, PLUS h[j-k+N-1] if... 
    #   Wait, let me think more carefully.
    #   conv[k] = sum_i h[i] h[k-i] where h[i]=0 for i not in [0,N-1].
    #   d(conv[k])/d(h[j]) = h[k-j] (from the i=j term in the sum over i)
    #   So grad_j = sum_k w_k * h[k-j] where w_k = softmax weights, and h[k-j] is 0 if k-j not in [0,N-1].
    #   This is a convolution of w with h! Specifically:
    #   grad = w * h (linear conv of w and h), but w has length 2N-1 and h has length N.
    #   Actually: grad[j] = sum_k w[k] * h[k-j] for k-j in [0,N-1]
    #   = sum_{m=0}^{N-1} w[j+m] * h[m]  (substituting m = k-j, so k = j+m, k in [0,2N-2])
    #   So grad[j] = sum_{m=0}^{N-1} w[j+m] * h[m], where w is defined on [0, 2N-2] and w[j+m]=0 if j+m > 2N-2.
    #   This is a cross-correlation. We can compute it via FFT.

    def soft_max_grad(h, beta=300.0):
        """Compute gradient of soft_max(conv(h)) w.r.t. h."""
        s = np.sum(h)
        if s == 0:
            return np.zeros(N), 0.0
        conv = _fft_convolve(h)
        m = np.max(conv)
        if m <= 0:
            return np.zeros(N), 0.0
        w = np.exp((conv - m) * beta)
        w_sum = np.sum(w)
        if w_sum == 0:
            return np.zeros(N), 0.0
        w = w / w_sum
        # soft_max value
        lse = m + np.log(w_sum / beta)
        # Gradient: grad[j] = sum_{m=0}^{N-1} w[j+m] * h[m]
        # This is the cross-correlation of w and h, evaluated at lags 0..N-1
        # We can compute this via FFT: it's the convolution of w with reversed h, or
        # equivalently, the circular cross-correlation.
        # Let's use: grad = ifft(fft(w) * conj(fft(h))) but with proper linear handling.
        # Actually, let's just do it directly via FFT linear convolution.
        # grad[j] = sum_{m=0}^{N-1} w[j+m] * h[m]
        # This is (w cross-correlate h)[j] = sum_m w[j+m] h[m]
        # = (w * flip(h))[j + N - 1] where * is linear conv? Let me verify.
        # flip(h)[k] = h[N-1-k]. 
        # (w * flip(h))[n] = sum_k w[k] flip(h)[n-k] = sum_k w[k] h[N-1-(n-k)] = sum_k w[k] h[N-1-n+k]
        # Set j = N-1-n, so n = N-1-j:
        # (w * flip(h))[N-1-j] = sum_k w[k] h[j+k]
        # So grad[j] = (w * flip(h))[N-1-j]
        # Let's compute w * flip(h) via FFT.
        h_flip = h[::-1]
        # Linear conv of w (length 2N-1) and h_flip (length N): result length 3N-2
        # We need indices N-1-j for j=0..N-1, i.e., indices N-1, N-2, ..., 0
        # FFT size for this conv:
        conv_len = len(w) + len(h_flip) - 1
        fft_size = 1
        while fft_size < conv_len:
            fft_size *= 2
        W = np.fft.rfft(w, n=fft_size)
        HF = np.fft.rfft(h_flip, n=fft_size)
        conv_wh = np.fft.irfft(W * HF, n=fft_size)[:conv_len]
        # grad[j] = conv_wh[N-1-j]
        grad = conv_wh[N-1::-1]  # indices N-1, N-2, ..., 0
        return grad, lse

    def objective_and_grad(h, beta=300.0):
        """Compute F(h) = soft_max(conv)/(s^2) and its gradient."""
        s = np.sum(h)
        if s <= 0:
            return 0.0, np.zeros(N)
        conv = _fft_convolve(h)
        m = np.max(conv)
        if m <= 0:
            return 0.0, np.zeros(N)
        w = np.exp((conv - m) * beta)
        w_sum = np.sum(w)
        if w_sum == 0:
            return 0.0, np.zeros(N)
        w = w / w_sum
        lse = m + np.log(w_sum / beta)
        F = lse / (s * s)
        # grad of F = (s^2 * grad_lse - lse * 2*s * 1) / s^4 = grad_lse/s^2 - 2*lse/(s^3)
        # where grad_lse is gradient of lse w.r.t. h
        grad_lse, _ = soft_max_grad(h, beta)
        grad = grad_lse / (s * s) - 2.0 * lse / (s ** 3) * np.ones(N)
        return F, grad

    # --- Initialize with several good starting points ---
    w_interval = 0.5 / N
    x = np.linspace(-0.25 + w_interval/2, 0.25 - w_interval/2, N)
    t = x / 0.25  # Map to [-1, 1]

    def make_gaussian(sigma, p=2.0):
        h = np.exp(-np.abs(x / sigma) ** p)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def make_poly_bump(a, b):
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, (1.0 - abs_t ** a) ** b, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    def make_cosine_bump(p):
        abs_t = np.abs(t)
        h = np.where(abs_t < 1.0, np.cos(np.pi * abs_t / 2.0) ** p, 0.0)
        s = np.sum(h)
        if s > 0:
            h = h / s
        return h

    # Generate candidate initializations
    candidates = []
    # Gaussian family
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
        for p in [1.0, 1.5, 2.0, 3.0, 4.0]:
            candidates.append(make_gaussian(sigma, p))
    # Polynomial bump family
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for b in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            candidates.append(make_poly_bump(a, b))
    # Cosine bump family
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        candidates.append(make_cosine_bump(p))
    # Flat
    candidates.append(np.ones(N) / N)

    # Evaluate all candidates
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    for h_cand in candidates:
        h_cand = np.maximum(0, h_cand)
        s_cand = np.sum(h_cand)
        if s_cand > 0:
            h_cand = h_cand / s_cand
        sc = compute_score(h_cand)
        if sc < best_score:
            best_score = sc
            best_h = h_cand.copy()

    # --- Projected gradient descent ---
    # Minimize F(h) subject to h >= 0 and sum(h) = 1 (normalization)
    # We'll work in the space of h >= 0, and renormalize after each step.
    
    beta = 200.0
    lr = 0.001
    h_current = best_h.copy()
    h_current = np.maximum(0, h_current)
    s_cur = np.sum(h_current)
    if s_cur > 0:
        h_current = h_current / s_cur

    # Backtracking line search
    def pgd_step(h, beta=200.0, max_backtrack=10):
        F0, grad0 = objective_and_grad(h, beta)
        if not np.isfinite(F0):
            return h, F0
        lr_local = 0.005
        for _ in range(max_backtrack):
            h_new = h - lr_local * grad0
            h_new = np.maximum(0, h_new)
            s_new = np.sum(h_new)
            if s_new > 0:
                h_new = h_new / s_new
            F_new, _ = objective_and_grad(h_new, beta)
            if F_new < F0:
                return h_new, F_new
            lr_local *= 0.5
        return h, F0

    # Run projected gradient descent with decreasing beta
    beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    for beta_val in beta_schedule:
        if time.time() > deadline - 10:
            break
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            h_current, F_val = pgd_step(h_current, beta_val)
            if F_val < best_score:
                best_score = F_val
                best_h = h_current.copy()
            # Check convergence
            if iteration > 0 and abs(F_prev - F_val) < 1e-12:
                break
            F_prev = F_val

    # --- Additional polish with L-BFGS on parametric families from best point ---
    if time.time() < deadline - 15:
        # Determine which parametric family is closest to best_h and polish it
        # Try L-BFGS on several families from multiple starting points
        def obj_gaussian(params, beta=800.0):
            sigma, p = params
            sigma = np.clip(sigma, 0.01, 0.5)
            p = np.clip(p, 0.3, 10.0)
            h = make_gaussian(sigma, p)
            conv = _fft_convolve(h)
            s = np.sum(h)
            if s == 0:
                return 0.0
            m = np.max(conv)
            if m <= 0:
                return 0.0
            lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
            return lse / (s * s)

        def obj_poly(params, beta=800.0):
            a, b = params
            a = np.clip(a, 0.1, 10.0)
            b = np.clip(b, 0.1, 10.0)
            h = make_poly_bump(a, b)
            conv = _fft_convolve(h)
            s = np.sum(h)
            if s == 0:
                return 0.0
            m = np.max(conv)
            if m <= 0:
                return 0.0
            lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
            return lse / (s * s)

        def obj_cosine(params, beta=800.0):
            p = params[0]
            p = np.clip(p, 0.1, 10.0)
            h = make_cosine_bump(p)
            conv = _fft_convolve(h)
            s = np.sum(h)
            if s == 0:
                return 0.0
            m = np.max(conv)
            if m <= 0:
                return 0.0
            lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
            return lse / (s * s)

        # Polish Gaussian family
        for s0 in [0.05, 0.08, 0.1, 0.12, 0.15]:
            for p0 in [1.0, 1.5, 2.0, 3.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    res = optimize.minimize(
                        obj_gaussian, [s0, p0], method='L-BFGS-B',
                        bounds=[(0.01, 0.5), (0.3, 10.0)],
                        options={'maxiter': 200, 'ftol': 1e-12}
                    )
                    if res.fun < best_score:
                        h_cand = make_gaussian(*np.clip(res.x, [0.01, 0.3], [0.5, 10.0]))
                        sc = compute_score(h_cand)
                        if sc < best_score:
                            best_score = sc
                            best_h = h_cand.copy()
                except Exception:
                    pass

        # Polish polynomial family
        for a0 in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for b0 in [0.5, 1.0, 1.5, 2.0, 3.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    res = optimize.minimize(
                        obj_poly, [a0, b0], method='L-BFGS-B',
                        bounds=[(0.1, 10.0), (0.1, 10.0)],
                        options={'maxiter': 200, 'ftol': 1e-12}
                    )
                    if res.fun < best_score:
                        h_cand = make_poly_bump(*np.clip(res.x, [0.1, 0.1], [10.0, 10.0]))
                        sc = compute_score(h_cand)
                        if sc < best_score:
                            best_score = sc
                            best_h = h_cand.copy()
                except Exception:
                    pass

        # Polish cosine family
        for p0 in [0.5, 1.0, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = optimize.minimize(
                    obj_cosine, [p0], method='L-BFGS-B',
                    bounds=[(0.1, 10.0)],
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.fun < best_score:
                    h_cand = make_cosine_bump(np.clip(res.x[0], 0.1, 10.0))
                    sc = compute_score(h_cand)
                    if sc < best_score:
                        best_score = sc
                        best_h = h_cand.copy()
            except Exception:
                pass

    # Final: one more round of PGD from best_h with high beta
    if time.time() < deadline - 5:
        h_current = best_h.copy()
        h_current = np.maximum(0, h_current)
        s_cur = np.sum(h_current)
        if s_cur > 0:
            h_current = h_current / s_cur
        for iteration in range(200):
            if time.time() > deadline - 3:
                break
            h_current, F_val = pgd_step(h_current, beta=1000.0)
            if F_val < best_score:
                best_score = F_val
                best_h = h_current.copy()

    # Ensure non-negativity and return
    best_h = np.maximum(0, best_h)
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h / s_final

    return best_h.tolist()
