import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    # Bisection on a global shift to fix sum while respecting [0,1] bounds
    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    # Fine-tune
    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _smooth_max_val_grad(h, beta):
    """Return smooth max value and gradient w.r.t. h using FFT-based convolution."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # Gradient: grad_j = sum_k w_k * dc_k/dh_j
    # dc_k/dh_j = -(2/n) * (h[(j+k)%n] + h[(j-k)%n]) for k>0
    #            = -(4/n) * h[j] for k=0
    # This is equivalent to -(2/n) * (w * h)_circular + -(2/n) * (w * h_flip)_circular
    # where h_flip is h reversed (for the cross-correlation part)
    # Actually, let's use the direct FFT convolution approach which is cleaner.
    # grad = -(2/n) * (conv(w, h) + conv(w, h_flip))
    # conv(w, h) is circular convolution.
    # w is real, h is real.
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]

    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))

        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _spectral_to_h(coeffs):
    """Convert spectral coefficients to height sequence."""
    K = (len(coeffs) - 1) // 2
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, K + 1):
        h += coeffs[k] * np.cos(2.0 * np.pi * k * i / N)
        h += coeffs[K + k] * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    """Convert height sequence to spectral coefficients."""
    K = 5
    H = np.fft.rfft(h, n=N)
    coeffs = np.zeros(2 * K + 1)
    coeffs[0] = H[0].real / N
    for k in range(1, K + 1):
        coeffs[k] = 2.0 * H[k].real / N
        coeffs[K + k] = -2.0 * H[k].imag / N
    return coeffs


def _project_spectral(coeffs, K):
    """Project spectral coefficients so that resulting h is in [0,1] and sums to N/2."""
    h = _spectral_to_h(coeffs)
    coeffs = coeffs.copy()
    # Adjust DC to fix sum
    target_dc = (N / 2.0 - np.sum(h - coeffs[0] * N)) / N
    coeffs[0] = target_dc
    h = _spectral_to_h(coeffs)
    h = np.clip(h, 0.0, 1.0)
    coeffs_new = _h_to_spectral(h)
    for _ in range(50):
        h = _spectral_to_h(coeffs_new)
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        coeffs_new[0] += diff / N
        h = _spectral_to_h(coeffs_new)
        h = np.clip(h, 0.0, 1.0)
        coeffs_new = _h_to_spectral(h)
    return coeffs_new


def _accelerated_spectral(coeffs0, beta, K, max_iter=300):
    """Nesterov accelerated gradient in spectral space."""
    coeffs = np.array(coeffs0, dtype=np.float64, copy=True)
    coeffs = _project_spectral(coeffs, K)
    h = _spectral_to_h(coeffs)
    h = _fix_sum(h)
    coeffs = _h_to_spectral(h)

    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = 0.05

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        h_curr = _spectral_to_h(coeffs)
        val, grad_h = _smooth_max_val_grad(h_curr, beta)
        i = np.arange(N)
        grad_coeffs = np.zeros(2 * K + 1)
        grad_coeffs[0] = np.sum(grad_h)
        for k in range(1, K + 1):
            grad_coeffs[k] = np.sum(grad_h * np.cos(2.0 * np.pi * k * i / N))
            grad_coeffs[K + k] = np.sum(grad_h * np.sin(2.0 * np.pi * k * i / N))

        coeffs_new = coeffs - lr * grad_coeffs
        coeffs_new = _project_spectral(coeffs_new, K)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)
        coeffs_next = _project_spectral(coeffs_next, K)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        if it % 50 == 49:
            lr = max(0.01, min(0.2, lr * 0.9))

    h_final = _spectral_to_h(coeffs)
    h_final = _fix_sum(h_final)
    return coeffs, h_final, _score(h_final)


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(8):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    for f in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            h_sine = 0.5 + 0.3 * np.sin(2 * np.pi * f * i / N + phase)
            starts.append(_fix_sum(h_sine))

    for f in [1, 2, 3]:
        for amp in [0.2, 0.3, 0.4]:
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    h_2f = 0.5 + 0.3 * np.cos(2 * np.pi * i / N) + 0.2 * np.cos(4 * np.pi * i / N)
    starts.append(_fix_sum(h_2f))
    h_2f2 = 0.5 + 0.25 * np.sin(2 * np.pi * i / N) + 0.25 * np.sin(3 * np.pi * i / N)
    starts.append(_fix_sum(h_2f2))

    # Add binary-like starts
    h_bin1 = np.array([1.0 if j % 2 == 0 else 0.0 for j in range(N)])
    h_bin1 = h_bin1 * 0.5 + 0.5 * np.random.rand(N) # Perturb
    starts.append(_fix_sum(h_bin1))
    h_bin2 = np.array([1.0 if j % 4 < 2 else 0.0 for j in range(N)])
    h_bin2 = h_bin2 * 0.5 + 0.5 * np.random.rand(N)
    starts.append(_fix_sum(h_bin2))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Spectral optimization from best starts
    if best_h is not None and time.time() < _DEADLINE - 30:
        K = 5
        best_coeffs = _h_to_spectral(best_h)
        for beta in [30.0, 60.0]:
            if time.time() > _DEADLINE - 30:
                break
            c_candidate, h_candidate, s_candidate = _accelerated_spectral(
                best_coeffs, beta=beta, K=K, max_iter=200
            )
            if s_candidate < best_score:
                _update_best(h_candidate)
                best_coeffs = _h_to_spectral(h_candidate)

    # Phase 3: Full-space Nesterov refinement with adaptive beta
    if best_h is not None and time.time() < _DEADLINE - 15:
        # Start with moderate beta
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=50.0, max_iter=300)
        if s_candidate < best_score:
            _update_best(h_candidate)

        # Refine with higher beta
        if time.time() < _DEADLINE - 10:
            h_candidate2, s_candidate2 = _accelerated_smooth_max(best_h, beta=100.0, max_iter=300)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

        # Very high beta for sharp max
        if time.time() < _DEADLINE - 5:
            h_candidate3, s_candidate3 = _accelerated_smooth_max(best_h, beta=200.0, max_iter=200, lr_init=0.01)
            if s_candidate3 < best_score:
                _update_best(h_candidate3)

    # Phase 4: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_val_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_val_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=100.0, maxiter=100, ftol=1e-10)

        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=100, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
