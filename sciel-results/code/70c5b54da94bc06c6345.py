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


def _smooth_max_val_grad(h, beta):
    """Return smooth max value and gradient w.r.t. h."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    # Gradient: dc_k/dh_j involves h[k-j] and h[j-k]
    # cs_k = (2/n)(n/2 - r_k) where r_k = sum_j h[j]h[j+k] (circular)
    # dc_k/dh_j = -(2/n) * (h[j+k] + h[j-k]) for circular indices
    # Actually r_k = sum_{j=0}^{n-1-k} h[j]h[j+k] + sum_{j=n-k}^{n-1} h[j]h[j+k-n]
    # But using FFT correlation gives circular autocorrelation
    # d r_k / d h_j = h[(j+k) mod n] + h[(j-k) mod n] for k != 0
    # d r_0 / d h_j = 2*h[j]
    # So d c_k / d h_j = -(2/n) * (h[(j+k)%n] + h[(j-k)%n]) for k>0, -(4/n)*h[j] for k=0
    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        if k == 0:
            grad += -(4.0 / n) * wk * h
        else:
            # h[(j+k)%n] is np.roll(h, -k), h[(j-k)%n] is np.roll(h, k)
            grad += -(2.0 / n) * wk * (np.roll(h, -k) + np.roll(h, k))
    return val, grad


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _spectral_to_h(coeffs):
    """Convert spectral coefficients to height sequence.
    coeffs[0] = DC offset, coeffs[1:K+1] = cosine amplitudes, coeffs[K+1:2K+1] = sine amplitudes
    """
    K = (len(coeffs) - 1) // 2
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, K + 1):
        h += coeffs[k] * np.cos(2.0 * np.pi * k * i / N)
        h += coeffs[K + k] * np.sin(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    """Convert height sequence to spectral coefficients (first K harmonics)."""
    K = 5  # Use first 5 harmonics
    H = np.fft.rfft(h, n=N)
    coeffs = np.zeros(2 * K + 1)
    coeffs[0] = H[0].real / N
    for k in range(1, K + 1):
        coeffs[k] = 2.0 * H[k].real / N
        coeffs[K + k] = -2.0 * H[k].imag / N
    return coeffs


def _spectral_smooth_max_grad(coeffs, beta, K):
    """Smooth max value and gradient w.r.t. spectral coefficients."""
    h = _spectral_to_h(coeffs)
    val, grad_h = _smooth_max_val_grad(h, beta)
    # Chain rule: d val / d coeffs[k] = sum_j grad_h[j] * d h[j] / d coeffs[k]
    i = np.arange(N)
    grad_coeffs = np.zeros(2 * K + 1)
    grad_coeffs[0] = np.sum(grad_h)
    for k in range(1, K + 1):
        grad_coeffs[k] = np.sum(grad_h * np.cos(2.0 * np.pi * k * i / N))
        grad_coeffs[K + k] = np.sum(grad_h * np.sin(2.0 * np.pi * k * i / N))
    return val, grad_coeffs


def _project_spectral(coeffs, K):
    """Project spectral coefficients so that resulting h is in [0,1] and sums to N/2."""
    h = _spectral_to_h(coeffs)
    # Adjust DC to fix sum
    target_dc = (N / 2.0 - np.sum(h - coeffs[0] * N)) / N
    coeffs = coeffs.copy()
    coeffs[0] = target_dc
    h = _spectral_to_h(coeffs)
    # Clip and re-project
    h = np.clip(h, 0.0, 1.0)
    # Re-extract spectral coefficients from clipped h
    coeffs_new = _h_to_spectral(h)
    # But we need to maintain the constraint... use iterative approach
    for _ in range(50):
        h = _spectral_to_h(coeffs_new)
        h = np.clip(h, 0.0, 1.0)
        # Fix sum by adjusting DC
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
    # Re-extract from fixed h
    coeffs = _h_to_spectral(h)

    coeffs_prev = coeffs.copy()
    t_k = 1.0
    lr = 0.05

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _spectral_smooth_max_grad(coeffs, beta, K)

        # Gradient step
        coeffs_new = coeffs - lr * grad

        # Project
        coeffs_new = _project_spectral(coeffs_new, K)

        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        coeffs_next = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs_prev)
        coeffs_next = _project_spectral(coeffs_next, K)

        coeffs_prev = coeffs
        coeffs = coeffs_next
        t_k = t_new

        # Adaptive learning rate
        if it % 50 == 49:
            lr = max(0.01, min(0.2, lr * 0.9))

    h_final = _spectral_to_h(coeffs)
    h_final = _fix_sum(h_final)
    return coeffs, h_final, _score(h_final)


def _accelerated_smooth_max(h0, beta, max_iter=500):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = 0.01

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        val, grad = _smooth_max_val_grad(h, beta)

        # Gradient step
        h_new = h - lr * grad

        # Project onto {h: 0 <= h_i <= 1, sum(h) = N/2}
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new = _fix_sum(h_new)

        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = np.clip(h_next, 0.0, 1.0)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        # Adaptive learning rate
        if it % 50 == 49:
            lr = max(0.001, min(0.1, lr * 0.95))

    return h, _score(h)


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

    # Phase 1: Spectral optimization
    K = 5  # Number of harmonics

    # Generate diverse spectral starts
    spectral_starts = []

    # Cosine fundamental dominant
    c1 = np.zeros(2 * K + 1)
    c1[0] = 0.5
    c1[1] = 0.4
    spectral_starts.append(c1)

    # Sine fundamental
    c2 = np.zeros(2 * K + 1)
    c2[0] = 0.5
    c2[K + 1] = 0.4
    spectral_starts.append(c2)

    # Two frequencies
    c3 = np.zeros(2 * K + 1)
    c3[0] = 0.5
    c3[1] = 0.3
    c3[2] = 0.2
    spectral_starts.append(c3)

    # Random spectral starts
    for _ in range(15):
        c_rand = np.zeros(2 * K + 1)
        c_rand[0] = 0.5
        for k in range(1, K + 1):
            c_rand[k] = np.random.uniform(-0.15, 0.15)
            c_rand[K + k] = np.random.uniform(-0.15, 0.15)
        spectral_starts.append(c_rand)

    # Evaluate spectral starts
    for c0 in spectral_starts:
        h0 = _spectral_to_h(c0)
        h0 = _fix_sum(h0)
        _update_best(h0)

    # Phase 2: Spectral Nesterov optimization
    if best_h is not None and time.time() < _DEADLINE - 15:
        best_coeffs = _h_to_spectral(best_h)
        for beta in [30.0, 60.0, 100.0]:
            if time.time() > _DEADLINE - 15:
                break
            c_candidate, h_candidate, s_candidate = _accelerated_spectral(
                best_coeffs, beta=beta, K=K, max_iter=200
            )
            if s_candidate < best_score:
                _update_best(h_candidate)
                best_coeffs = _h_to_spectral(h_candidate)

    # Phase 3: Full-space Nesterov refinement
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=50.0, max_iter=300)
        if s_candidate < best_score:
            _update_best(h_candidate)

        if time.time() < _DEADLINE - 10:
            h_candidate2, s_candidate2 = _accelerated_smooth_max(best_h, beta=100.0, max_iter=300)
            if s_candidate2 < best_score:
                _update_best(h_candidate2)

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

        _run_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)

        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=200, ftol=1e-12)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
