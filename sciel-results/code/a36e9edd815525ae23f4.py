"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_soft(h, beta):
    """Soft-max surrogate objective. h is normalized internally so sum(h) = N."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, beta)
    return abs(2.0 * N * sm / (N * N))

def _objective_exact(h):
    """Exact objective for comparison."""
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    return abs(2.0 * N * np.max(conv) / (N * N))

def _gradient_of_max_conv(h):
    """
    Compute gradient of the max autoconvolution entry w.r.t. h.
    Returns (gradient, max_index, max_value).
    """
    conv = _convolve_fft(h)
    k_star = int(np.argmax(conv))
    M = conv[k_star]
    N_local = len(h)
    g = np.zeros(N_local)
    for i in range(N_local):
        k_minus_i = k_star - i
        if 0 <= k_minus_i < N_local:
            if i == k_minus_i:
                g[i] = h[k_minus_i]
            else:
                g[i] = 2.0 * h[k_minus_i]
    return g, k_star, M

def _gradient_objective(h):
    """
    Compute gradient of V = |2*N*max(conv)/S^2| w.r.t. h.
    Assumes max(conv) > 0 for simplicity (common case).
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return None
    conv = _convolve_fft(h)
    k_star = int(np.argmax(conv))
    M = conv[k_star]
    N_local = len(h)
    
    # Gradient of M w.r.t. h
    gM = np.zeros(N_local)
    for i in range(N_local):
        k_minus_i = k_star - i
        if 0 <= k_minus_i < N_local:
            if i == k_minus_i:
                gM[i] = h[k_minus_i]
            else:
                gM[i] = 2.0 * h[k_minus_i]
    
    # V = 2*N*M / S^2 (assuming M > 0)
    # dV/dh[i] = 2*N/S^2 * gM[i] - 4*N*M/S^3
    if M > 0:
        grad = 2.0 * N / (s * s) * gM - 4.0 * N * M / (s ** 3)
    else:
        # V = -2*N*M/S^2, dV/dh[i] = -2*N/S^2 * gM[i] + 4*N*M/S^3
        grad = -2.0 * N / (s * s) * gM + 4.0 * N * M / (s ** 3)
    
    return grad

def _soft_max_gradient(h, beta):
    """
    Compute gradient of the soft-max surrogate objective.
    Returns gradient of V_soft = 2*N*softmax(conv)/N^2 w.r.t. h.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return None
    h_norm = h * (N / s)
    conv = _convolve_fft(h_norm)
    
    # Soft max weights
    m = np.max(conv)
    exp_vals = np.exp(beta * (conv - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_max w.r.t. conv[k] is weights[k]
    # conv[k] = sum_i h[i]*h[k-i] (with bounds)
    # d(soft_max)/dh[j] = sum_k weights[k] * d(conv[k])/dh[j]
    # d(conv[k])/dh[j] = h[k-j] if 0<=k-j<N, times 2 if j != k-j, times 1 if j == k-j
    
    N_local = len(h)
    grad_conv = np.zeros(N_local)
    for k in range(len(conv)):
        wk = weights[k]
        if wk < 1e-15:
            continue
        for j in range(N_local):
            k_minus_j = k - j
            if 0 <= k_minus_j < N_local:
                if j == k_minus_j:
                    grad_conv[j] += wk * h_norm[k_minus_j]
                else:
                    grad_conv[j] += 2.0 * wk * h_norm[k_minus_j]
    
    # Chain rule: h_norm = h * N/s, so d(h_norm)/dh[j] = N/s * e_j - h * N/s^2
    # V = 2*N*softmax/N^2 = 2*softmax/N
    # dV/dh[j] = (2/N) * sum_k weights[k] * d(conv[k])/dh_norm[j] * d(h_norm)/dh[j]
    # This is complex; simpler to just use numerical gradient or the direct formula
    
    # Actually, let's just return the gradient of the soft objective directly
    # V_soft = 2*N*softmax(conv(h_norm))/N^2 where h_norm = h*N/s
    # This is getting complex. Let's use a simpler approach.
    return None

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses FFT-based convolution, soft-max surrogate optimization,
    gradient-based polish, and coordinate descent.
    """
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Start with all-ones (value = 2.0)
    best_h = np.ones(N)
    best_val = 2.0

    # Phase 1: Quick parametric families to get a good starting point
    idx = np.arange(N)

    # Family 1: h[i] = 1 + a*cos(2*pi*m*i/N + phase) - reduced search
    for m in range(1, 15):
        if time.time() > deadline - 100:
            break
        for a in np.linspace(-1.0, 1.0, 11):
            for phase in [0.0, np.pi / 4, np.pi / 2]:
                h = 1.0 + a * np.cos(2.0 * np.pi * m * idx / N + phase)
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Family 2: Square wave / sign patterns - reduced
    for m in range(1, 11):
        if time.time() > deadline - 95:
            break
        for a in np.linspace(0.5, 2.0, 6):
            h = 1.0 + a * np.sign(np.sin(2.0 * np.pi * m * idx / N))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 3: Two-block pattern
    for frac in np.linspace(0.1, 0.9, 9):
        if time.time() > deadline - 90:
            break
        for a in np.linspace(0.5, 3.0, 7):
            h = np.ones(N)
            h[:int(frac * N)] = a
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()

    # Family 4: Sinusoid with offset - reduced
    for m in range(1, 8):
        if time.time() > deadline - 85:
            break
        for a in np.linspace(-1.5, 1.5, 8):
            for offset in [-0.5, 0.0, 0.5]:
                h = offset + a * np.sin(2.0 * np.pi * m * idx / N)
                s = np.sum(h)
                if abs(s) < 1e-6:
                    continue
                v = _eval_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # Phase 2: Soft-max surrogate optimization with L-BFGS-B
    # More restarts and adaptive beta
    rng = np.random.RandomState(123)

    beta_schedule = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    n_restarts = 6

    for beta in beta_schedule:
        if time.time() > deadline - 30:
            break

        def obj_time(h, _beta=beta):
            return _objective_soft(h, _beta)

        for trial in range(n_restarts):
            if time.time() > deadline - 25:
                break

            # Start from best_h or perturbed version
            if trial == 0:
                h_init = best_h.copy()
            elif trial == 1:
                h_init = best_h + 0.05 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            elif trial == 2:
                h_init = best_h + 0.1 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            elif trial == 3:
                # Random restart with structure
                h_init = 1.0 + 0.5 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))
            elif trial == 4:
                # DCT-based random restart
                c = dct(best_h, type=2, norm='ortho')
                c[1:] += 0.15 * rng.randn(N - 1)
                h_init = idct(c, type=2, norm='ortho')
                h_init = h_init * (N / np.sum(h_init))
            else:
                # Larger perturbation
                h_init = best_h + 0.3 * rng.randn(N)
                h_init = h_init * (N / np.sum(h_init))

            try:
                result = minimize(
                    obj_time,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )

                h_opt = result.x
                v_opt = _eval_value(h_opt)
                if v_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    best_val = v_opt
                    best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 3: Gradient-based subgradient descent with adaptive step size
    if time.time() < deadline - 20:
        h = best_h.copy()
        alpha = 0.01  # initial step size
        for it in range(300):
            if time.time() > deadline - 15:
                break
            grad = _gradient_objective(h)
            if grad is None:
                break
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-10:
                break
            
            # Backtracking line search
            best_step_val = _eval_value(h)
            best_step_h = h.copy()
            step = alpha
            for _ls in range(10):
                h_try = h - step * grad / gnorm
                v_try = _eval_value(h_try)
                if v_try < best_step_val:
                    best_step_val = v_try
                    best_step_h = h_try.copy()
                    alpha = min(step * 1.5, 0.1)  # increase step if it worked
                    break
                else:
                    step *= 0.5
                    alpha = max(step, 0.0001)
            
            if best_step_val < best_val:
                best_val = best_step_val
                best_h = best_step_h.copy()
                h = best_step_h.copy()
            else:
                # Try opposite direction
                h_try = h + alpha * grad / gnorm
                v_try = _eval_value(h_try)
                if v_try < best_val:
                    best_val = v_try
                    best_h = h_try.copy()
                    h = h_try.copy()
                else:
                    # No improvement, try smaller step
                    h_try = h - 0.001 * grad / gnorm
                    v_try = _eval_value(h_try)
                    if v_try < best_val:
                        best_val = v_try
                        best_h = h_try.copy()
                        h = h_try.copy()
                    else:
                        break

    # Phase 4: Fast coordinate descent with batched evaluation
    if time.time() < deadline - 10:
        h = best_h.copy()
        # Use smaller set of step sizes and fewer sweeps
        step_sizes = [0.02, 0.01, 0.005, 0.002]

        for step in step_sizes:
            if time.time() > deadline - 5:
                break
            for sweep in range(3):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    old = h[i]
                    for delta in [step, -step]:
                        h[i] = old + delta
                        s = np.sum(h)
                        if abs(s) < 1e-10:
                            h[i] = old
                            continue
                        v = _eval_value(h)
                        if v < best_val:
                            best_val = v
                            best_h = h.copy()
                            improved = True
                        else:
                            h[i] = old
                    h[i] = old
                if not improved:
                    break

    # Phase 5: Final high-precision soft-max polish
    if time.time() < deadline - 5:
        beta_final = 200.0

        def obj_final(h):
            return _objective_soft(h, beta_final)

        try:
            result = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            h_opt = result.x
            v_opt = _eval_value(h_opt)
            if v_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                best_val = v_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 6: Final exact gradient polish
    if time.time() < deadline - 2:
        h = best_h.copy()
        for it in range(50):
            if time.time() > deadline - 1:
                break
            grad = _gradient_objective(h)
            if grad is None:
                break
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            # Very small step
            h_try = h - 0.001 * grad / gnorm
            v_try = _eval_value(h_try)
            if v_try < best_val:
                best_val = v_try
                best_h = h_try.copy()
                h = h_try.copy()
            else:
                break

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
