"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Strategy: Parametric geometric search over narrow Gaussian lobe configurations,
optimized via L-BFGS-B on lobe parameters (centers, widths, heights), followed by
direct h-space polishing. Narrow disjoint lobes concentrate L2 norm while
controlling autoconvolution supremum.
"""
import numpy as np
import time

N = 50
GRID_START = -0.25
GRID_END = 0.25
X_GRID = np.linspace(GRID_START, GRID_END, N)

def _autoconv(h):
    """Compute autoconvolution of h using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return conv

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup)

def _lobes_to_h(params):
    """
    Convert lobe parameters to h.
    params: [c1, w1, h1, c2, w2, h2, ..., cK, wK, hK]
    Each lobe is a Gaussian: h_i * exp(-(x - c_i)^2 / (2*w_i^2))
    """
    K = len(params) // 3
    h = np.zeros(N)
    for k in range(K):
        c = params[3*k]
        w = max(abs(params[3*k+1]), 1e-4)
        ht = max(params[3*k+2], 0.0)
        h += ht * np.exp(-((X_GRID - c)**2) / (2.0 * w**2))
    return np.maximum(h, 0.0)

def _neg_log_ratio_lobes(params):
    """Objective: negative log of ratio, for lobe parameters."""
    h = _lobes_to_h(params)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 1e10
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 1e10
    ratio = l2_sq / (l1 * sup)
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)

def _neg_log_ratio_direct(h):
    """Objective: negative log of ratio, for direct h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 1e10
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 1e10
    ratio = l2_sq / (l1 * sup)
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_h = np.ones(N)
    best_r = _ratio(best_h)

    from scipy.optimize import minimize

    # Phase 1: Optimize lobe parameters for different numbers of lobes
    for n_lobe in [1, 2, 3, 4, 5]:
        if time_left() < 20:
            break

        # Try multiple random initializations
        for trial in range(15):
            if time_left() < 15:
                break

            rng = np.random.RandomState(42 + n_lobe * 100 + trial)

            if trial == 0:
                # Symmetric configuration
                if n_lobe == 1:
                    centers = [0.0]
                elif n_lobe % 2 == 0:
                    centers = list(np.linspace(-0.18, 0.18, n_lobe))
                else:
                    centers = list(np.linspace(-0.18, 0.18, n_lobe))
                widths = [0.025] * n_lobe
                heights = [1.0] * n_lobe
            else:
                centers = sorted(rng.uniform(-0.22, 0.22, n_lobe).tolist())
                widths = rng.uniform(0.012, 0.045, n_lobe).tolist()
                heights = rng.uniform(0.5, 2.0, n_lobe).tolist()

            params0 = np.array([])
            for c, w, ht in zip(centers, widths, heights):
                params0 = np.append(params0, [c, w, ht])

            # Bounds: centers in [-0.25, 0.25], widths in [0.005, 0.15], heights in [0, 5]
            bounds = []
            for k in range(n_lobe):
                bounds.append((-0.25, 0.25))  # center
                bounds.append((0.005, 0.15))   # width
                bounds.append((0.0, 5.0))      # height

            try:
                res = minimize(
                    _neg_log_ratio_lobes,
                    params0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                h_opt = _lobes_to_h(res.x)
                r_opt = _ratio(h_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Phase 2: Direct h-space optimization from best lobe result
    if time_left() > 25:
        try:
            res = minimize(
                _neg_log_ratio_direct,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.maximum(res.x, 0)
            r_opt = _ratio(h_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 3: SLSQP with L2 normalization constraint (ratio is scale-invariant)
    if time_left() > 15:
        try:
            def neg_score_normed(h):
                h = np.maximum(h, 0)
                n2 = np.sqrt(np.sum(h * h))
                if n2 < 1e-15:
                    return 1e10
                h_n = h / n2
                return _neg_log_ratio_direct(h_n)

            def l2_constraint(h):
                return np.sum(h**2) - 1.0

            def l2_jac(h):
                return 2.0 * h

            h0 = best_h.copy()
            n2 = np.sqrt(np.sum(h0**2))
            if n2 > 1e-15:
                h0 = h0 / n2

            res = minimize(
                neg_score_normed,
                h0,
                method='SLSQP',
                bounds=[(0, None)] * N,
                constraints=[{'type': 'eq', 'fun': l2_constraint, 'jac': l2_jac}],
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            h_opt = np.maximum(res.x, 0)
            r_opt = _ratio(h_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 4: Coordinate descent polish with adaptive steps
    if time_left() > 8:
        h = best_h.copy()
        for step in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
            if time_left() < 3:
                break
            improved = True
            while improved and time_left() > 3:
                improved = False
                for i in range(N):
                    if time_left() < 3:
                        break
                    h_save = h[i]
                    # Try +step
                    h[i] = h_save + step
                    h[i] = max(h[i], 0)
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
                        improved = True
                        continue
                    # Try -step
                    h[i] = max(h_save - step, 0)
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
                        improved = True
                        continue
                    h[i] = h_save

    # Phase 5: Try sparse spike configurations
    if time_left() > 5:
        for n_spikes in [1, 2, 3, 4, 5]:
            if time_left() < 2:
                break
            for trial in range(10):
                if time_left() < 2:
                    break
                rng = np.random.RandomState(999 + n_spikes * 10 + trial)
                h_sp = np.zeros(N)
                if n_spikes == 1:
                    pos = N // 2
                    h_sp[pos-1:pos+2] = 1.0
                else:
                    positions = sorted(rng.choice(N, n_spikes, replace=False))
                    for p in positions:
                        h_sp[max(0,p-1):min(N,p+2)] = 1.0
                h_sp = np.maximum(h_sp, 0)
                r_sp = _ratio(h_sp)
                if r_sp > best_r:
                    best_r = r_sp
                    best_h = h_sp.copy()

    # Final polish
    if time_left() > 3:
        h = best_h.copy()
        for step in [0.001, 0.0005, 0.0002, 0.0001]:
            if time_left() < 1:
                break
            for i in range(N):
                if time_left() < 1:
                    break
                h_save = h[i]
                for delta in [step, -step]:
                    h[i] = max(h_save + delta, 0)
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
                h[i] = h_save

    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return list(best_h)
