"""Model-and-optimize: epigraph + multi-restart L-BFGS + spectral polish + dominant-lag CD.
Rewritten for speed (precomputed FFT plan) and a more aggressive search budget."""
N = 400
L = 2 * N - 1
L_fft = 1
while L_fft < L:
    L_fft *= 2
L_corr = 1
while L_corr < 2 * L_fft:
    L_corr *= 2

import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145.0

# Precomputed FFT plans
_plan = np.fft.Plan
_AC = np.fft.Plan(np.zeros(L_fft))          # autoconv: rfft(x, L_fft) -> irfft(|.|^2, L_fft)
_ACf = np.fft.Plan(np.zeros(L_fft))          # forward rfft(x, L_fft)
_CORR = np.fft.Plan(np.zeros(L_corr))        # cross-corr: rfft(g, L_corr) * conj(rfft(x, L_corr)) -> irfft
_CORRf = np.fft.Plan(np.zeros(L_corr))       # forward rfft(x, L_corr)

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - 2.0)

def _autoconv_fft(x):
    X = _ACf.rfft(x)
    return _AC.irfft(X * X)[:L]

def _objective_only(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """Soft-max surrogate for max|ac| plus gradient w.r.t. x."""
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    sum_exps = np.sum(exps)
    soft_val = m + np.log(sum_exps) / beta
    weights = exps / sum_exps
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs

    g_padded = np.zeros(L_corr)
    g_padded[:L] = grad_ac
    G = _CORRf.rfft(g_padded)
    X = _CORRf.rfft(x)
    corr = _CORR.irfft(G * np.conj(X))
    grad_x = 2.0 * corr[:N]
    return soft_val, grad_x

def _objective_and_grad(x, beta):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    s2 = s * s
    val = soft_val / s2
    grad = grad_soft / s2 - 2.0 * soft_val * np.ones(N) / (s2 * s)
    return val, grad

def _epigraph_objective_and_grad(z, beta):
    """Epigraph: minimize t + pen*soft_max(|ac|/s^2 - t).  z = [x, t]."""
    x = z[:N]
    t_val = z[N]
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N + 1)
    ac = _autoconv_fft(x)
    s2 = s * s
    abs_ac = np.abs(ac) / s2
    shifted = abs_ac - t_val
    m = np.max(shifted)
    pen = 200.0
    if m < -1e-9:
        soft_pen = 0.0
        weights = np.zeros(L)
        signs = np.zeros(L)
    else:
        exps = np.exp(beta * (shifted - m))
        sum_exps = np.sum(exps)
        soft_pen = m + np.log(sum_exps) / beta
        weights = exps / sum_exps
        signs = np.sign(ac)
        signs[ac == 0] = 1.0

    obj_val = t_val + pen * soft_pen

    if m < -1e-9:
        grad_x = np.zeros(N)
    else:
        g = weights * signs / s2
        g_padded = np.zeros(L_corr)
        g_padded[:L] = g
        G = _CORRf.rfft(g_padded)
        X = _CORRf.rfft(x)
        corr = _CORR.irfft(G * np.conj(X))
        grad_x = pen * 2.0 * corr[:N]
        # correction for the -2*ac*sign/s^3 term
        h = weights * signs * ac / (s2 * s)
        h_padded = np.zeros(L_corr)
        h_padded[:L] = h
        H = _CORRf.rfft(h_padded)
        corr2 = _CORR.irfft(H * np.conj(X))
        grad_x -= pen * 2.0 * corr2[:N]

    grad_t = 1.0 - pen
    grad = np.concatenate([grad_x, [grad_t]])
    return obj_val, grad

def _dominant_lag_polish(x0, rng, max_iters=3000, step_size=0.05):
    """Vectorized coordinate descent attacking the top autoconvolution lags."""
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step = step_size
    no_improve = 0
    K = 12
    t_arr = np.arange(N)

    for _ in range(max_iters):
        if _check_deadline():
            break
        ac = _autoconv_fft(best_x)
        s = np.sum(best_x)
        s2 = s * s
        abs_ac = np.abs(ac) / s2
        top_k = np.argsort(abs_ac)[-K:]

        # gradient of |ac[lag]|/s^2 w.r.t. x, vectorized per lag
        # d(ac[lag])/dx[n] = x[n-lag] (in range) + x[n+lag] (in range)
        best_in_iter = False
        for lag in top_k:
            if _check_deadline():
                break
            lagp = abs(int(lag))
            if lagp == 0:
                g = 2.0 * best_x
            else:
                g = np.zeros(N)
                lo = max(0, lagp)
                hi = N - lagp
                # x[n-lagp] for n in [lagp, N-1]
                g[lagp:] = best_x[:N - lagp]
                # x[n+lagp] for n in [0, N-1-lagp]
                g[:N - lagp] += best_x[lagp:]
            sign_ac = np.sign(ac[lag]) if ac[lag] != 0 else 1.0
            grad_lag = sign_ac * g / s2
            x_trial = best_x - step * grad_lag
            val = _objective_only(x_trial)
            if val < best_val:
                best_val = val
                best_x = x_trial.copy()
                no_improve = 0
                best_in_iter = True
            else:
                no_improve += 1

        if not best_in_iter:
            no_improve += 1
        if no_improve > 400:
            step = max(0.004, step * 0.9)
            no_improve = 0
        else:
            step = min(0.12, step * 1.01)

    return best_x, best_val

def _spectral_init(rng, mode=0):
    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(L_fft // 2 + 1)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=L_fft // 2 + 1)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(L_fft // 2 + 1)

    phase = rng.uniform(0, 2 * np.pi, size=L_fft // 2 + 1)
    phase[0] = 0
    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]
    x = _AC.irfft(spectrum)[:N]
    if np.sum(x) < 0:
        x = -x
    return x

def _spectral_phase_optimize(x0, rng, max_iters=150, lr=0.012):
    X = _ACf.rfft(x0)
    mag = np.abs(X).copy()
    phase = np.angle(X).copy()
    phase[0] = 0
    best_x = x0.copy()
    best_val = _objective_only(x0)

    for it in range(max_iters):
        if _check_deadline():
            break
        val, grad = _objective_and_grad(best_x, 60.0)
        G = _ACf.rfft(grad)
        X_curr = mag * np.exp(1j * phase)
        phase_grad = np.imag(X_curr * np.conj(G)) / (mag + 1e-10)
        phase_grad[0] = 0
        phase -= lr * phase_grad
        spectrum = mag * np.exp(1j * phase)
        spectrum[0] = mag[0]
        x_trial = _AC.irfft(spectrum)[:N]
        if np.sum(x_trial) < 0:
            x_trial = -x_trial
            phase = -phase
            phase[0] = 0
        val_trial = _objective_only(x_trial)
        if val_trial < best_val:
            best_val = val_trial
            best_x = x_trial.copy()
            if it % 20 == 0 and it > 0:
                lr = max(0.001, lr * 0.95)
        else:
            phase += lr * phase_grad
            lr = max(0.001, lr * 0.8)
    return best_x, best_val

def _amplitude_optimize(x0, rng, iters=25):
    X = _ACf.rfft(x0)
    phase = np.angle(X)
    phase[0] = 0
    amp = np.abs(X).copy()
    n_freq = L_fft // 2 + 1
    best_amp = amp.copy()
    best_val = _objective_only(x0)

    for _ in range(iters):
        if _check_deadline():
            break
        amp_trial = amp * (1 + 0.05 * rng.normal(0, 1, size=n_freq))
        amp_trial = np.maximum(0.01, amp_trial)
        spectrum = amp_trial * np.exp(1j * phase)
        spectrum[0] = amp_trial[0]
        x_trial = _AC.irfft(spectrum)[:N]
        if np.sum(x_trial) < 0:
            x_trial = -x_trial
        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            amp = amp_trial.copy()
            best_amp = amp.copy()

    spectrum = best_amp * np.exp(1j * phase)
    spectrum[0] = best_amp[0]
    x_best = _AC.irfft(spectrum)[:N]
    if np.sum(x_best) < 0:
        x_best = -x_best
    return x_best, best_val

def _coordinate_descent(x0, rng, max_iters=15000, initial_step=0.02):
    best_x = x0.copy()
    best_val = _objective_only(best_x)
    step_size = initial_step
    no_improve = 0
    t_arr = np.arange(N)

    for _ in range(max_iters):
        if _check_deadline():
            break
        r = rng.random()
        if r < 0.25:
            idx = rng.integers(0, N)
            delta = step_size * rng.choice([-1, 1])
            x_trial = best_x.copy()
            x_trial[idx] += delta
        elif r < 0.5:
            n_perturb = rng.integers(2, 8)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size, size=n_perturb)
            x_trial = best_x + perturbation
        elif r < 0.7:
            freq = rng.integers(1, 6)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = step_size * np.sin(2 * np.pi * freq * t_arr / N + phase)
            x_trial = best_x + perturbation
        elif r < 0.85:
            n_perturb = rng.integers(5, 20)
            indices = rng.choice(N, size=n_perturb, replace=False)
            perturbation = np.zeros(N)
            perturbation[indices] = rng.normal(0, step_size * 2, size=n_perturb)
            x_trial = best_x + perturbation
        else:
            n_flip = rng.integers(3, 15)
            indices = rng.choice(N, size=n_flip, replace=False)
            x_trial = best_x.copy()
            x_trial[indices] = -x_trial[indices] * 0.7

        val = _objective_only(x_trial)
        if val < best_val:
            best_val = val
            best_x = x_trial.copy()
            step_size = max(0.001, step_size * 0.97)
            no_improve = 0
        else:
            no_improve += 1
            step_size = min(0.05, step_size * 1.015)

        if no_improve > 3000 and rng.random() < 0.02:
            x_restart = best_x + rng.normal(0, 0.05, size=N)
            val_restart = _objective_only(x_restart)
            if val_restart < best_val:
                best_val = val_restart
                best_x = x_restart.copy()
                no_improve = 0
    return best_x, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective_only(best_x)
    rng = np.random.default_rng(42)

    t = np.linspace(0, 1, N)
    initial_points = [
        np.ones(N),
        1.0 + 0.3 * np.sin(2 * np.pi * t),
        1.0 + 0.5 * np.cos(2 * np.pi * t),
        1.0 + 0.3 * np.sin(4 * np.pi * t),
        1.0 + 0.5 * np.cos(4 * np.pi * t),
        np.exp(1.5 * t),
        np.exp(-1.5 * t),
        np.exp(0.5 * t),
        np.exp(-0.5 * t),
        1.0 + 0.5 * (t - 0.5),
        1.0 + 0.8 * (t - 0.5) ** 2,
        1.0 + 0.5 * np.sin(6 * np.pi * t) * np.cos(2 * np.pi * t),
        1.0 + 0.3 * (t - 0.5) ** 3,
        np.maximum(0.5, 1.0 - 2.0 * np.abs(t - 0.5)),
        np.maximum(0.5, 1.0 - 3.0 * np.abs(t - 0.5)),
        1.0 + 0.4 * np.sin(8 * np.pi * t),
        1.0 + 0.4 * np.cos(10 * np.pi * t),
        np.maximum(0.3, 1.0 + 0.6 * np.sin(2 * np.pi * 3 * t)),
    ]
    for mode in range(5):
        for _ in range(3):
            initial_points.append(_spectral_init(rng, mode))
    for _ in range(8):
        scale = rng.uniform(0.5, 1.5)
        initial_points.append(scale * rng.normal(1.0, 0.2, size=N))

    # Phase 1: multi-restart L-BFGS with soft-max
    for x0 in initial_points[:16]:
        if _check_deadline():
            break
        x_start = x0 + rng.normal(0, 0.01, size=N)
        for beta in [60.0, 120.0]:
            if _check_deadline():
                break
            try:
                def obj_func(x, _beta=beta):
                    return _objective_and_grad(x, _beta)
                result = minimize(obj_func, x_start, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 140, 'gtol': 1e-9,
                                                     'ftol': 1e-12, 'maxls': 30})
                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

    if _check_deadline():
        return best_x

    # Phase 2: dominant-lag polish
    x_peak, val_peak = _dominant_lag_polish(best_x, rng, max_iters=2500, step_size=0.04)
    if val_peak < best_val:
        best_val = val_peak
        best_x = x_peak.copy()

    if _check_deadline():
        return best_x

    # Phase 3: spectral phase optimization
    x_phase, val_phase = _spectral_phase_optimize(best_x, rng, max_iters=120, lr=0.012)
    if val_phase < best_val:
        best_val = val_phase
        best_x = x_phase.copy()

    if _check_deadline():
        return best_x

    # Phase 4: amplitude optimization
    for _ in range(3):
        if _check_deadline():
            break
        x_amp, val_amp = _amplitude_optimize(best_x, rng, iters=20)
        if val_amp < best_val:
            best_val = val_amp
            best_x = x_amp.copy()

    if _check_deadline():
        return best_x

    # Phase 5: epigraph L-BFGS (auxiliary t) for a tighter local solve
    for beta in [150.0, 300.0]:
        if _check_deadline():
            break
        try:
            z0 = np.concatenate([best_x, [best_val]])
            def epi_func(z, _beta=beta):
                return _epigraph_objective_and_grad(z, _beta)
            result = minimize(epi_func, z0, method='L-BFGS-B',
                              jac=True, options={'maxiter': 120, 'gtol': 1e-10,
                                                 'ftol': 1e-13, 'maxls': 25})
            x_trial = result.x[:N]
            val_trial = _objective_only(x_trial)
            if val_trial < best_val:
                best_val = val_trial
                best_x = x_trial.copy()
        except Exception:
            pass

    if _check_deadline():
        return best_x

    # Phase 6: coordinate descent refinement
    best_x, best_val = _coordinate_descent(best_x, rng, max_iters=15000, initial_step=0.015)

    # Phase 7: dominant-lag polish again
    if not _check_deadline():
        x_peak2, val_peak2 = _dominant_lag_polish(best_x, rng, max_iters=2000, step_size=0.03)
        if val_peak2 < best_val:
            best_val = val_peak2
            best_x = x_peak2.copy()

    # Phase 8: final high-precision epigraph polish
    if not _check_deadline():
        for beta in [500.0]:
            if _check_deadline():
                break
            try:
                z0 = np.concatenate([best_x, [best_val]])
                def epi_func(z, _beta=beta):
                    return _epigraph_objective_and_grad(z, _beta)
                result = minimize(epi_func, z0, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 90, 'gtol': 1e-11,
                                                     'ftol': 1e-14, 'maxls': 20})
                x_trial = result.x[:N]
                val_trial = _objective_only(x_trial)
                if val_trial < best_val:
                    best_val = val_trial
                    best_x = x_trial.copy()
            except Exception:
                pass

    return best_x
