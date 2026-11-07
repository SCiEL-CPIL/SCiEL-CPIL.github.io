import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under 150s kill
_EPS = 1e-12


def _value(h):
    """Exact objective value (higher is better)."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return -np.inf
    if not np.all(np.isfinite(h)):
        return -np.inf
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    ninf = np.max(g)
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)


def _smooth_value(h, tau=0.05):
    """Smooth surrogate for the objective (differentiable everywhere)."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    # smooth max via log-sum-exp
    gmax = np.max(g)
    ninf = gmax + tau * np.log(np.sum(np.exp((g - gmax) / tau)))
    if n1 * ninf < 1e-30:
        return -1e10
    return l2 / (n1 * ninf)


def _neg_smooth(h, tau=0.05):
    v = _smooth_value(h, tau)
    if v == -1e10:
        return 1e10
    return -v


def _lbfgs(h0, max_iter=400, tau=0.05):
    try:
        bounds = [(0.0, 25.0)] * N
        res = minimize(_neg_smooth, h0, args=(tau,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-9, 'maxls': 40})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _value(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _value(h0)


def _nelder_mead(h0, max_iter=250):
    try:
        res = minimize(_neg_smooth, h0, args=(0.02,), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-11})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _value(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _value(h0)


def _cd(h0, budget):
    """Coordinate descent polish with adaptive step."""
    h = np.maximum(np.array(h0, dtype=float), 0.0)
    best = _value(h)
    step = 0.3
    while time.time() < budget:
        improved = False
        for i in range(N):
            for d in (step, -step):
                cand = h.copy()
                cand[i] = max(0.0, cand[i] + d)
                v = _value(cand)
                if v > best + 1e-13:
                    h = cand
                    best = v
                    improved = True
        if not improved:
            step *= 0.65
            if step < 1e-8:
                break
    return h, best


def _make_guesses():
    t = np.linspace(-0.5, 0.5, N)
    guesses = []

    # Uniform
    guesses.append(np.ones(N))

    # Single Gaussians (varying width and center)
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]:
        for c in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            guesses.append(np.exp(-(t - c) ** 2 / (2 * sigma ** 2)))

    # Cosine bumps
    for width in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for c in [-0.2, 0.0, 0.2]:
            guesses.append(np.maximum(np.cos(np.pi * (t - c) / width), 0.0))

    # Tapered (linear, quadratic)
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        for c in [-0.2, 0.0, 0.2]:
            guesses.append(np.maximum(1.0 - a * np.abs(t - c), 0.0))
            guesses.append(np.maximum(1.0 - a * (t - c) ** 2, 0.0))

    # Power law
    for p in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for c in [-0.2, 0.0, 0.2]:
            guesses.append(np.maximum(1.0 - (2.0 * np.abs(t - c)) ** p, 0.0))

    # Exponential
    for a in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        for c in [-0.2, 0.0, 0.2]:
            guesses.append(np.exp(-a * np.abs(t - c)))
            guesses.append(np.exp(-a * (t - c) ** 2))

    # Lorentzian
    for c in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
        for cc in [-0.2, 0.0, 0.2]:
            guesses.append(1.0 / (1.0 + c * (t - cc) ** 2))

    # Sine-based
    for a in [0.5, 1.0, 2.0, 3.0, 4.0]:
        guesses.append(np.maximum(np.sin(np.pi * t) ** a, 0.0))
        guesses.append(np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0))

    # Two-bump shapes (symmetric and asymmetric)
    for d in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for sigma in [0.08, 0.1, 0.15, 0.2]:
            guesses.append(np.exp(-t ** 2 / (2 * sigma ** 2)) + np.exp(-(t - d) ** 2 / (2 * sigma ** 2)))
            guesses.append(np.exp(-t ** 2 / (2 * sigma ** 2)) + np.exp(-(t + d) ** 2 / (2 * sigma ** 2)))
            guesses.append(np.exp(-(t - d) ** 2 / (2 * sigma ** 2)) + np.exp(-(t + d) ** 2 / (2 * sigma ** 2)))

    # Random smooth guesses (Fourier basis)
    rng = np.random.RandomState(42)
    for _ in range(40):
        ncomp = rng.randint(3, 8)
        guess = np.zeros(N)
        for _k in range(ncomp):
            freq = rng.randint(1, 6)
            amp = rng.randn()
            phase = rng.rand() * 2 * np.pi
            guess += amp * np.cos(freq * np.pi * t + phase)
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)

    # Random Gaussian mixtures (2-3 components)
    for _ in range(40):
        ncomp = rng.randint(2, 4)
        guess = np.zeros(N)
        for _k in range(ncomp):
            c = rng.uniform(-0.4, 0.4)
            s = rng.uniform(0.05, 0.3)
            a = rng.uniform(0.3, 1.0)
            guess += a * np.exp(-(t - c) ** 2 / (2 * s ** 2))
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)

    return guesses


def construct():
    # Phase 1: Evaluate all initial guesses
    guesses = _make_guesses()
    candidates = []

    for g in guesses:
        g = np.asarray(g, dtype=float)
        if not np.all(np.isfinite(g)):
            continue
        g = np.maximum(g, 0.0)
        if np.max(g) < 1e-12:
            continue
        v = _value(g)
        if np.isfinite(v):
            candidates.append((v, g))

    if not candidates:
        return np.ones(N).tolist()

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Take top 8 for L-BFGS-B refinement
    top_k = candidates[:8]

    best_h = top_k[0][1]
    best_v = top_k[0][0]

    # Phase 2: L-BFGS-B from top 8 guesses
    refined = []
    for v, h in top_k:
        if time.time() > _DEADLINE - 50:
            break
        h_new, v_new = _lbfgs(h, max_iter=400, tau=0.05)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new
        refined.append((v_new, h_new))

    refined.sort(key=lambda x: x[0], reverse=True)
    top_k_refined = refined[:4]

    # Phase 3: Nelder-Mead on top 4 refined
    for v, h in top_k_refined:
        if time.time() > _DEADLINE - 25:
            break
        h_new, v_new = _nelder_mead(h, max_iter=250)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new

    # Phase 4: Final Coordinate Descent
    budget = min(time.time() + 12.0, _DEADLINE - 4.0)
    if time.time() < budget:
        h_new, v_new = _cd(best_h, budget)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new

    # Final sanity check
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)

    return final.tolist()
