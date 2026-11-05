"""Optimized step-height search for third autocorrelation inequality."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400


def _value(h):
    s = float(h.sum())
    if abs(s) < 1e-12:
        return 1e18
    c = np.convolve(h, h)
    return abs(2.0 * N * float(c.max()) / (s * s))


def _qr_sequence(p):
    seq = np.zeros(p, dtype=float)
    for i in range(1, p):
        seq[i] = 1.0 if pow(i, (p - 1) // 2, p) == 1 else -1.0
    seq[0] = 1.0
    return seq


def _smooth_obj(h, beta=50.0):
    s = float(h.sum())
    if abs(s) < 1e-6:
        return 1e18
    c = np.convolve(h, h)
    c_max = c.max()
    lse = c_max + np.log(np.sum(np.exp((c - c_max) * beta))) / beta
    return abs(2.0 * N * lse / (s * s))


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    rng = np.random.default_rng(2024)

    best_h = np.ones(N, dtype=float)
    best_val = 2.0

    # --- Phase 1: QR sequence grid search over shifts and sign-flip masks ---
    primes_to_try = [397, 401, 409, 389, 383, 419, 379, 431, 373, 433]

    for p in primes_to_try:
        if time.time() > deadline - 50:
            break
        qr = _qr_sequence(p)
        qr_ext = np.tile(qr, 3)

        for shift in range(0, min(p, 40)):
            if time.time() > deadline - 50:
                break
            base = qr_ext[shift:shift + N].copy()
            v = _value(base)
            if v < best_val:
                best_val = v
                best_h = base.copy()

            # Try all-positive
            h_pos = np.ones(N, dtype=float)
            v_pos = _value(h_pos)
            if v_pos < best_val:
                best_val = v_pos
                best_h = h_pos.copy()

            # Random sign flips to adjust sum and peak
            for trial in range(20):
                if time.time() > deadline - 50:
                    break
                h = base.copy()
                k = rng.integers(1, 11)
                idxs = rng.choice(N, size=k, replace=False)
                h[idxs] = -h[idxs]
                if abs(h.sum()) < 2.0:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # --- Phase 2: Smooth profile initialization ---
    if time.time() < deadline - 40:
        x = np.arange(N)
        for f in range(1, 12):
            if time.time() > deadline - 40:
                break
            for a in np.linspace(0, 1.0, 10):
                for phase in np.linspace(0, 2 * np.pi, 4):
                    h = 1.0 + a * np.cos(2 * np.pi * f * x / N + phase)
                    if abs(h.sum()) < 1.0:
                        continue
                    v = _value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()

        for a in np.linspace(-0.5, 0.5, 8):
            for b in np.linspace(-0.3, 0.3, 8):
                if time.time() > deadline - 40:
                    break
                t = 2 * x / N - 1
                h = 1.0 + a * t + b * t * t
                if abs(h.sum()) < 1.0:
                    continue
                v = _value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()

    # --- Phase 3: Gradient-based optimization with multiple starts ---
    if time.time() < deadline - 10:
        inits = [best_h.copy()]
        for p in [397, 401]:
            qr = _qr_sequence(p)
            inits.append(np.tile(qr, 2)[:N].copy())
        x = np.arange(N)
        inits.append(1.0 + 0.5 * np.cos(2 * np.pi * x / N))
        inits.append(1.0 + 0.3 * np.sin(4 * np.pi * x / N))
        inits.append(1.0 + 0.4 * np.cos(2 * np.pi * 3 * x / N + 0.5))

        for init in inits:
            if time.time() > deadline - 10:
                break
            if abs(init.sum()) < 1.0:
                continue
            try:
                res = minimize(
                    _smooth_obj,
                    init,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-5, 'maxfun': 3000}
                )
                h_opt = res.x
                v = _value(h_opt)
                if v < best_val and np.all(np.isfinite(h_opt)) and abs(h_opt.sum()) > 1e-10:
                    best_val = v
                    best_h = h_opt.copy()
            except Exception:
                pass

    # --- Phase 4: Fine coordinate refinement ---
    if time.time() < deadline - 3:
        current = best_h.copy()
        current_val = best_val
        step_size = 0.2
        stall = 0
        max_stall = 300

        while time.time() < deadline - 2:
            idx = rng.integers(0, N)
            old_val = current[idx]
            current[idx] = old_val + rng.normal(0, step_size)

            new_val = _value(current)
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current.copy()
                stall = 0
                step_size = max(0.05, step_size * 1.02)
            else:
                current[idx] = old_val
                stall += 1
                if stall % 50 == 0:
                    step_size = max(0.05, step_size * 0.9)
                if stall > max_stall:
                    current = best_h + rng.normal(0, 0.1, size=N)
                    current_val = _value(current)
                    if not np.isfinite(current_val):
                        current = best_h.copy()
                        current_val = best_val
                    stall = 0
                    max_stall = int(max_stall * 1.5)

    if not np.all(np.isfinite(best_h)) or abs(best_h.sum()) < 1e-10:
        return np.ones(N)

    return best_h
