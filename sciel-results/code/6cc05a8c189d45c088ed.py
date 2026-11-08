"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0  # safety margin before the 150 s kill
_N = N


def _overlaps_fft(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k using FFT."""
    n = len(h)
    one_minus = 1.0 - h
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus)
    S = np.fft.ifft(np.conj(H) * O).real
    return S


def _softmax(S, beta):
    """Smooth max of S via log-sum-exp (scaled)."""
    m = np.max(S)
    return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta


def _project(h, n):
    """Project h onto [0,1]^n with sum = n/2 using iterative clipping."""
    h = h.copy()
    for _ in range(10):
        h = np.clip(h, 0.0, 1.0)
        h += (n / 2.0 - h.sum()) / n
    h = np.clip(h, 0.0, 1.0)
    # Final exact sum fix
    diff = n / 2.0 - h.sum()
    if abs(diff) > 1e-15:
        # Distribute the residual to elements not at bounds
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
            h += (n / 2.0 - h.sum()) / n
            h = np.clip(h, 0.0, 1.0)
    return h


def _gradient(h, S, beta, n):
    """Compute gradient of softmax(S, beta) w.r.t. h, projected to zero-sum."""
    S_max = np.max(S)
    e = np.exp(beta * (S - S_max))
    e_sum = e.sum()
    if e_sum < 1e-30:
        e = np.ones(n) / n
    else:
        e = e / e_sum
    g = np.zeros(n)
    for k in range(n):
        ek = e[k]
        if ek < 1e-14:
            continue
        g += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
    g -= g.mean()
    return g


def _line_search(h, g, lr, beta, n, max_iter=8):
    """Backtracking line search."""
    S_curr = _overlaps_fft(h)
    val_curr = _softmax(S_curr, beta)
    for i in range(max_iter):
        lr_try = lr / (2 ** i)
        h_try = _project(h - lr_try * g, n)
        S_try = _overlaps_fft(h_try)
        val_try = _softmax(S_try, beta)
        if val_try < val_curr:
            return h_try, val_try
    return h, val_curr


def _structured_init(n, pattern_type, rng):
    """Generate diverse structured initializations."""
    if pattern_type == 0:
        # Alternating
        h = np.full(n, 0.5)
        h[::2] = 0.7
        h[1::2] = 0.3
    elif pattern_type == 1:
        # Sine wave
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        h = 0.5 + 0.3 * np.sin(t)
    elif pattern_type == 2:
        # Two-frequency
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        h = 0.5 + 0.2 * np.sin(t) + 0.1 * np.sin(3 * t)
    elif pattern_type == 3:
        # Random
        h = rng.uniform(0, 1, n)
    else:
        # Slightly perturbed uniform
        h = np.full(n, 0.5)
        d = rng.standard_normal(n)
        d -= d.mean()
        h += 0.1 * d / (np.linalg.norm(d) + 1e-12)
    return _project(h, n)


def construct():
    t0 = time.time()
    n = _N
    rng = np.random.default_rng(42)

    best_h = np.full(n, 0.5)
    best_val = float('inf')

    # Phase 1: Multiple restarts with adaptive beta homotopy
    n_restarts = 6
    patterns = [0, 1, 2, 3, 4, 0]

    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 15.0:
            break

        h = _structured_init(n, patterns[restart % len(patterns)], rng)

        # Adaptive beta: start smooth, get sharper
        betas = [10.0, 30.0, 60.0, 100.0]
        lrs = [0.08, 0.04, 0.02, 0.01]
        steps_per_beta = [800, 800, 600, 500]

        for bi, beta in enumerate(betas):
            if time.time() > _DEADLINE - 10.0:
                break
            lr = lrs[bi]
            steps = steps_per_beta[bi]

            for it in range(steps):
                if time.time() > _DEADLINE - 8.0:
                    break

                S = _overlaps_fft(h)
                val = _softmax(S, beta)

                if val < best_val:
                    best_val = val
                    best_h = h.copy()

                g = _gradient(h, S, beta, n)
                h_new, val_new = _line_search(h, g, lr, beta, n)

                if val_new < val:
                    lr = min(lr * 1.15, 0.2)
                else:
                    lr = max(lr * 0.5, 0.001)

                h = h_new

    # Phase 2: High-precision polish on the best solution
    if time.time() < _DEADLINE - 10.0:
        h = best_h.copy()
        beta_final = 150.0
        lr_final = 0.005

        for it in range(2000):
            if time.time() > _DEADLINE - 5.0:
                break

            S = _overlaps_fft(h)
            val = _softmax(S, beta_final)

            if val < best_val:
                best_val = val
                best_h = h.copy()

            g = _gradient(h, S, beta_final, n)
            h_new, val_new = _line_search(h, g, lr_final, beta_final, n, max_iter=5)

            if val_new < val:
                lr_final = min(lr_final * 1.05, 0.03)
            else:
                lr_final = max(lr_final * 0.7, 0.0003)

            h = h_new

    # Phase 3: Final micro-polish with very high beta
    if time.time() < _DEADLINE - 5.0:
        h = best_h.copy()
        beta_micro = 300.0
        lr_micro = 0.002

        for it in range(1000):
            if time.time() > _DEADLINE - 3.0:
                break

            S = _overlaps_fft(h)
            val = _softmax(S, beta_micro)

            if val < best_val:
                best_val = val
                best_h = h.copy()

            g = _gradient(h, S, beta_micro, n)
            h_new, val_new = _line_search(h, g, lr_micro, beta_micro, n, max_iter=3)

            if val_new < val:
                lr_micro = min(lr_micro * 1.03, 0.01)
            else:
                lr_micro = max(lr_micro * 0.8, 0.0001)

            h = h_new

    # Enforce exact constraints one final time
    best_h = _project(best_h, n)

    return best_h
# EVOLVE-BLOCK-END
