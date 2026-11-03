"""Rewritten: spectral-flatness driven search + continuous-height refinement.

Key insight: for a sum-1 sequence, the zero-lag autoconvolution c0 = sum(x^2)
is always >= 1 (by Cauchy-Schwarz), and is exactly 1 iff |x_i| = 1/sqrt(n)
for all i.  The metric max|c_k|/S^2 therefore has a hard floor of 1/N,
attained by any +/-1 sequence.  The real optimization is to reduce the
OFF-diagonal peaks c_k (k != 0) below 1 while keeping the sequence near
unit-magnitude.

We do this by:
  1. Generating diverse +/-1 initializations (random, m-sequences, blocks,
     low-frequency sinusoids, Gaussian-smoothed signs).
  2. Minimizing the off-diagonal autoconvolution peaks via subgradient
     descent on the continuous relaxation, with a soft-max surrogate for
     the max to get smoother gradients.
  3. A final high-precision polish pass with a tighter tolerance.
"""
import numpy as np
import time

N = 400


def _autoconv_lin(x):
    """Full linear autoconvolution of x (length 2n-1)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c)))


def _autoconv_max_abs_offdiag(x):
    """Max |c_k| for k != 0 (off-diagonal peaks)."""
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:])))
    return 0.0


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _metric_offdiag(x):
    """max |off-diag autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs_offdiag(x) / (s * s)


def _norm_sum1(x):
    s = np.sum(x)
    if abs(s) < 1e-15:
        return x + 1e-9
    return x / s


def _softmax_max(c, beta=50.0):
    """Soft-max surrogate: (1/beta) * log(sum(exp(beta*|c_k|))) for k != 0."""
    abs_c = np.abs(c)
    # Exclude k=0
    if len(abs_c) > 1:
        abs_c = abs_c[1:]
    else:
        return 0.0
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return float((m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta))


def _refine_subgrad(x0, deadline, rng, iters=20000, use_softmax=False, beta=50.0):
    """Projected subgradient minimization of off-diagonal max|c| s.t. sum(x)=1.
    
    The ratio max|c|/S^2 is scale-invariant, so we fix sum=1 and minimize
    max|c_k| for k != 0.  The zero-lag c0 = sum(x^2) >= 1 by Cauchy-Schwarz
    and equals 1 iff |x_i| = 1/sqrt(N) for all i.
    """
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 5e-4
    no_improve = 0
    patience = 2000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        
        if use_softmax:
            # Use soft-max for smoother gradient
            abs_c = np.abs(c[1:])  # off-diagonal only
            m = np.max(abs_c)
            if m < 1e-15:
                break
            weights = np.exp(beta * (abs_c - m))
            weights /= weights.sum()
            # Gradient of soft-max w.r.t. c_k for k != 0
            g = np.zeros(n)
            # For each off-diagonal lag k, gradient of c_k w.r.t. x
            for k in range(1, n):
                wk = weights[k - 1]
                if wk < 1e-8:
                    continue
                sign_ck = np.sign(c[k])
                # gradient of c_k = sum_i x_i x_{i+k} w.r.t. x
                # dc_k/dx_i = x_{i+k} for i < n-k, x_{i-k} for i >= k
                if k < n:
                    g[:n - k] += wk * sign_ck * x[k:]
                    g[k:] += wk * sign_ck * x[:n - k]
        else:
            # Hard max: pick the worst off-diagonal lag
            abs_c = np.abs(c[1:])
            k = int(np.argmax(abs_c)) + 1  # +1 because we excluded k=0
            ck = float(c[k])
            g = np.zeros(n)
            sign_ck = np.sign(ck)
            if k < n:
                g[:n - k] += sign_ck * x[k:]
                g[k:] += sign_ck * x[:n - k]
        
        # Step
        x = x - lr * g
        # Project to sum=1
        x = x - (np.sum(x) - 1.0) / n
        # Keep magnitudes bounded to avoid blowup
        x = np.clip(x, -2.0, 2.0)
        # Re-project after clip
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.03, 2e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.85
                no_improve = 0
                if lr < 1e-6:
                    break
    return bestx, best


def _refine_polish(x0, deadline, rng, iters=30000):
    """High-precision polish: very small steps, tight patience."""
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-5
    no_improve = 0
    patience = 5000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        k = int(np.argmax(abs_c)) + 1
        ck = float(c[k])
        g = np.zeros(n)
        sign_ck = np.sign(ck)
        if k < n:
            g[:n - k] += sign_ck * x[k:]
            g[k:] += sign_ck * x[:n - k]
        
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        x = np.clip(x, -1.5, 1.5)
        x = x - (np.sum(x) - 1.0) / n
        
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.02, 5e-5)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-8:
                    break
    return bestx, best


def _snap_and_refine(x0, deadline, rng, iters=20000):
    """Snap to +/-1, refine, snap, refine (alternating)."""
    x = x0.copy()
    best = _metric(x)
    bestx = x.copy()
    for _ in range(3):
        if time.time() > deadline:
            break
        # Snap to +/-1 with normalization
        xs = np.where(x >= 0, 1.0, -1.0)
        s = np.sum(xs)
        if abs(s) > 1e-9:
            xs = xs / s
        x, v = _refine_subgrad(xs, deadline, rng, iters)
        if v < best:
            best = v
            bestx = x.copy()
    return bestx, best


def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])


def _try_m_sequences(deadline, rng):
    """m-sequence based shapes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3],
             [8, 4], [9, 5], [10, 7], [11, 6], [12, 8], [13, 4], [14, 10], [15, 12]]
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(8, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            v = _metric(seq)
            if v < best_val:
                best_val = v
                best_x = seq.copy()
    return best_x, best_val


def _try_block_patterns(deadline, rng):
    """Block / run patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for block_size in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if time.time() > deadline:
            break
        if block_size > N:
            continue
        n_blocks = N // block_size
        rem = N % block_size
        # Deterministic alternating
        x = np.zeros(N)
        for i in range(n_blocks):
            x[i * block_size:(i + 1) * block_size] = 1.0 if i % 2 == 0 else -1.0
        if rem:
            x[n_blocks * block_size:] = 1.0 if n_blocks % 2 == 0 else -1.0
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        # Random blocks
        for _ in range(5):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            for i in range(n_blocks):
                x[i * block_size:(i + 1) * block_size] = float(rng.choice([-1.0, 1.0]))
            if rem:
                x[n_blocks * block_size:] = float(rng.choice([-1.0, 1.0]))
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val


def _try_gaussian_smoothed(deadline, rng):
    """Gaussian-smoothed sign patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for sigma in [2, 4, 6, 10, 15, 25, 40, 60]:
        if time.time() > deadline:
            break
        for _ in range(8):
            if time.time() > deadline:
                break
            raw = rng.choice([-1.0, 1.0], size=N)
            kernel_size = min(61, N // 2)
            half = kernel_size // 2
            kernel = np.exp(-0.5 * (np.arange(-half, half + 1)) ** 2 / sigma ** 2)
            kernel = kernel / kernel.sum()
            size = 1
            while size < 2 * N:
                size *= 2
            sm = np.fft.irfft(np.fft.rfft(raw, size) * np.fft.rfft(kernel, size), size)[:N]
            x = np.where(sm >= 0, 1.0, -1.0)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val


def _try_fourier(deadline, rng):
    """Sinusoidal / low-frequency sign patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    idx = np.arange(N)
    for f in np.arange(0.5, 50.0, 0.5):
        if time.time() > deadline:
            break
        for phase in [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            if time.time() > deadline:
                break
            x = np.where(np.sin(2 * np.pi * f * idx / N + phase) >= 0, 1.0, -1.0)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val


def _try_random(deadline, rng, trials=200):
    """Uniform random +/-1 shapes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for _ in range(trials):
        if time.time() > deadline:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    return best_x, best_val


def _try_local_search(x, deadline, rng, max_rounds=50000):
    """Local search: flip a few signs and accept improvements."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        k = int(rng.integers(1, min(4, N // 4) + 1))
        idx = rng.choice(N, size=k, replace=False)
        xf = best_x.copy()
        xf[idx] *= -1.0
        v = _metric(xf)
        if v < best_val:
            best_val = v
            best_x = xf
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 2000:
                break
    return best_x, best_val


def _try_simulated_annealing(x, deadline, rng, max_rounds=30000):
    """Simulated annealing on signs."""
    best_x = x.copy()
    best_val = _metric(best_x)
    cur_x = x.copy()
    cur_val = _metric(cur_x)
    T = 0.03
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        k = int(rng.integers(1, min(4, N // 4) + 1))
        idx = rng.choice(N, size=k, replace=False)
        xn = cur_x.copy()
        xn[idx] *= -1.0
        vn = _metric(xn)
        d = vn - cur_val
        if d < 0 or (T > 1e-6 and rng.random() < np.exp(-d / T)):
            cur_x = xn
            cur_val = vn
            if vn < best_val:
                best_val = vn
                best_x = xn.copy()
        T *= 0.9998
    return best_x, best_val


def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = np.random.default_rng(2024)

    best_x = np.ones(N)
    best_val = _metric(best_x)

    def consider(x, v):
        nonlocal best_x, best_val
        if v < best_val:
            best_val = v
            best_x = x.copy()

    # Phase 1: Generate diverse +/-1 initializations
    t_gen = deadline - 30  # leave 30s for refinement
    sources = [
        lambda: _try_random(t_gen, rng, 150),
        lambda: _try_block_patterns(t_gen, rng),
        lambda: _try_fourier(t_gen, rng),
        lambda: _try_gaussian_smoothed(t_gen, rng),
        lambda: _try_m_sequences(t_gen, rng),
    ]
    for src in sources:
        if time.time() > t_gen - 5:
            break
        x, v = src()
        consider(x, v)

    # Phase 2: Continuous-height subgradient refinement (hard max)
    if time.time() < deadline - 10:
        x, v = _refine_subgrad(best_x, deadline - 10, rng, iters=50000, use_softmax=False)
        consider(x, v)

    # Phase 3: Soft-max refinement for smoother gradients
    if time.time() < deadline - 8:
        x, v = _refine_subgrad(best_x, deadline - 8, rng, iters=30000, use_softmax=True, beta=30.0)
        consider(x, v)

    # Phase 4: Snap + refine cycle
    if time.time() < deadline - 5:
        x, v = _snap_and_refine(best_x, deadline - 5, rng, iters=30000)
        consider(x, v)

    # Phase 5: High-precision polish
    if time.time() < deadline - 2:
        x, v = _refine_polish(best_x, deadline - 2, rng, iters=20000)
        consider(x, v)

    # Phase 6: Final local search on signs
    if time.time() < deadline - 1:
        x, v = _try_local_search(np.where(best_x >= 0, 1.0, -1.0), deadline - 1, rng, 15000)
        consider(x, v)

    return best_x
