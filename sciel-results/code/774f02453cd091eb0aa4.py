"""Erdos minimum overlap: combined structural search + accelerated projected gradient.
Strategy: Use the clipped-sinusoidal manifold (h_i = clip(A*cos(2*pi*k*i/N + phi) + 0.5, 0, 1))
as a low-dimensional structural initializer that naturally enforces bounds and has
interesting overlap profiles. Then refine with Nesterov accelerated projected gradient
on a sharp log-sum-exp surrogate of the max overlap, with bisection on the target value.
This combines the structural constraint idea (3D search) with the refinement machinery
(inspiration 1's accelerated PGD + bisection).
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145.0


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}.
    Uses FFT-based circular autocorrelation for O(N log N) evaluation."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap (the quantity we minimize)."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """Return (smooth_max_value, gradient) of max(C_k) using log-sum-exp.
    The gradient is computed analytically via the chain rule through the
    circular autocorrelation."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # Gradient of C_k w.r.t. h:
    # C_k = (2/N)*(N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    # dC_k/dh_j = -(2/N) * (h_{j-k} + h_{j+k}) for the circular version
    # More precisely, for circular autocorrelation r_k = sum_i h_i h_{(i-k) mod N}:
    # dr_k/dh_j = h_{(j-k) mod N} + h_{(j+k) mod N} (for k != 0)
    # dr_0/dh_j = 2*h_j
    # dC_k/dh_j = -(2/N) * dr_k/dh_j
    grad = np.zeros(n)
    for k in range(n):
        wk = w[k]
        if wk < 1e-15:
            continue
        factor = -(2.0 / n) * wk
        if k == 0:
            grad += factor * 2.0 * h
        else:
            # Circular shifts
            grad += factor * np.roll(h, k)
            grad += factor * np.roll(h, -k)
    return val, grad


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
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


def _accelerated_smooth_max(h0, beta, max_iter=500):
    """Nesterov accelerated projected gradient on smooth max with box+sum constraints."""
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)

    h_prev = h.copy()
    t_k = 1.0
    lr = 0.05

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

        # Adaptive learning rate: decrease slowly
        if it % 100 == 99:
            lr = max(0.005, lr * 0.85)

    return h, _score(h)


def _clipped_sin_init(freq, phase, amplitude):
    """Generate initial h from clipped sinusoid: h_i = clip(A*cos(2*pi*f*i/N + phi) + 0.5, 0, 1)."""
    i = np.arange(N)
    h = amplitude * np.cos(2.0 * np.pi * freq * i / N + phase) + 0.5
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(2024)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    i = np.arange(N)

    # Phase 1: Structural search over clipped sinusoidal manifold
    # h_i = clip(A*cos(2*pi*f*i/N + phi) + 0.5, 0, 1)
    # This gives diverse, bound-respecting starting points with interesting overlap structure.
    starts = []

    # Grid over frequency, phase, amplitude
    freqs = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
    phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4]
    amps = [0.3, 0.5, 0.7, 0.9, 1.0]

    for f in freqs:
        for phi in phases:
            for A in amps:
                h = _clipped_sin_init(f, phi, A)
                starts.append(h)

    # Add some random inits
    for _ in range(10):
        h_r = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_r))

    # Add cosine-basis init (from current program's approach)
    for K in [1, 2, 5, 10]:
        a = np.zeros(K)
        a[0] = 0.3
        if K > 1:
            a[1] = 0.1
        h = np.full(N, 0.5)
        for k in range(K):
            h += a[k] * np.cos(2 * np.pi * (k + 1) * i / N)
        h = np.clip(h, 0, 1)
        starts.append(_fix_sum(h))

    # Evaluate all starts
    for h0 in starts:
        _update_best(h0)

    # Phase 2: Bisection on target t with accelerated gradient feasibility
    # For each target t, try to find h with max_overlap <= t
    t_low = 0.0
    t_high = best_score

    # Sort starts by score for diversity in restarts
    scored_starts = [( _score(h), h) for h in starts]
    scored_starts.sort()

    for bisect_iter in range(30):
        if time.time() > _DEADLINE - 15:
            break

        t_mid = (t_low + t_high) / 2.0
        # Increase beta for sharper max approximation as we refine
        beta = 80.0 + bisect_iter * 20.0

        # Try accelerated gradient from best_h
        if best_h is not None:
            h_cand, s_cand = _accelerated_smooth_max(best_h, beta, max_iter=400)
            _update_best(h_cand)

            # Try from a few diverse starts
            for idx in [0, 5, 10, 20]:
                if idx < len(scored_starts):
                    h_start = scored_starts[idx][1]
                    h_cand2, s_cand2 = _accelerated_smooth_max(h_start, beta, max_iter=200)
                    _update_best(h_cand2)

            # Perturbed best
            h_pert = best_h + 0.03 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_cand3, s_cand3 = _accelerated_smooth_max(h_pert, beta, max_iter=200)
            _update_best(h_cand3)

        # Bisection update
        if best_score <= t_mid + 1e-6:
            t_high = t_mid
        else:
            t_low = t_mid

        if t_high - t_low < 1e-8:
            break

    # Phase 3: Final polish with very high beta
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_final, s_final = _accelerated_smooth_max(best_h, beta=500.0, max_iter=600)
        _update_best(h_final)

    # Phase 4: One more round of diverse refinement if time permits
    if time.time() < _DEADLINE - 10 and best_h is not None:
        for trial in range(5):
            if time.time() > _DEADLINE - 8:
                break
            # Random perturbation of best
            h_pert = best_h + 0.02 * np.random.randn(N)
            h_pert = _fix_sum(h_pert)
            h_cand, s_cand = _accelerated_smooth_max(h_pert, beta=300.0, max_iter=300)
            _update_best(h_cand)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
