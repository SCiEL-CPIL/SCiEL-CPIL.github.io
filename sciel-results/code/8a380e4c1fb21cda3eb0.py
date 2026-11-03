"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated linear overlap with Nesterov accelerated gradient
descent on a smooth-max surrogate, adaptive learning rate, projection onto
the constraint set, multiple diverse initializations, and a high-beta polish.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0

    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid

    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)

    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)

    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp

    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)

    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2

    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h

    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)

    conv = np.fft.irfft(W * H, n=n_fft)
    A = conv[:N]

    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)

    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]

    grad = -(2.0 / N) * (A + B)

    return val, grad


def _nesterov_accelerated_gradient(h0, beta, n_iter=300, lr=0.05):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with adaptive learning rate and backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    h_prev = h.copy()
    t = 1.0

    best_h = h.copy()
    best_score = _score(h)

    lr_current = lr
    max_backtracks = 12

    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new

        y = h + momentum * (h - h_prev)

        val_y, grad = _smooth_max_obj_grad(y, beta)

        step_lr = lr_current
        h_new = None
        for bt in range(max_backtracks):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            if val_candidate <= val_y + 1e-8:
                h_new = h_candidate
                lr_current = step_lr
                break
            step_lr *= 0.5

        if h_new is None:
            h_new = y - lr_current * 0.5 * grad
            h_new = _project(h_new, N / 2.0)
            lr_current *= 0.5

        h_prev = h
        h = h_new
        t = t_new

        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h, best_score


def _high_beta_polish(h0, n_iter=80, beta=200.0, lr=0.005):
    """
    Final high-precision polish pass with very high beta and small learning rate.
    Uses plain projected gradient descent for stability at high beta.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)

    best_h = h.copy()
    best_score = _score(h)

    for i in range(n_iter):
        if time.time() > _DEADLINE - 2:
            break

        val, grad = _smooth_max_obj_grad(h, beta)

        h_new = h - lr * grad
        h_new = _project(h_new, N / 2.0)

        current_score = _score(h_new)
        if current_score < best_score:
            best_score = current_score
            best_h = h_new.copy()
        h = h_new

    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # 1. Diverse initializations
    initializations = []

    # Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(h_cos)

    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(h_sin)

    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt[1::2] = 0.0
    initializations.append(h_alt)

    # Random
    for _ in range(3):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(h_rand)

    # Block patterns
    h_block1 = np.zeros(N)
    h_block1[:N//2] = 1.0
    initializations.append(h_block1)

    h_block2 = np.zeros(N)
    h_block2[N//2:] = 1.0
    initializations.append(h_block2)

    # 2. Run Nesterov gradient descent from each initialization
    for h0 in initializations:
        if time.time() > _DEADLINE - 20:
            break
        h0 = _project(h0, N / 2.0)
        _update_best(h0)
        h_opt, s_opt = _nesterov_accelerated_gradient(h0, beta=50.0, n_iter=200, lr=0.05)
        _update_best(h_opt)

    # 3. Final polish with high beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_polished, s_polished = _high_beta_polish(best_h, n_iter=80, beta=200.0, lr=0.005)
        _update_best(h_polished)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _project(best_h, N / 2.0)
    return best_h
