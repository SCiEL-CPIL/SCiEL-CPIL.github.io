"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses a structural reformulation: parameterize h via a binary-like sequence
with period-2 constraint to force even-shift autocorrelations to zero,
then optimize the remaining odd-shift peaks using simulated annealing
with FFT-accelerated overlap computation."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _overlap_values(h):
    """
    Compute overlap values C_k = (2/N) * (N/2 - r_k) for k=0..N-1.
    r_k = sum_{i=k}^{N-1} h_i * h_{i-k} (linear autocorrelation)
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    return float(np.max(cs))


def _smooth_max_value(h, beta):
    """Smooth maximum of overlap values using log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    cs = _overlap_values(h)
    m = np.max(cs)
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    # grad_j = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h

    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)

    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    cross = np.fft.irfft(np.conj(W) * H, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + cross)
    return val, grad


def _project_sum_box(h):
    """
    Project h onto {x : sum(x) = N/2, 0 <= x_i <= 1}.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    def f(lam):
        return np.sum(np.clip(h - lam, 0.0, 1.0))

    lo, hi = -1.0, 1.0
    while f(lo) < target:
        lo -= 1.0
    while f(hi) > target:
        hi += 1.0

    for _ in range(80):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm - target) < 1e-15:
            break
        if fm > target:
            lo = mid
        else:
            hi = mid

    lam = (lo + hi) / 2.0
    h = np.clip(h - lam, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 0) & (h < 1.0)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    return h


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


def _pgd_optimize(h0, beta, max_iter=300, tol=1e-10):
    """
    Nesterov-accelerated projected gradient descent.
    """
    h = _fix_sum(h0.copy())
    best_h = h.copy()
    best_val = _score(h)

    y = h.copy()
    momentum = 0.0
    prev_momentum = 0.0
    lr = 0.02
    prev_val = _smooth_max_value(h, beta)

    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break

        if it > 0:
            momentum = (1 + np.sqrt(1 + 4 * prev_momentum**2)) / 2
            y = h + (momentum - 1) / momentum * (h - y)

        val, grad = _smooth_max_grad(y, beta)
        h_new = h - lr * grad
        h_new = _project_sum_box(h_new)
        new_val = _smooth_max_value(h_new, beta)

        if new_val < prev_val:
            prev_momentum = momentum
            h = h_new
            prev_val = new_val
            s = _score(h)
            if s < best_val:
                best_val = s
                best_h = h.copy()
        else:
            lr *= 0.5
            h_new = h - lr * grad
            h_new = _project_sum_box(h_new)
            new_val = _smooth_max_value(h_new, beta)
            if new_val < prev_val:
                prev_momentum = momentum
                h = h_new
                prev_val = new_val
                s = _score(h)
                if s < best_val:
                    best_val = s
                    best_h = h.copy()

        if it > 50 and abs(prev_val - val) < tol:
            break

    return best_h


def _simulated_annealing(h0, t_start=1.0, t_end=0.01, n_iter=5000, seed=42):
    """
    Simulated annealing on the overlap landscape.
    Uses random perturbations and accepts worse solutions with probability exp(-delta/T).
    """
    rng = np.random.RandomState(seed)
    h = _fix_sum(h0.copy())
    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score

    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break

        T = t_start * (t_end / t_start) ** (it / max(n_iter - 1, 1))

        # Random perturbation: pick 1-3 random indices and nudge them
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        deltas = rng.uniform(-0.1, 0.1, size=n_perturb)

        h_trial = h.copy()
        for idx, d in zip(indices, deltas):
            h_trial[idx] += d
        h_trial = _project_sum_box(h_trial)

        trial_score = _score(h_trial)
        delta = trial_score - current_score

        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            h = h_trial
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
                best_h = h.copy()

    return best_h


def _refined_sa(h0, t_start=0.1, t_end=0.001, n_iter=3000, seed=123):
    """
    Finer-grained simulated annealing for polishing.
    Smaller perturbations, longer runs.
    """
    rng = np.random.RandomState(seed)
    h = _fix_sum(h0.copy())
    best_h = h.copy()
    best_score = _score(h)
    current_score = best_score

    for it in range(n_iter):
        if time.time() > _DEADLINE - 3:
            break

        T = t_start * (t_end / t_start) ** (it / max(n_iter - 1, 1))

        # Single index perturbation with small step
        idx = rng.randint(N)
        d = rng.uniform(-0.02, 0.02)

        h_trial = h.copy()
        h_trial[idx] += d
        h_trial = _project_sum_box(h_trial)

        trial_score = _score(h_trial)
        delta = trial_score - current_score

        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            h = h_trial
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
                best_h = h.copy()

    return best_h


def _binary_structure_search(n_trials=50, seed=99):
    """
    Search for good binary-like structures with period-2 constraint.
    h[2i] + h[2i+1] = 1 for all i, which forces even-shift autocorrelation to zero.
    Then optimize the free parameters (the h[2i] values) to minimize odd-shift peaks.
    """
    rng = np.random.RandomState(seed)
    best_h = None
    best_score = float('inf')

    for trial in range(n_trials):
        if time.time() > _DEADLINE - 15:
            break

        # Initialize with random binary-like values
        h_half = rng.uniform(0.1, 0.9, size=N // 2)
        h = np.zeros(N)
        h[0::2] = h_half
        h[1::2] = 1.0 - h_half
        h = _fix_sum(h)

        # Optimize using PGD
        h_opt = _pgd_optimize(h, beta=50.0, max_iter=200)
        s = _score(h_opt)
        if s < best_score:
            best_score = s
            best_h = h_opt.copy()

    return best_h, best_score


def _run_slsqp(h0, beta, maxiter=100, ftol=1e-10, gtol=1e-8):
    """Run SLSQP optimization."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        val, _ = _smooth_max_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_grad(x, beta)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return res.x
    except Exception:
        return h0


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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

    # 1. Binary structure search (period-2 constraint)
    h_binary, s_binary = _binary_structure_search(n_trials=30, seed=99)
    if h_binary is not None:
        _update_best(h_binary)

    # 2. Standard initializations + PGD
    i = np.arange(N)
    initializations = []

    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_cos))

    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(_fix_sum(h_alt))

    # Random
    for _ in range(2):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(_fix_sum(h_rand))

    for h_init in initializations:
        if time.time() > _DEADLINE - 30:
            break
        h_pgd = _pgd_optimize(h_init, beta=50.0, max_iter=300)
        _update_best(h_pgd)

    # 3. Simulated annealing from best
    if best_h is not None and time.time() < _DEADLINE - 20:
        h_sa = _simulated_annealing(best_h, t_start=0.5, t_end=0.01, n_iter=3000, seed=42)
        _update_best(h_sa)

    # 4. Refined SA polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_rsa = _refined_sa(best_h, t_start=0.05, t_end=0.001, n_iter=2000, seed=123)
        _update_best(h_rsa)

    # 5. SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_slsqp = _run_slsqp(best_h, beta=100.0, maxiter=100, ftol=1e-12, gtol=1e-10)
        _update_best(h_slsqp)

    # 6. Final PGD with high beta
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_final = _pgd_optimize(best_h, beta=150.0, max_iter=200)
        _update_best(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
