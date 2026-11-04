import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor)."""
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) using hard max."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax = w * c.max()
    if acmax <= 0:
        return 0.0
    return s2 / (s1 * acmax)


def _soft_max(c, beta):
    """Soft max of c using log-sum-exp trick."""
    cmax = c.max()
    if cmax <= 0:
        return 0.0
    return cmax + np.log(np.sum(np.exp((c - cmax) / beta)) * beta)


def _ratio_smooth(h, beta):
    """Smoothed ratio for gradient-based optimization."""
    h = np.maximum(h, 1e-12)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax_soft = w * _soft_max(c, beta)
    if acmax_soft <= 0:
        return 0.0
    return s2 / (s1 * acmax_soft)


def _neg_ratio_smooth(z, beta):
    """Negative smooth ratio for minimization, where h = |z|^2."""
    h = np.abs(z) ** 2
    return -_ratio_smooth(h, beta)


def _neg_ratio_grad(z, beta, eps=1e-5):
    """Compute negative smooth ratio and gradient via finite differences for complex z."""
    z = np.asarray(z, dtype=complex)
    f0 = _neg_ratio_smooth(z, beta)
    grad = np.zeros_like(z)
    for i in range(N):
        # Real part
        z[i] += eps
        f1 = _neg_ratio_smooth(z, beta)
        z[i] -= eps
        grad[i] += (f1 - f0) / eps
        # Imaginary part
        z[i] += 1j * eps
        f2 = _neg_ratio_smooth(z, beta)
        z[i] -= 1j * eps
        grad[i] += (f2 - f0) / eps / (1j)
    return f0, grad


def _find_sidon_set(n, rng):
    """Find a Sidon set (B2 sequence) of size n in [0, N)."""
    # Try random sets and check Sidon property
    for _ in range(10000):
        s = np.sort(rng.choice(N, size=n, replace=False))
        # Check all pairwise sums are distinct
        sums = set()
        valid = True
        for i in range(n):
            for j in range(i, n):
                sm = s[i] + s[j]
                if sm in sums:
                    valid = False
                    break
                sums.add(sm)
            if not valid:
                break
        if valid:
            return s
    return None


def _spike_pattern(support, amplitude=1.0):
    """Create a spike pattern on the given support."""
    h = np.zeros(N)
    for idx in support:
        h[int(idx)] = amplitude
    return h


def _dft_parameterize(z):
    """Convert complex vector z to non-negative heights via |z|^2, normalized to sum=1."""
    h = np.abs(z) ** 2
    s = h.sum()
    if s > 0:
        h = h / s
    return h


def _random_complex(rng):
    """Generate a random complex vector."""
    return rng.standard_normal(N) + 1j * rng.standard_normal(N)


def _optimize_complex(z0, beta, max_iters=200, lr_init=0.01):
    """Gradient ascent on complex vector to maximize ratio."""
    z = np.array(z0, dtype=complex)
    best_z = z.copy()
    best_f = _ratio_smooth(_dft_parameterize(z), beta)
    lr = lr_init

    for it in range(max_iters):
        if _elapsed() > DEADLINE - 5:
            break

        f, grad = _neg_ratio_grad(z, beta)
        neg_f = -f  # We want to minimize neg_ratio, so maximize ratio

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            step = grad / grad_norm * lr
            z_new = z - step  # Gradient descent on neg_ratio
            h_new = _dft_parameterize(z_new)
            f_new = _ratio_smooth(h_new, beta)

            if f_new > best_f:
                best_f = f_new
                best_z = z_new.copy()
                z = z_new
                lr = min(lr * 1.05, 0.1)
            else:
                lr *= 0.8
        else:
            break

    return best_z, best_f


def _local_optimize_heights(h0, beta, max_iters=300, lr_init=0.01):
    """Local optimization directly on heights with non-negativity constraint."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 1e-12)
    best_h = h.copy()
    best_f = _ratio_smooth(h, beta)
    lr = lr_init

    for it in range(max_iters):
        if _elapsed() > DEADLINE - 5:
            break

        f0 = _ratio_smooth(h, beta)
        grad = np.zeros(N)
        eps = 1e-5
        for i in range(N):
            h[i] += eps
            f1 = _ratio_smooth(h, beta)
            h[i] -= eps
            grad[i] = (f1 - f0) / eps

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            step = grad / grad_norm * lr
            h_new = np.maximum(h + step, 0.0)
            f_new = _ratio_smooth(h_new, beta)

            if f_new > best_f:
                best_f = f_new
                best_h = h_new.copy()
                h = h_new
                lr = min(lr * 1.05, 0.1)
            else:
                lr *= 0.8
        else:
            break

    return best_h, best_f


def construct():
    best_h = np.ones(N) / N
    best_f = _ratio(best_h)

    rng = np.random.default_rng(42)

    # Phase 1: Try Sidon-set spike patterns
    for n_spikes in [5, 7, 10, 12, 15, 20]:
        if _elapsed() > DEADLINE - 40:
            break
        sidon = _find_sidon_set(n_spikes, rng)
        if sidon is not None:
            h = _spike_pattern(sidon)
            f = _ratio(h)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Phase 2: Try DFT-based random initializations
    for _ in range(50):
        if _elapsed() > DEADLINE - 30:
            break
        z = _random_complex(rng)
        h = _dft_parameterize(z)
        f = _ratio(h)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 3: Try smooth profiles
    for _ in range(50):
        if _elapsed() > DEADLINE - 25:
            break
        kind = rng.integers(0, 3)
        if kind == 0:
            center = rng.uniform(0, N)
            sigma = rng.uniform(2, 10)
            amp = rng.uniform(0.5, 2.0)
            x = np.arange(N)
            h = amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        elif kind == 1:
            center = rng.uniform(0, N)
            gamma = rng.uniform(2, 8)
            amp = rng.uniform(0.5, 2.0)
            x = np.arange(N)
            h = amp / (1 + ((x - center) / gamma) ** 2)
        else:
            c1 = rng.uniform(0, N)
            c2 = rng.uniform(0, N)
            s1 = rng.uniform(2, 6)
            s2 = rng.uniform(2, 6)
            a1 = rng.uniform(0.3, 1.0)
            a2 = rng.uniform(0.3, 1.0)
            x = np.arange(N)
            h = a1 * np.exp(-0.5 * ((x - c1) / s1) ** 2) + a2 * np.exp(-0.5 * ((x - c2) / s2) ** 2)
        h = np.maximum(h, 0.0)
        f = _ratio(h)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 4: Optimize from best height profile using complex parameterization
    if _elapsed() < DEADLINE - 15:
        # Initialize complex vector from best_h
        z_init = np.sqrt(best_h) * np.exp(1j * rng.standard_normal(N))
        for beta in [5.0, 10.0, 20.0]:
            if _elapsed() > DEADLINE - 10:
                break
            z_opt, f_opt = _optimize_complex(z_init, beta, max_iters=150, lr_init=0.01)
            h_opt = _dft_parameterize(z_opt)
            f_hard = _ratio(h_opt)
            if f_hard > best_f:
                best_f = f_hard
                best_h = h_opt.copy()
                z_init = z_opt.copy()

    # Phase 5: Local optimization on heights
    if _elapsed() < DEADLINE - 10:
        for beta in [5.0, 10.0, 20.0]:
            if _elapsed() > DEADLINE - 5:
                break
            h_opt, f_opt = _local_optimize_heights(best_h, beta, max_iters=200, lr_init=0.01)
            f_hard = _ratio(h_opt)
            if f_hard > best_f:
                best_f = f_hard
                best_h = h_opt.copy()

    # Phase 6: Try more Sidon sets with optimization
    if _elapsed() < DEADLINE - 10:
        for n_spikes in [5, 7, 10]:
            if _elapsed() > DEADLINE - 5:
                break
            for _ in range(20):
                if _elapsed() > DEADLINE - 5:
                    break
                sidon = _find_sidon_set(n_spikes, rng)
                if sidon is not None:
                    h = _spike_pattern(sidon)
                    # Optimize from this spike pattern
                    if _elapsed() < DEADLINE - 5:
                        h_opt, f_opt = _local_optimize_heights(h, 10.0, max_iters=100, lr_init=0.01)
                        f_hard = _ratio(h_opt)
                        if f_hard > best_f:
                            best_f = f_hard
                            best_h = h_opt.copy()

    # Phase 7: Final refinement
    if _elapsed() < DEADLINE - 5:
        h_opt, f_opt = _local_optimize_heights(best_h, 15.0, max_iters=150, lr_init=0.005)
        f_hard = _ratio(h_opt)
        if f_hard > best_f:
            best_f = f_hard
            best_h = h_opt.copy()

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
