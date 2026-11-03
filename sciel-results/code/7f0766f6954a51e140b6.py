"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def construct():
    # Deadline guard: return with a safety margin before the 150 s kill.
    deadline = time.time() + 130.0

    # The constant sequence attains the Cauchy-Schwarz lower bound:
    #   max_k |A_k| >= A_0 = sum x_i^2 >= (sum x_i)^2 / N,
    # so  max_k |A_k| / (sum x_i)^2  >=  1/N,
    # with equality for any constant sequence.  Start from all-ones.
    best_x = np.ones(N)
    best_val = _metric(best_x)

    rng = np.random.default_rng(2024)

    # Candidate generators for broad search
    def gen_pm1():
        return rng.choice([-1.0, 1.0], size=N)

    def gen_gauss():
        return rng.standard_normal(N)

    def gen_biased():
        p = rng.uniform(0.05, 0.5)
        return rng.choice([-1.0, 1.0], size=N, p=[p, 1 - p])

    def gen_smoothed():
        x = rng.standard_normal(N)
        kernel = np.ones(5) / 5
        return np.convolve(x, kernel, mode='same')

    generators = [gen_pm1, gen_gauss, gen_biased, gen_smoothed]

    # Phase 1: broad random search
    iter_count = 0
    phase1_deadline = deadline - 60.0
    while time.time() < phase1_deadline:
        for gen in generators:
            if time.time() >= phase1_deadline:
                break
            x = gen()
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
            iter_count += 1

    # Phase 2: local optimization (sign-flip / small perturbations) on best
    x = best_x.copy()
    v = best_val
    improved = True
    max_inner = 50
    inner_iter = 0
    phase2_deadline = deadline - 10.0
    while improved and time.time() < phase2_deadline and inner_iter < max_inner:
        improved = False
        inner_iter += 1
        for i in range(N):
            if time.time() >= phase2_deadline:
                break
            old = x[i]
            x[i] = -old
            v2 = _metric(x)
            if v2 < v:
                v = v2
                improved = True
            else:
                x[i] = old
            if v < best_val:
                best_val = v
                best_x = x.copy()

    # Phase 3: small Gaussian perturbations on best
    x = best_x.copy()
    v = best_val
    phase3_deadline = deadline - 5.0
    for _ in range(200):
        if time.time() >= phase3_deadline:
            break
        x_pert = x + rng.normal(0, 0.1, size=N)
        v2 = _metric(x_pert)
        if v2 < v:
            v = v2
            x = x_pert.copy()
            if v < best_val:
                best_val = v
                best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
