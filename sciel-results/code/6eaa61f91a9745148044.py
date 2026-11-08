"""
Optimization of the second autocorrelation inequality constant.
Combines the scale-invariance insight with L-BFGS-B optimization on a 
log-sum-exp smoothed objective, followed by Nelder-Mead polishing 
on the exact objective.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    if np.sum(h) == 0:
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate using Log-Sum-Exp for norm_inf.
    Returns negative value (for minimization).
    """
    h = np.maximum(0.0, h)
    if np.sum(h) == 0:
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)

    if norm_1 <= 0:
        return 0.0

    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    
    # Numerically stable LSE
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0

    val = l2 / (norm_1 * lse)
    return -val


def _neg_exact_value(h):
    """Negative of exact value (for minimization)."""
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian
    for sigma in [N/8, N/4, N/6]:
        starts.append(np.exp(-0.5 * ((x - N / 2) / sigma) ** 2))
    # Two peaks
    for spread in [N/10, N/6]:
        p1 = np.exp(-0.5 * ((x - N * 0.25) / spread) ** 2)
        p2 = np.exp(-0.5 * ((x - N * 0.75) / spread) ** 2)
        starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Power-law profiles
    for p in [0.5, 1.5, 2.0]:
        starts.append(np.power(np.abs(2 * x / (N - 1) - 1), p))
    # Sparse spike patterns
    for frac in [0.1, 0.2, 0.3]:
        spike = np.zeros(N)
        idx = int(N * frac / 2)
        spike[N // 2 - idx:N // 2 + idx + 1] = 1.0
        starts.append(spike)
    # Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 10, size=5)
        phases = RNG.uniform(0, 2 * np.pi, size=5)
        amps = RNG.uniform(0.5, 1.0, size=5)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)

    best_val = 0.0
    best_h = np.ones(N) / N

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if np.sum(h) == 0:
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()

    # Initial evaluation
    for s in starts:
        track(s)

    bounds = [(0, None)] * N

    # Phase 1: L-BFGS-B with multiple beta values
    # Lower beta allows broader exploration, higher beta focuses on peak reduction
    betas = [10.0, 25.0, 50.0, 100.0, 200.0]
    
    for s in starts:
        if time.time() > deadline - 10:
            break
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        # Normalize start to avoid scaling issues in initial steps
        if np.sum(x0) > 0:
            x0 = x0 / np.sum(x0)
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < 0:
                    track(res.x)
            except Exception:
                pass

    # Phase 2: Nelder-Mead polish on exact objective
    # This helps refine the solution where the LSE approximation might be slightly suboptimal
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        scale = 0.05 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        if np.sum(pert) == 0:
            continue
        try:
            res = opt.minimize(
                _neg_exact_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            track(res.x)
        except Exception:
            pass

    # Phase 3: High-beta L-BFGS-B polish from best
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        pert = best_h + RNG.normal(0.0, 0.03, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 300.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
