"""
Refined L-BFGS-B + log-sum-exp smoothing for the second autocorrelation inequality.
Key improvements over the parent:
  1. Epigraph reformulation: introduce t >= max(g) as an explicit variable,
     optimizing l2 / (norm_1 * t) subject to t >= max(g). This is handled
     via a penalty-free approach where we jointly optimize (h, t).
  2. Multi-scale beta schedule with more aggressive refinement.
  3. Better starting points including structured low-rank profiles.
  4. Final high-precision Nelder-Mead polish on the exact objective.
  5. Time-budget-aware: all loops check a deadline.
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
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
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
    Smooth surrogate: minimize -l2 / (norm_1 * lse(g)).
    lse(g) approximates max(g) smoothly.
    """
    h = np.maximum(0.0, h)
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
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0
    val = l2 / (norm_1 * lse)
    return -val

def _epigraph_objective(x, beta=100.0):
    """
    Epigraph-style objective: x = [h (N), t (1)].
    Objective: -l2 / (norm_1 * t) + penalty * sum(relu(g - t)^2)
    This encourages t >= max(g) while maximizing the ratio.
    """
    h = np.maximum(0.0, x[:N])
    t = x[N]
    if t <= 0:
        t = 1e-8
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 1e6
    # Penalty for violating t >= max(g)
    violation = np.maximum(0.0, g - t)
    penalty = 1e4 * np.sum(violation ** 2)
    val = l2 / (norm_1 * t)
    return -val + penalty

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Comb: alternating
    comb = np.ones(N)
    comb[::2] = 0.1
    starts.append(comb)
    # Sparse peaks every 10
    sparse = np.zeros(N)
    sparse[::10] = 1.0
    starts.append(sparse)
    # Power-law decay
    center = N / 2
    starts.append(np.maximum(0.05, 1.0 / (1.0 + np.abs(x - center) ** 1.5)))
    # Low-rank DCT-like: sum of a few cosines
    for k in [1, 2, 3, 5]:
        prof = np.sum(np.cos(2 * np.pi * j * x / N) for j in range(1, k + 1))
        starts.append(np.maximum(0.05, prof))
    # Random smooth profiles
    for _ in range(25):
        freqs = RNG.randint(1, 10, size=7)
        phases = RNG.uniform(0, 2 * np.pi, size=7)
        amps = RNG.uniform(0.5, 1.5, size=7)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.05, prof))
    # Random sparse spikes
    for _ in range(15):
        spike = np.zeros(N)
        n_spikes = RNG.randint(3, 10)
        positions = RNG.choice(N, size=n_spikes, replace=False)
        heights = RNG.uniform(0.5, 2.0, size=n_spikes)
        spike[positions] = heights
        spike += RNG.uniform(0.01, 0.1, N)
        starts.append(spike)
    # Bimodal with gap
    gap = np.zeros(N)
    gap[:N//4] = 1.0
    gap[3*N//4:] = 1.0
    starts.append(gap)
    # Ramp up then down
    ramp = np.concatenate([np.linspace(0.1, 1.0, N//2), np.linspace(1.0, 0.1, N - N//2)])
    starts.append(ramp)
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 15.0)  # 15s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()

    # Initial evaluation
    for s in starts:
        track(s)

    bounds = [(0, None)] * N

    # Phase 1: Multi-beta L-BFGS-B from diverse starts
    betas = [10.0, 30.0, 80.0, 200.0, 500.0]
    all_results = []

    for s in starts:
        if time.time() > deadline:
            break
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        for beta in betas[:3]:  # First three betas for broad exploration
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.fun < 0:
                    track(res.x)
                    all_results.append(np.maximum(0.0, res.x))
                    x0 = np.maximum(0.0, res.x)
            except Exception:
                pass

    # Phase 2: Epigraph optimization from best points
    if time.time() < deadline:
        # Use best_h and a few top results
        candidates = [best_h.copy()]
        if len(all_results) > 5:
            # Sort by value
            scored = [(compute_value(r), r) for r in all_results[:30]]
            scored.sort(reverse=True)
            candidates.extend([r for _, r in scored[:5]])

        for cand in candidates:
            if time.time() > deadline:
                break
            # Initialize t to max(g) of cand
            g_cand = np.convolve(cand, cand)
            t0 = np.max(g_cand) * 1.05
            x0 = np.concatenate([cand, [t0]])
            bounds_ext = bounds + [(1e-8, None)]
            try:
                res = opt.minimize(
                    lambda x: _epigraph_objective(x, 100.0),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_ext,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                h_res = np.maximum(0.0, res.x[:N])
                track(h_res)
            except Exception:
                pass

    # Phase 3: High-beta refinement from best
    if time.time() < deadline:
        for beta in betas[2:]:
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    lambda x: _smooth_value(x, beta),
                    best_h.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 0:
                    track(res.x)
            except Exception:
                pass

    # Phase 4: Perturbation + L-BFGS-B with very high beta
    for _ in range(15):
        if time.time() > deadline:
            break
        scale = 0.05 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, 500.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x)
        except Exception:
            pass

    # Phase 5: Nelder-Mead polish on exact objective
    def _neg_exact_value(h):
        h = np.maximum(0.0, h)
        return -compute_value(h)

    if time.time() < deadline - 5:
        for _ in range(25):
            if time.time() > deadline - 3:
                break
            scale = 0.03 * (1.0 + 0.5 * RNG.rand())
            pert = best_h + RNG.normal(0.0, scale, N)
            pert = np.maximum(0.0, pert)
            try:
                res = opt.minimize(
                    _neg_exact_value,
                    pert,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12}
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
