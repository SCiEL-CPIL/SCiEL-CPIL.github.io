"""Second autocorrelation inequality: maximize value = l2/(norm_1*norm_inf).

Strategy: scale-invariant SA over the 49-D simplex (fix h[0]=1).
"""
import time
import numpy as np

N = 50

def _eval(h):
    """Exact evaluator (mirrors the public contract)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    return l2 / (norm_1 * norm_inf)

def construct():
    t0 = time.time()
    deadline = t0 + 118.0  # safety margin under the 150s limit

    rng = np.random.default_rng(42)

    # --- seed: 50 equal heights (scale-invariant, so h[0]=1, others=1) ---
    h = np.ones(N)
    best_h = h.copy()
    best_val = _eval(h)

    # --- SA setup ---
    T0 = 1.0
    T_end = 1e-6
    # We'll run until deadline; compute total steps we can afford.
    # First measure cost of one eval.
    t1 = time.time()
    for _ in range(200):
        _eval(h)
    t2 = time.time()
    eval_time = (t2 - t1) / 200.0
    # Each SA step: 1 eval + overhead ~ 2*eval_time
    step_time = 2.0 * eval_time + 0.001
    remaining = deadline - time.time()
    max_steps = int(remaining / step_time)
    if max_steps < 1000:
        max_steps = 1000

    # Geometric cooling schedule
    if max_steps > 1:
        alpha = (T_end / T0) ** (1.0 / max_steps)
    else:
        alpha = 1.0

    T = T0
    for step in range(max_steps):
        if time.time() > deadline:
            break

        # --- propose perturbation ---
        # 90%: single-coordinate multiplicative perturbation
        # 10%: full random restart (new uniform simplex point)
        if rng.random() < 0.10:
            # Random restart: uniform on simplex (h[0]=1 fixed, others random)
            h_new = np.ones(N)
            # Generate random nonneg values for indices 1..N-1
            h_new[1:] = rng.uniform(0.0, 3.0, size=N - 1)
            # Occasionally make it more concentrated
            if rng.random() < 0.5:
                center = rng.integers(10, N - 10)
                width = rng.uniform(2.0, 15.0)
                x = np.arange(N, dtype=float)
                h_new = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
                h_new[0] = 1.0  # keep scale anchor
        else:
            h_new = h.copy()
            i = rng.integers(1, N)  # don't perturb h[0] (scale anchor)
            factor = np.exp(rng.normal(0.0, 0.3))
            h_new[i] *= factor
            if h_new[i] < 0:
                h_new[i] = 0.0

        v_new = _eval(h_new)
        dE = v_new - best_val  # we want to maximize, so "energy" = -value

        # Accept if better, or with probability exp(dE / T) (Metropolis)
        if v_new >= best_val or (T > 1e-12 and rng.random() < np.exp((v_new - best_val) / T)):
            h = h_new
            if v_new > best_val:
                best_val = v_new
                best_h = h_new.copy()

        T *= alpha

    # --- quick coordinate polish on best_h ---
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline:
                break
            improved = False
            for i in range(1, N):
                if time.time() > deadline:
                    break
                orig = best_h[i]
                for f in (1.1, 0.9, 1.2, 0.8, 1.4, 0.7, 1.6, 0.5):
                    cand = best_h.copy()
                    cand[i] = orig * f
                    v = _eval(cand)
                    if v > best_val:
                        best_val = v
                        best_h = cand
                        improved = True
                        break
            if not improved:
                break

    return list(best_h)
