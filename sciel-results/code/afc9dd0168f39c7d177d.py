import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import time

N = 100
TARGET_SUM = N / 2.0
TOL = 1e-9 * N  # 1e-7


def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0


def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(300):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max_objective(h, beta=80.0):
    """Smooth approximation of max correlation for gradient-based optimization."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    c_max = np.max(corr)
    if c_max == 0:
        return 0.0
    scaled = beta * (corr - c_max)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -100, 0)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return (c_max + log_sum / beta)


def _construct_diverse_seeds(rng):
    """Generate a set of diverse starting points."""
    seeds = []
    x = np.arange(N)

    # Uniform
    seeds.append(np.full(N, 0.5))

    # Alternating low-high
    seeds.append(np.where(x % 2 == 0, 0.2, 0.8).copy())

    # Sine waves of various frequencies
    for freq in range(1, 11):
        for amp in [0.3, 0.45, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * x / N)
            seeds.append(_project_feasible(h))
            h = 0.5 + amp * np.cos(2 * np.pi * freq * x / N)
            seeds.append(_project_feasible(h))

    # Block patterns
    for block_size in [10, 20, 25, 33, 50]:
        h = np.zeros(N)
        n_ones = N // (2 * block_size) * block_size  # ensure sum ~ N/2
        # Place ones in evenly spaced blocks
        positions = np.arange(0, N, block_size * 2)
        for pos in positions:
            end = min(pos + block_size, N)
            h[pos:end] = 1.0
        h = _project_feasible(h)
        seeds.append(h)

    # Random smooth (low-pass filtered)
    for _ in range(15):
        noise = rng.randn(N)
        # Apply simple low-pass
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        smooth = np.convolve(noise, kernel, mode='same')
        h = 0.5 + 0.4 * smooth
        seeds.append(_project_feasible(h))

    # Random binary-like (many 0s and 1s)
    for _ in range(10):
        h = rng.choice([0.0, 1.0], size=N, p=[0.5, 0.5])
        seeds.append(_project_feasible(h))

    # Sawtooth
    for period in [10, 20, 25, 50]:
        sawtooth = np.linspace(0, 1, period, endpoint=False)
        h = np.tile(sawtooth, N // period + 1)[:N]
        seeds.append(_project_feasible(h))

    # Piecewise constant
    for n_segments in [4, 5, 10, 20]:
        seg_len = N // n_segments
        vals = rng.uniform(0.2, 0.8, n_segments)
        h = np.repeat(vals, seg_len + (1 if seg_len * n_segments < N else 0))[:N]
        seeds.append(_project_feasible(h))

    return seeds


def _polish_with_slsqp(h0, deadline):
    """Polish a candidate with SLSQP using soft-max objective."""
    bounds = [(0.0, 1.0)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM,
         'jac': lambda x: np.ones(N)}
    ]

    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 5:
            break
        obj = lambda x: _soft_max_objective(x, beta=beta)
        try:
            res = minimize(
                obj, h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            return h_res
        except Exception:
            pass
    return _project_feasible(h0)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    rng = np.random.RandomState(42)

    # Phase 1: Generate diverse seeds and polish with SLSQP
    seeds = _construct_diverse_seeds(rng)

    for h0 in seeds:
        if time.time() > deadline - 15:
            break
        h_polished = _polish_with_slsqp(h0, deadline)
        val = _correlation_max(h_polished)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h_polished.copy()

    # Phase 2: Basin hopping around the best solution
    if time.time() < deadline - 10:
        bounds = [(0.0, 1.0)] * N

        def bh_objective(h):
            return _soft_max_objective(h, beta=100.0)

        def bh_step(x):
            # Random perturbation with smoothing
            step = rng.randn(N) * 0.1
            # Low-pass the step
            step = np.convolve(step, np.ones(3)/3.0, mode='same')
            y = x + step
            y = _project_feasible(y)
            return y

        try:
            bh_res = basinhopping(
                bh_objective,
                best_h,
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds},
                step=bh_step,
                niter=30,
                T=0.1,
                seed=42
            )
            h_bh = _project_feasible(bh_res.x)
            val = _correlation_max(h_bh)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_bh.copy()
        except Exception:
            pass

    # Phase 3: More targeted SLSQP polishing with higher beta
    if time.time() < deadline - 5:
        h_final = _polish_with_slsqp(best_h, deadline)
        val = _correlation_max(h_final)
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_final.copy()

    # Phase 4: Try differential evolution on a subset (if time allows)
    if time.time() < deadline - 8:
        try:
            bounds = [(0.0, 1.0)] * N

            def de_objective(h):
                return _soft_max_objective(h, beta=60.0)

            de_res = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=200, popsize=15,
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='sobol'
            )
            h_de = _project_feasible(de_res.x)
            # Polish with SLSQP
            h_de = _polish_with_slsqp(h_de, deadline)
            val = _correlation_max(h_de)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_de.copy()
        except Exception:
            pass

    # Final: ensure validity
    best_h = _project_feasible(best_h)
    # Double-check sum
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute the correction
        mask = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
        n_free = np.sum(mask)
        if n_free > 0:
            best_h[mask] += diff / n_free
        best_h = np.clip(best_h, 0.0, 1.0)
        best_h = _project_feasible(best_h)

    return best_h
