"""Erdos minimum overlap – piecewise-linear parameterization with global + local search.

Represent h as a piecewise-linear function with 10 control points, interpolated
to 100 samples. This low-dimensional, smooth parameterization lets the optimizer
explore global shapes (bimodal, aperiodic) more effectively than 100 independent
variables. Combine with incommensurate-frequency sinusoidal starts and a
high-precision coordinate-pair polish.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 10  # number of control points for piecewise-linear representation


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h > 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1.0 - 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _interp_control_to_samples(cp):
    """Interpolate K control points to N samples via linear interpolation."""
    cp = np.asarray(cp, dtype=float)
    # K control points define K-1 segments; we place them at positions 0..N-1
    # Control point i is at position i * (N-1) / (K-1)
    x_cp = np.linspace(0, N - 1, K)
    x_samples = np.arange(N)
    h = np.interp(x_samples, x_cp, cp)
    return h


def _smooth_max_corr(h, lam):
    """Smooth max of correlation via log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    if m > 1e15:
        return m
    shifted = corr - m
    log_sum = m + np.log(np.sum(np.exp(shifted * lam)) / len(corr)) / lam
    return log_sum


def _objective_from_control(cp):
    """Objective: build h from control points, project, return correlation value."""
    h = _interp_control_to_samples(cp)
    h = _project_to_feasible(h)
    return _correlation_value(h)


def _smooth_objective_from_control(cp, lam):
    """Smooth objective from control points."""
    h = _interp_control_to_samples(cp)
    h = _project_to_feasible(h)
    return _smooth_max_corr(h, lam)


def _golden_section_pair(h, i, j, deadline=None):
    """
    Optimize along the 1-D direction: h[i] += t, h[j] -= t, keeping sum constant.
    Find t* that minimizes _correlation_value using golden-section search.
    """
    # Range of t: h[i]+t in [0,1] => t in [-h[i], 1-h[i]]
    #               h[j]-t in [0,1] => t in [h[j]-1, h[j]]
    t_lo = max(-h[i], h[j] - 1.0)
    t_hi = min(1.0 - h[i], h[j])
    if t_lo >= t_hi:
        return h, _correlation_value(h)

    best_t = 0.0
    best_val = _correlation_value(h)

    # Golden-section search
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio ~0.618
    a, b = t_lo, t_hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _correlation_value(h + _apply_delta(h, i, j, c))
    fd = _correlation_value(h + _apply_delta(h, i, j, d))

    for _ in range(60):
        if deadline is not None and time.time() > deadline:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _correlation_value(h + _apply_delta(h, i, j, c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _correlation_value(h + _apply_delta(h, i, j, d))

    t_star = (a + b) / 2.0
    new_h = h + _apply_delta(h, i, j, t_star)
    new_val = _correlation_value(new_h)
    if new_val < best_val:
        return new_h, new_val
    return h, best_val


def _apply_delta(h, i, j, t):
    """Return a vector with +t at i and -t at j (zero elsewhere)."""
    delta = np.zeros_like(h)
    delta[i] = t
    delta[j] = -t
    return delta


def _coordinate_pair_polish(h, max_rounds=5, deadline=None, rng=None):
    """
    Repeatedly sweep over coordinate pairs and do golden-section 1-D optimization.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = N

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        # Random permutation of pairs for diversity
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        # Limit pairs per round for speed
        max_pairs = min(len(pairs), 2000)
        for i, j in pairs[:max_pairs]:
            if deadline is not None and time.time() > deadline:
                break
            new_h, new_val = _golden_section_pair(best_h, i, j, deadline)
            if new_val < best_val - 1e-14:
                best_h = new_h
                best_val = new_val
    return best_h


def _gradient_smooth_max(h, lam):
    """Gradient of smooth max correlation w.r.t. h."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    shifted = corr - m
    exp_s = np.exp(shifted * lam)
    w = exp_s / np.sum(exp_s)

    grad = np.zeros(N)
    L = N - 1
    for k in range(2 * N - 1):
        wk = w[k]
        if wk < 1e-14:
            continue
        lag = k - L
        if lag == 0:
            grad += wk * (1.0 - 2.0 * h)
        elif lag > 0:
            # d/dh_j of corr[k] where lag>0:
            # corr[k] = sum_{j=0}^{N-1-lag} h[j+lag]*(1-h[j])
            # d/dh_m: if m is in [0, N-1-lag] as j: -(h[m+lag]) + (1-h[m]) * 0... 
            # Actually: corr[k] = sum_j h[j+lag]*(1-h[j])
            # d/dh_m = (1-h[m]) if m+lag is in range and m is j-term... 
            # Let me be more careful:
            # corr[k] for lag = k-L > 0: sum_{j=0}^{N-1-lag} h[j+lag] * (1 - h[j])
            # d/dh_m:
            #   If m is in the "h[j+lag]" slot: m = j+lag => j = m-lag, valid if 0<=m-lag<=N-1-lag
            #     contribution: (1 - h[m-lag])
            #   If m is in the "(1-h[j])" slot: m = j, valid if 0<=m<=N-1-lag
            #     contribution: -h[m+lag]
            # So:
            if m_ := m if False else 0:  # placeholder
                pass
            # Simpler: build the gradient contribution
            contrib = np.zeros(N)
            # h[j+lag] slot: m = j+lag, j in [0, N-1-lag] => m in [lag, N-1]
            # contribution to h[m]: (1 - h[m-lag])
            contrib[lag:] = (1.0 - h[:N - lag])
            # (1-h[j]) slot: m = j, j in [0, N-1-lag] => m in [0, N-1-lag]
            contrib[:N - lag] -= h[lag:]
            grad += wk * contrib
        else:  # lag < 0
            l = -lag
            # corr[k] = sum_{j=l}^{N-1} h[j] * (1 - h[j-l])
            contrib = np.zeros(N)
            # h[j] slot: m = j, j in [l, N-1] => m in [l, N-1]
            contrib[l:] = (1.0 - h[:N - l])
            # (1-h[j-l]) slot: m = j-l, j in [l, N-1] => m in [0, N-1-l]
            contrib[:N - l] -= h[l:]
            grad += wk * contrib
    return grad


def _gradient_local_search(h, max_iters=300, deadline=None, rng=None):
    """Gradient-based local search on the smooth surrogate."""
    if rng is None:
        rng = np.random.default_rng(42)
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    lam = 1000.0
    lr = 0.005

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        grad = _gradient_smooth_max(best_h, lam)
        new_h = best_h - lr * grad
        new_h = _project_to_feasible(new_h)
        val = _correlation_value(new_h)
        if val < best_val - 1e-14:
            best_val = val
            best_h = new_h
            lr = min(0.05, lr * 1.05)
        else:
            lr = max(0.0005, lr * 0.5)
            if rng.random() < 0.15:
                i, j = rng.choice(N, size=2, replace=False)
                d = rng.uniform(-0.05, 0.05)
                pert = best_h.copy()
                pert[i] += d
                pert[j] -= d
                pert = _project_to_feasible(pert)
                vp = _correlation_value(pert)
                if vp < best_val:
                    best_val = vp
                    best_h = pert
    return best_h


def _make_sinusoid_start(freqs, amps, phases, offset=0.5):
    """Build a sinusoidal h pattern."""
    x = np.arange(N) / N
    h = np.full(N, offset)
    for f, a, p in zip(freqs, amps, phases):
        h += a * np.sin(2 * np.pi * f * x + p)
    return _project_to_feasible(h)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # === Phase 1: Differential evolution on control-point parameterization ===
    # 10 control points, each in [0, 1]
    bounds_cp = [(0.0, 1.0)] * K

    def de_obj(cp):
        if time.time() > deadline - 100.0:
            raise TimeoutError()
        return _objective_from_control(cp)

    de_deadline = start_time + 35.0
    try:
        result = differential_evolution(
            de_obj,
            bounds=bounds_cp,
            maxiter=80,
            popsize=12,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='sobol'
        )
        if result.x is not None:
            h_de = _project_to_feasible(_interp_control_to_samples(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    # === Phase 2: Multi-start SLSQP on control points ===
    rng = np.random.default_rng(123)
    starts_cp = [best_h if best_h is not None else np.full(K, 0.5)]

    # Inverse-interpolate best_h to get control points
    x_cp = np.linspace(0, N - 1, K)
    cp_best = np.interp(x_cp, np.arange(N), best_h)
    starts_cp.append(cp_best)

    # Random control points
    for _ in range(10):
        cp_rand = rng.uniform(0, 1, K)
        starts_cp.append(cp_rand)

    # Structured: bimodal
    for gap in [0.15, 0.2, 0.25, 0.3]:
        cp_bi = np.zeros(K)
        for i in range(K):
            t = i / (K - 1)
            if abs(t - 0.5) < gap:
                cp_bi[i] = 0.1
            else:
                cp_bi[i] = 0.9
        starts_cp.append(cp_bi)

    # Incommensurate frequency sinusoids mapped to control points
    golden = (1 + np.sqrt(5)) / 2
    for base_f in [1.0, golden, 2.0, 3.0, golden ** 2, 5.0]:
        x_cp = np.linspace(0, N - 1, K)
        cp_sin = 0.5 + 0.35 * np.sin(2 * np.pi * base_f * x_cp / (N - 1))
        starts_cp.append(cp_sin)

    # Multi-frequency
    for f1, f2 in [(1.0, golden), (2.0, 3.0), (1.0, 3.0), (golden, 2.0 * golden)]:
        x_cp = np.linspace(0, N - 1, K)
        cp_mf = 0.5 + 0.2 * np.sin(2 * np.pi * f1 * x_cp / (N - 1)) + 0.2 * np.sin(2 * np.pi * f2 * x_cp / (N - 1))
        starts_cp.append(cp_mf)

    # Sawtooth / triangular at control point level
    for period in [4, 5, 6, 8]:
        cp_tri = np.zeros(K)
        for i in range(K):
            t = (i % period) / period
            cp_tri[i] = 1.0 - 2.0 * abs(t - 0.5)
        starts_cp.append(cp_tri)

    slsqp_deadline = start_time + 120.0

    for idx, cp_start in enumerate(starts_cp):
        if time.time() > slsqp_deadline - 10.0:
            break
        cp_start = np.clip(cp_start, 0.0, 1.0)

        # Smooth SLSQP on control points
        for lam in [100.0, 500.0, 2000.0]:
            if time.time() > slsqp_deadline - 5.0:
                break

            def smooth_cp_obj(cp, lam=lam):
                h = _interp_control_to_samples(cp)
                h = _project_to_feasible(h)
                return _smooth_max_corr(h, lam)

            try:
                res = minimize(
                    smooth_cp_obj,
                    cp_start,
                    method='SLSQP',
                    bounds=bounds_cp,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                cp_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _project_to_feasible(_interp_control_to_samples(cp_opt))
                val = _correlation_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    cp_start = cp_opt.copy()
            except Exception:
                pass

        # Exact SLSQP on control points
        if time.time() > slsqp_deadline - 3.0:
            break

        try:
            def exact_cp_obj(cp):
                h = _interp_control_to_samples(cp)
                h = _project_to_feasible(h)
                return np.max(np.correlate(h, 1 - h, mode="full"))

            res_exact = minimize(
                exact_cp_obj,
                cp_start,
                method='SLSQP',
                bounds=bounds_cp,
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            cp_ex = np.clip(res_exact.x, 0.0, 1.0)
            h_ex = _project_to_feasible(_interp_control_to_samples(cp_ex))
            val_ex = _correlation_value(h_ex)
            if val_ex < best_val:
                best_val = val_ex
                best_h = h_ex.copy()
        except Exception:
            pass

    # === Phase 3: Full 100-dim SLSQP refinement from best control-point solution ===
    if time.time() < slsqp_deadline - 2.0:
        try:
            def exact_full_obj(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))

            res_full = minimize(
                exact_full_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 400, 'ftol': 1e-15}
            )
            h_full = _project_to_feasible(res_full.x)
            val_full = _correlation_value(h_full)
            if val_full < best_val:
                best_val = val_full
                best_h = h_full.copy()
        except Exception:
            pass

    # === Phase 4: Coordinate-pair golden-section polish ===
    if time.time() < deadline - 15.0:
        try:
            polish_dl = deadline - 2.0
            best_h = _coordinate_pair_polish(best_h, max_rounds=4, deadline=polish_dl)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # === Phase 5: Gradient local search ===
    if time.time() < deadline - 5.0:
        try:
            best_h = _gradient_local_search(best_h, max_iters=300, deadline=deadline - 1.0)
            best_val = _correlation_value(best_h)
        except Exception:
            pass

    # === Phase 6: Final SLSQP polish ===
    if time.time() < deadline - 1.0:
        try:
            def final_obj(x):
                return np.max(np.correlate(x, 1 - x, mode="full"))

            res_final = minimize(
                final_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
                options={'maxiter': 500, 'ftol': 1e-16}
            )
            h_final = _project_to_feasible(res_final.x)
            val_final = _correlation_value(h_final)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Final projection
    best_h = _project_to_feasible(best_h)
    return best_h
