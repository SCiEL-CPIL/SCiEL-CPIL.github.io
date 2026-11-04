import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all non-zero shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1; center (zero shift) is at index N-1
    c1 = c[:N-1]      # negative shifts
    c2 = c[N:]        # positive shifts
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2 using iterative proportional fitting."""
    h = np.clip(h, 0.0, 1.0)
    target = TARGET_SUM
    for _ in range(100):
        current = np.sum(h)
        diff = target - current
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            room = 1.0 - h
            room[h >= 1.0 - 1e-15] = 0.0
            total_room = room.sum()
            if total_room < 1e-15:
                break
            h += room * (diff / total_room)
            h = np.clip(h, 0.0, 1.0)
        else:
            avail = h.copy()
            avail[h <= 1e-15] = 0.0
            total_avail = avail.sum()
            if total_avail < 1e-15:
                break
            h -= avail * (-diff / total_avail)
            h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            room = 1.0 - h
            room[h >= 1.0 - 1e-14] = 0.0
            tr = room.sum()
            if tr < 1e-14:
                break
            h += room * (diff / tr)
            h = np.clip(h, 0.0, 1.0)
        else:
            avail = h.copy()
            avail[h <= 1e-14] = 0.0
            ta = avail.sum()
            if ta < 1e-14:
                break
            h -= avail * (-diff / ta)
            h = np.clip(h, 0.0, 1.0)
    return h

def _objective_lse(x, beta, lam_sum):
    """LogSumExp approximation of max overlap with sum penalty."""
    h = np.clip(x, 0.0, 1.0)
    overlaps = _compute_overlaps(h)
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500.0
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse + sum_penalty

def _objective_epigraph(x, lam_sum):
    """Epigraph: minimize t + penalties, x = [h, t]."""
    h = np.clip(x[:N], 0.0, 1.0)
    t = x[N]
    overlaps = _compute_overlaps(h)
    violations = np.maximum(overlaps - t, 0.0)
    constraint_penalty = np.sum(violations ** 2) + 0.1 * np.sum(violations)
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + constraint_penalty + sum_penalty

def _spectral_init(rng, n_modes=10):
    """
    Generate initial h via spectral factorization approach.
    Model h as 0.5 + sum of sinusoidal modes with optimized amplitudes/phases.
    The Wiener-Khinchin theorem relates overlap to power spectrum;
    we aim for flat power spectrum at non-zero frequencies.
    """
    t = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, n_modes + 1):
        amp = rng.uniform(0.05, 0.45)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(2 * np.pi * k * t / N + phase)
    return _project_to_feasible(h)

def _sturmian_init(alpha, n_terms=N):
    """
    Generate a Sturmian (mechanical) word mapped to [0,1] heights.
    Sturmian words have low discrepancy and uniform distribution properties.
    """
    # Binary Sturmian word
    s = np.zeros(n_terms, dtype=int)
    for i in range(n_terms):
        s[i] = 1 if np.floor((i + 1) * alpha) > np.floor(i * alpha) else 0
    # Map to heights: use local averaging to get smooth values
    h = np.zeros(N)
    for i in range(N):
        window = s[max(0, i-2):min(n_terms, i+3)]
        h[i] = 0.3 + 0.4 * np.mean(window)
    return _project_to_feasible(h)

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit

    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_epigraph = bounds_h + [(-1.0, 2.0)]

    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)

    rng = np.random.default_rng(12345)

    # ---- Phase 1: Diverse initial candidates ----
    candidates = []

    # Uniform
    candidates.append(np.full(N, 0.5))

    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)

    # Sinusoidal
    t = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())

    # Cosine
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            trial = 0.5 + amp * np.cos(2*np.pi*freq*t/N)
            candidates.append(trial.copy())

    # Sawtooth
    candidates.append(np.linspace(0.1, 0.9, N).copy())

    # Triangular
    h_tri = 0.5 + 0.4 * np.abs(2 * (t / N - 0.5) - 1)
    candidates.append(h_tri.copy())

    # Step functions
    for split in range(10, 90, 10):
        h_step = np.zeros(N)
        h_step[:split] = 0.6
        h_step[split:] = 0.4
        candidates.append(h_step.copy())

    # Spectral init (Wiener-Khinchin inspired)
    for _ in range(20):
        candidates.append(_spectral_init(rng, n_modes=rng.integers(3, 12)))

    # Sturmian word inits (low discrepancy)
    for alpha in [0.5 + 0.1/np.sqrt(5), 0.5 + 0.2/np.sqrt(5),
                  0.5 + 0.3/np.sqrt(5), (1 + np.sqrt(5))/4 - 0.5,
                  0.5 + 0.15/np.sqrt(2), 0.5 + 0.25/np.sqrt(3)]:
        candidates.append(_sturmian_init(alpha))

    # Random
    for _ in range(30):
        candidates.append(rng.uniform(0.05, 0.95, size=N))

    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # ---- Phase 2: LSE optimization with beta schedule ----
    beta_schedule = [3, 5, 10, 15, 20, 30, 50, 75, 100]
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        x0 = best_h.copy()

        def obj_func(x):
            return _objective_lse(x, beta, 500.0)

        try:
            result = minimize(
                obj_func, x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # ---- Phase 3: Epigraph optimization ----
    if time.time() < deadline - 25.0:
        t_init = best_score
        x0_ep = np.concatenate([best_h, [t_init]])

        def obj_ep(x):
            return _objective_epigraph(x, 1000.0)

        try:
            result = minimize(
                obj_ep, x0_ep,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # ---- Phase 4: Random restarts with LSE ----
    for i in range(15):
        if time.time() > deadline - 30.0:
            break
        x0 = rng.uniform(0.1, 0.9, size=N)
        x0 = _project_to_feasible(x0)
        beta = 15.0 + i * 5.0

        def obj_rand(x):
            return _objective_lse(x, beta, 500.0)

        try:
            result = minimize(
                obj_rand, x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 250, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # ---- Phase 5: High-precision polish ----
    if time.time() < deadline - 15.0:
        def obj_polish(x):
            return _objective_lse(x, 300.0, 5000.0)

        try:
            result = minimize(
                obj_polish, best_h,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass

    # ---- Phase 6: Epigraph polish ----
    if time.time() < deadline - 10.0:
        t_init = best_score
        x0_ep = np.concatenate([best_h, [t_init]])

        def obj_ep_polish(x):
            return _objective_epigraph(x, 5000.0)

        try:
            result = minimize(
                obj_ep_polish, x0_ep,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass

    # ---- Phase 7: Local search with small perturbations ----
    if time.time() < deadline - 3.0:
        for _ in range(80):
            if time.time() > deadline - 1.0:
                break
            sigma = 0.005 + 0.01 * rng.random()
            perturbation = rng.normal(0, sigma, size=N)
            h_perturbed = best_h + perturbation
            h_perturbed = _project_to_feasible(h_perturbed)
            score_perturbed = _max_overlap(h_perturbed)
            if score_perturbed < best_score:
                best_score = score_perturbed
                best_h = h_perturbed

    return best_h
