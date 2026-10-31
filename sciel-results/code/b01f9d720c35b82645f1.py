import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    # Clip to avoid overflow
    shifted = np.clip(shifted, 0, 100)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(x_flat, temperature):
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(x_flat, temperature):
    s = _soft_score(x_flat, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Fourier-based parameterization ----------

def _fourier_to_points(params):
    """
    Convert Fourier parameters to 13 points on a smooth convex curve.
    params: [r0, a1, b1, a2, b2] (5 parameters)
    r0: base radius
    a1, b1: first harmonic amplitudes
    a2, b2: second harmonic amplitudes
    """
    r0 = params[0]
    a1, b1 = params[1], params[2]
    a2, b2 = params[3], params[4]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = r0 + a1 * np.cos(angles) + b1 * np.sin(angles) + \
            a2 * np.cos(2 * angles) + b2 * np.sin(2 * angles)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _neg_score_fourier(params):
    points = _fourier_to_points(params)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_fourier(params, temperature):
    points = _fourier_to_points(params)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.min(areas) < 1e-15:
        return 1e10
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 1e10
    return -(soft_min / hull_area)


# ---------- CMA-ES implementation (lightweight) ----------

def _cmaes_optimize(objective, x0, sigma0, max_iter=500, deadline=None, pop_size=None):
    """
    Simple CMA-ES implementation for the Fourier parameter space.
    """
    n = len(x0)
    if pop_size is None:
        pop_size = max(4 + int(3 * np.log(n)), 8)
    
    x = np.array(x0, dtype=float)
    sigma = sigma0
    best_x = x.copy()
    best_f = objective(x)
    
    # Covariance matrix
    C = np.eye(n)
    # Strategy parameters
    alpha_mu = 0.5
    mu = pop_size // 2
    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c_c = (4 - mu_eff / n) / (n + 4 - mu_eff / n)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    step_size = sigma
    
    for it in range(max_iter):
        if deadline is not None and time.time() >= deadline - 5:
            break
        
        # Generate population
        pops = []
        for i in range(pop_size):
            z = np.random.randn(n)
            # Approximate covariance update (simplified)
            y = z * sigma
            pops.append(x + y)
        
        # Evaluate
        fitnesses = [objective(p) for p in pops]
        indices = np.argsort(fitnesses)
        
        # Update best
        if fitnesses[indices[0]] < best_f:
            best_f = fitnesses[indices[0]]
            best_x = pops[indices[0]].copy()
        
        # Weighted mean of best mu solutions
        x_new = np.zeros(n)
        for i in range(mu):
            x_new += weights[i] * pops[indices[i]]
        
        # Update step size
        y_mean = (x_new - x) / sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
        
        # Update covariance
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
        rank_one = np.outer(y_mean, y_mean)
        rank_mu = np.zeros((n, n))
        for i in range(mu):
            y_i = (pops[indices[i]] - x) / sigma
            rank_mu += weights[i] * np.outer(y_i, y_i)
        
        C = (1 - c_c * (c_c + 2 - c_sigma)) * C + \
            (c_c * (c_c + 2 - c_sigma) + c_sigma * (2 - c_sigma) * mu_eff) * rank_one + \
            c_sigma * (2 - c_sigma) * (1 / mu_eff - 1) * rank_mu
        
        x = x_new
        
        # Ensure convergence
        if sigma < 1e-10:
            break
    
    return best_x, best_f


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular():
    rng = np.random.RandomState(42)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.05 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_random_ring(rng):
    n_outer = int(rng.integers(5, 10))
    n_inner = N - n_outer
    if n_inner > 0:
        o_angles = np.sort(rng.uniform(0, 2 * np.pi, n_outer))
        i_angles = np.sort(rng.uniform(0, 2 * np.pi, n_inner))
        r_in = rng.uniform(0.3, 0.7)
        o = np.column_stack([np.cos(o_angles), np.sin(o_angles)])
        i = np.column_stack([r_in * np.cos(i_angles), r_in * np.sin(i_angles)])
        return np.vstack([o, i])
    else:
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        return np.column_stack([np.cos(angles), np.sin(angles)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(x0, deadline, max_temp=0.05, min_temp=1e-4, n_steps=12):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for temp in temps:
        if time.time() >= deadline - 8:
            break

        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                res = minimize(
                    _neg_soft_score,
                    best,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x
                s = _score(cand.reshape(N, 2))
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

    return best, best_s


def _polish_true_objective(x0, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = x0.copy()
    best_s = _score(best.reshape(N, 2))

    # Powell
    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score,
                best,
                method='Nelder-Mead',
                options={'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x
            s = _score(cand.reshape(N, 2))
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass

    return best, best_s


def _local_search(points, deadline, n_iter=300, sigma_start=0.02, sigma_end=0.001):
    """
    Coordinate-wise local search with adaptive step size.
    Tries moving each point in random directions, keeping improvements.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)

    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break

        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress

        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction

            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break

    return best, best_s


def _optimize_fourier(deadline):
    """
    Optimize in the 5-dimensional Fourier parameter space using CMA-ES.
    Returns the best points found.
    """
    # Initial guess: regular polygon (r0=1, all harmonics 0)
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    sigma0 = 0.1
    
    best_points = _fourier_to_points(x0)
    best_s = _score(best_points)
    
    # Multiple CMA-ES runs with different initializations
    rng = np.random.default_rng(123)
    
    for run in range(5):
        if time.time() >= deadline - 10:
            break
        
        # Perturb initial point
        if run == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(0, 0.1, 5)
            x_init[0] = np.clip(x_init[0], 0.5, 1.5)  # Keep radius reasonable
        
        try:
            x_opt, f_opt = _cmaes_optimize(
                _neg_score_fourier,
                x_init,
                sigma0,
                max_iter=300,
                deadline=deadline
            )
            points = _fourier_to_points(x_opt)
            s = _score(points)
            if s > best_s:
                best_s = s
                best_points = points.copy()
        except Exception:
            pass
    
    return best_points, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin against 150s kill

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Generate diverse seeds
    rng = np.random.default_rng(42)
    seeds = [
        _seed_regular(),
        _seed_two_ring(7, 0.55, np.pi / 6),
        _seed_two_ring(6, 0.5, np.pi / 5),
        _seed_two_ring(8, 0.4, np.pi / 4),
        _seed_three_ring(5, 4, 4, 0.65, 0.35, np.pi / 5, np.pi / 4),
        _seed_three_ring(6, 4, 3, 0.6, 0.3, np.pi / 3, np.pi / 6),
        _seed_quasi_regular(),
    ]

    # Add randomized ring-based seeds
    for _ in range(8):
        seeds.append(_seed_random_ring(rng))

    # Evaluate all seeds
    for seed in seeds:
        _update_best(seed)

    # NEW: Fourier-based CMA-ES optimization
    if time.time() < deadline - 30:
        try:
            fourier_points, fourier_s = _optimize_fourier(deadline)
            _update_best(fourier_points)
        except Exception:
            pass

    # Annealed optimization from each seed
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 25:
            break
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                seed.flatten(),
                deadline,
                max_temp=0.05,
                min_temp=1e-4,
                n_steps=10,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # Polish the best found so far
    if best_points is not None and time.time() < deadline - 20:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    # Local search from best
    if best_points is not None and time.time() < deadline - 15:
        try:
            ls_points, s_ls = _local_search(best_points, deadline, n_iter=200)
            _update_best(ls_points)
        except Exception:
            pass

    # Final hard-objective polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            x_final, s_final = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_final.reshape(N, 2))
        except Exception:
            pass

    # More annealed restarts if time permits
    while time.time() < deadline - 10:
        seed_idx = int(rng.integers(0, len(seeds)))
        base_seed = seeds[seed_idx].copy()
        noise = rng.normal(0, 0.05, (N, 2))
        perturbed = base_seed + noise
        try:
            x_opt, s_opt = _annealed_soft_minimize(
                perturbed.flatten(),
                deadline,
                max_temp=0.03,
                min_temp=1e-4,
                n_steps=8,
            )
            _update_best(x_opt.reshape(N, 2))
        except Exception:
            pass

    # One more polish if we improved
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_pol, s_pol = _polish_true_objective(best_points.flatten(), deadline)
            _update_best(x_pol.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
