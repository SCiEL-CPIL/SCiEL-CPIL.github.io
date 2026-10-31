"""Optimized 16-point placement: minimize (max_dist/min_dist)^2.
Strategy: hexagonal lattice starts + differential evolution + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 16
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]


def _exact_score(P):
    """Compute (max_d / min_d)^2 = max_d2 / min_d2."""
    d2 = _pairwise_dists_sq(P)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)


def _soft_objective(params, beta=100.0):
    """
    Soft min/max surrogate in reduced space.
    params: (28,) flattened 14 points (p0=(0,0), p1=(1,0) fixed).
    Returns soft max_d2 / soft min_d2.
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d2 = _pairwise_dists_sq(P)
    d2 = np.maximum(d2, 1e-12)
    
    # Soft max via log-sum-exp
    max_d2 = np.max(d2)
    soft_max = max_d2 + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d2 - max_d2))))
    
    # Soft min
    min_d2 = np.min(d2)
    soft_min = min_d2 - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d2 - min_d2))))
    
    if soft_min < 1e-8:
        return 1e10
    
    return soft_max / soft_min


def _exact_objective(params):
    """Exact objective in reduced parameter space."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return _exact_score(P)


def _prepare_params(P):
    """
    Fix closest pair at (0,0) and (1,0), return params for remaining 14 points.
    """
    d2 = _pairwise_dists_sq(P)
    min_idx = np.argmin(d2)
    iu = np.triu_indices(N, 1)
    i_min, j_min = iu[0][min_idx], iu[1][min_idx]
    
    p0 = P[i_min]
    p1 = P[j_min]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return None
    
    # Rotation to align v with x-axis
    cos_t = v[0] / norm_v
    sin_t = v[1] / norm_v
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    
    # Transform: translate so p0 is origin, rotate, scale so p1 is at (1,0)
    transformed = (P - p0) @ R.T / norm_v
    
    # Reorder: i_min -> 0, j_min -> 1, rest in original order
    perm = list(range(N))
    perm[0] = i_min
    perm[1] = j_min
    remaining = [i for i in range(N) if i != i_min and i != j_min]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    params = transformed[2:].flatten()
    return params


def _reconstruct(params):
    """Reconstruct points from params (p0=(0,0), p1=(1,0), rest=params)."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return P


def _hex_lattice_points(radius):
    """Generate hexagonal lattice points within radius of origin.
    Hex lattice with min distance 1: basis vectors at 60 degrees."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    points = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            p = i * v1 + j * v2
            if np.sum(p ** 2) <= (radius + 0.5) ** 2:
                points.append(p)
    return np.array(points)


def _get_hex_starts():
    """Get top hexagonal lattice configurations as starting points.
    Use N closest points to origin, plus some variations."""
    hex_pts = _hex_lattice_points(4)
    n_hex = len(hex_pts)
    
    # Sort by distance to origin
    dists = np.sum(hex_pts ** 2, axis=1)
    sorted_idx = np.argsort(dists)
    
    starts = []
    
    # 1. N closest points
    P1 = hex_pts[sorted_idx[:N]]
    starts.append(P1)
    
    # 2. Try swapping one of the last few points with a nearby alternative
    base_set = set(sorted_idx[:N])
    for swap_out in sorted_idx[N-3:N]:
        for swap_in in sorted_idx[N:N+5]:
            if swap_in not in base_set:
                new_set = list(base_set - {swap_out} | {swap_in})
                P = hex_pts[np.array(new_set)]
                starts.append(P)
    
    # 3. Also try a slightly larger radius subset
    hex_pts2 = _hex_lattice_points(3)
    dists2 = np.sum(hex_pts2 ** 2, axis=1)
    sorted_idx2 = np.argsort(dists2)
    if len(hex_pts2) >= N:
        starts.append(hex_pts2[sorted_idx2[:N]])
    
    # Deduplicate and limit
    unique_starts = []
    seen = set()
    for P in starts:
        # Use sorted tuple of rounded coords as key
        key = tuple(sorted([tuple(np.round(p, 6)) for p in P]))
        if key not in seen:
            seen.add(key)
            unique_starts.append(P)
        if len(unique_starts) >= 8:
            break
    
    return unique_starts


def _local_optimize(params, maxiter_nm=500, maxiter_lbfgs=500):
    """Two-phase local optimization: Nelder-Mead then L-BFGS-B."""
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Nelder-Mead
    try:
        res_nm = minimize(
            _exact_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        if res_nm.fun < best_score:
            best_score = res_nm.fun
            best_params = res_nm.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B on soft objective
    try:
        res_lbfgs = minimize(
            lambda p: _soft_objective(p, beta=100.0),
            best_params,
            method='L-BFGS-B',
            options={'maxiter': maxiter_lbfgs, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        exact_s = _exact_objective(res_lbfgs.x)
        if exact_s < best_score:
            best_score = exact_s
            best_params = res_lbfgs.x.copy()
    except Exception:
        pass
    
    return best_params, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = float('inf')
    
    # Phase 1: Hexagonal lattice starts with local optimization
    hex_starts = _get_hex_starts()
    for P0 in hex_starts:
        if time.time() > DEADLINE - 40:
            break
        try:
            params = _prepare_params(P0)
            if params is None:
                continue
            p_opt, score = _local_optimize(params, maxiter_nm=300, maxiter_lbfgs=300)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            pass
    
    # Phase 2: Differential Evolution in reduced space
    if time.time() < DEADLINE - 50:
        try:
            # Bounds: points should be within reasonable range
            bounds = [(-5, 5)] * (2 * (N - 2))
            
            # Initial population: hex starts + random
            init_pop = []
            for P0 in hex_starts:
                params = _prepare_params(P0)
                if params is not None:
                    init_pop.append(params)
            # Add random points to fill population
            while len(init_pop) < 20:
                init_pop.append(rng.uniform(-3, 3, (N - 2, 2)).flatten())
            
            init_pop = np.array(init_pop[:20])
            
            de_res = differential_evolution(
                _soft_objective,
                bounds=bounds,
                args=(100.0,),
                maxiter=30,
                popsize=20,
                tol=1e-8,
                mutation=(0.3, 0.9),
                recombination=0.8,
                seed=42,
                init=init_pop,
                polish=True,
                workers=1
            )
            
            if de_res.fun < best_score:
                # Verify with exact score
                exact_s = _exact_objective(de_res.x)
                if exact_s < best_score:
                    best_score = exact_s
                    best_params = de_res.x.copy()
            else:
                # Still try local optimization from DE result
                p_opt, score = _local_optimize(de_res.x, maxiter_nm=200, maxiter_lbfgs=200)
                if score < best_score:
                    best_score = score
                    best_params = p_opt
        except Exception:
            pass
    
    # Phase 3: Random restarts with local optimization
    while time.time() < DEADLINE - 15:
        try:
            # Random restart near best solution
            if best_params is not None:
                params = best_params + rng.normal(0, 0.1, (N - 2, 2)).flatten()
            else:
                params = rng.uniform(-2, 2, (N - 2, 2)).flatten()
            
            p_opt, score = _local_optimize(params, maxiter_nm=150, maxiter_lbfgs=150)
            if score < best_score:
                best_score = score
                best_params = p_opt
        except Exception:
            break
    
    # Phase 4: Final high-precision polish
    if best_params is not None and time.time() < DEADLINE - 5:
        try:
            res_final = minimize(
                lambda p: _soft_objective(p, beta=500.0),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            exact_s = _exact_objective(res_final.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Reconstruct final points
    if best_params is None:
        # Fallback: hexagonal lattice subset
        hex_starts = _get_hex_starts()
        if hex_starts:
            return hex_starts[0]
        else:
            # Simple fallback grid
            import itertools
            grid = list(itertools.product(range(4), repeat=2))[:N]
            return np.array(grid, dtype=float)
    
    return _reconstruct(best_params)


if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
