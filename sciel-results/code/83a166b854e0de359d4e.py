"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _smooth_score(c_flat, beta=50.0):
    """Smooth surrogate for the radius score using log-sum-exp.
    
    Instead of min(distances), we use a soft-min that is differentiable.
    This allows gradient-based optimizers to navigate the landscape smoothly.
    """
    c = c_flat.reshape(N, 2)
    # Clip to avoid boundary issues in gradients, though bounds handle this
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances
    d = pdist(c)
    if d.size == 0:
        return 0.0
    
    # Wall distances for each point
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    # Soft-min of pairwise distances (each pair contributes distance/2 to radius)
    # Use log-sum-exp trick for numerical stability
    d_half = d / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    # Soft-min of wall distances
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    # Combine: we want min(soft_min_pair, soft_min_wall)
    # Use another soft-min
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat, beta=50.0):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat, beta)


def _penalty_function(c_flat, r, beta=50.0):
    """Penalty for infeasibility at radius r.
    
    Returns a smooth penalty that is 0 when all constraints are satisfied
    (pairwise distances >= 2r, wall distances >= r), and positive otherwise.
    """
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-12, HALF - 1e-12)
    
    # Pairwise distances / 2 (half-distance, must be >= r)
    d = pdist(c)
    d_half = d / 2.0
    # Violation: max(0, r - d_half) for each pair
    # Use soft version: smooth max(0, r - d_half)
    # For pairs where d_half < r, penalty = (r - d_half)
    # Smooth approximation using log-sum-exp on violations
    violations = np.maximum(0.0, r - d_half)
    if violations.max() > 0:
        # Soft sum of violations
        v_max = violations.max()
        exp_v = np.exp(-beta * (violations - v_max))
        soft_sum_pair = v_max + np.log(exp_v.sum()) / beta
    else:
        soft_sum_pair = 0.0
    
    # Wall distances (must be >= r)
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    wall_violations = np.maximum(0.0, r - wall_dists)
    if wall_violations.max() > 0:
        w_max = wall_violations.max()
        exp_w = np.exp(-beta * (wall_violations - w_max))
        soft_sum_wall = w_max + np.log(exp_w.sum()) / beta
    else:
        soft_sum_wall = 0.0
    
    return soft_sum_pair + soft_sum_wall


def _bisect_radius(c0, r_lo=None, r_hi=None, n_iter=60, beta=100.0, deadline=None):
    """Bisect on radius to find max feasible r for given centres.
    
    Uses penalty function to check feasibility.
    """
    if r_lo is None:
        r_lo = _score_from_centres(c0)
    if r_hi is None:
        r_hi = r_lo + 0.01
    
    c = c0.copy()
    best_c = c.copy()
    best_r = r_lo
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to optimize centres to make radius r_mid feasible
        try:
            res = minimize(
                lambda cf, r=r_mid: _penalty_function(cf, r, beta=beta),
                c.ravel(),
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-12, HALF - 1e-12)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'iprint': 0}
            )
            c_new = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        except Exception:
            c_new = c.copy()
        
        # Check actual feasibility
        penalty = _penalty_function(c_new.ravel(), r_mid, beta=1000.0)
        actual_r = _score_from_centres(c_new)
        
        if penalty < 1e-10:
            # Feasible: r_mid is achievable
            r_lo = r_mid
            best_c = c_new.copy()
            best_r = actual_r
            c = c_new.copy()
        else:
            # Not feasible: reduce r
            r_hi = r_mid
            # Use the optimized positions as starting point for next iteration
            c = c_new.copy()
        
        # Also track best actual score
        s_actual = _score_from_centres(c_new)
        if s_actual > best_r:
            best_r = s_actual
            best_c = c_new.copy()
    
    return best_c, best_r


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    # Hexagonal close packing for better initial configuration
    # 8 cols, 7 rows approx
    rows = int(np.ceil(N / 8))
    cols = 8
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2) * 0.22
            y = (i - (rows - 1) / 2) * 0.2
            if i % 2 == 1:
                x += 0.11
            points.append([x, y])
    return np.array(points[:N])


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    
    # Try multiple initial configurations
    inits = [_grid_init(), _hex_init()]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)

    # Strategy 1: Local optimisation from multiple starts with smooth objective
    # Start with the better initial config
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    # Use adaptive beta: start low for smooth exploration, increase for refinement
    n_starts = 25
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        elif i == 2:
            x0 = _grid_init().ravel()
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)

        try:
            # Multi-stage optimization:
            # Stage 1: Low beta (smooth) to find basin
            # Stage 2: Medium beta to refine
            # Stage 3: High beta to lock in
            betas = [10.0, 30.0, 100.0]
            current_x = x0
            
            for beta in betas:
                if time.time() - t0 > DEADLINE - 35:
                    break
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), current_x,
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                current_x = np.clip(res.x, -HALF + 1e-9, HALF - 1e-9)
                c = current_x.reshape(N, 2)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 2: Simulated Annealing with targeted local moves
    if time.time() - t0 < DEADLINE - 25:
        def _sa_optimize(c0, t_max=0.01, t_min=1e-5, n_iter=5000, rng_sa=None):
            """Simulated annealing with local refinement."""
            if rng_sa is None:
                rng_sa = np.random.default_rng()
            
            c = c0.copy()
            current_score = _score_from_centres(c)
            best_c = c.copy()
            best_s = current_score
            
            for it in range(n_iter):
                if time.time() - t0 > DEADLINE - 20:
                    break
                
                # Temperature schedule
                T = t_max * (t_min / t_max) ** (it / n_iter)
                
                # Generate move: perturb a random subset of points
                n_move = rng_sa.integers(1, 6)
                idx = rng_sa.choice(N, size=n_move, replace=False)
                
                c_new = c.copy()
                # Scale perturbation with temperature
                scale = 0.02 + 0.08 * (T / t_max)
                delta = rng_sa.normal(0, scale, size=(n_move, 2))
                c_new[idx] += delta
                c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
                
                new_score = _score_from_centres(c_new)
                delta_s = new_score - current_score
                
                # Accept if better or with probability exp(-delta_s/T) for worse
                if delta_s > 0 or (T > 0 and rng_sa.random() < np.exp(max(-100, delta_s / T))):
                    c = c_new
                    current_score = new_score
                    if current_score > best_s:
                        best_s = current_score
                        best_c = c.copy()
                
                # Every 50 iterations, do a local refinement on best with high beta
                if it % 50 == 0 and it > 0:
                    try:
                        # Use high beta for refinement to sharpen the minimum
                        res = minimize(
                            lambda cf: _negative_score(cf, 100.0), best_c.ravel(),
                            method='L-BFGS-B',
                            bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                            options={'maxiter': 200, 'ftol': 1e-13, 'iprint': 0}
                        )
                        c_refined = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                        s_refined = _score_from_centres(c_refined)
                        if s_refined > best_s:
                            best_s = s_refined
                            best_c = c_refined
                            c = best_c.copy()
                            current_score = best_s
                    except Exception:
                        pass
            
            return best_c, best_s
        
        sa_c, sa_s = _sa_optimize(best_centres, t_max=0.015, t_min=1e-6, n_iter=6000, rng_sa=rng)
        if sa_s > best_score:
            best_score = sa_s
            best_centres = sa_c

    # Strategy 3: Bisection on radius with penalty-based feasibility
    if time.time() - t0 < DEADLINE - 10:
        try:
            r_lo = best_score
            r_hi = best_score + 0.005
            best_centres, best_r = _bisect_radius(
                best_centres, r_lo=r_lo, r_hi=r_hi,
                n_iter=40, beta=100.0,
                deadline=DEADLINE - 5
            )
            if best_r > best_score:
                best_score = best_r
                best_centres = best_centres
        except Exception:
            pass

    # Strategy 4: Final high-precision polish with very high beta smooth surrogate
    if time.time() - t0 < DEADLINE - 3:
        try:
            for beta in [200.0, 500.0, 1000.0]:
                if time.time() - t0 > DEADLINE - 1:
                    break
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), best_centres.ravel(),
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-12, HALF - 1e-12)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'iprint': 0}
                )
                c = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 5: One more bisection pass for final precision
    if time.time() - t0 < DEADLINE - 1:
        try:
            r_lo = best_score
            r_hi = best_score + 0.001
            best_centres, best_r = _bisect_radius(
                best_centres, r_lo=r_lo, r_hi=r_hi,
                n_iter=30, beta=500.0,
                deadline=DEADLINE - 0.5
            )
            if best_r > best_score:
                best_score = best_r
                best_centres = best_centres
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    return best_centres
