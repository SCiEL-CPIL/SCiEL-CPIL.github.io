"""Packing 50 equal circles in a unit square.

Strategy:
  1. Generate diverse initial configurations (grid, hexagonal, random, and
     a hand-crafted template inspired by known packings).
  2. For each initial configuration, run a multi-stage local optimization:
     a) Fast gradient-free push-apart iterations to quickly resolve overlaps.
     b) L-BFGS-B on a smooth penalty objective (sum of squared violations
        of non-overlap and boundary constraints) to polish the configuration.
  3. Track the best radius found and return the corresponding centres.

The objective is to maximize r = min( min_{i<j} ||c_i - c_j||/2,
                                       min_i min_k (1/2 - |c_ik|) ).

We minimize a smooth surrogate:  sum over all pairs of max(0, d_ij - 2r)^2
plus boundary penalties, where d_ij is the distance between centres.
Actually, a cleaner approach: for a fixed target radius r, check feasibility.
But since we want to maximize r, we use the standard approach:
  maximize  min( min_{i<j} d_ij/2, min_i margin_i )
where margin_i = 1/2 - max(|x_i|, |y_i|).

Equivalent smooth objective to minimize:
  F = sum_{i<j} max(0, (2r - d_ij))^2  +  sum_i sum_k max(0, |c_ik| + r - 1/2)^2
But r is also a variable.  A common trick: introduce r as a variable and
minimize  -r  +  penalty terms.  However, L-BFGS-B works on smooth functions.

Alternative smooth formulation that works well in practice:
  Minimize  sum_{i<j} (1/d_ij)^2  +  sum_i (1/(1/2 - |x_i|))^2 + (1/(1/2 - |y_i|))^2
This pushes circles apart and away from boundaries.  The resulting packing's
radius is then computed directly.

We'll use this inverse-distance objective with L-BFGS-B, combined with
multi-start and a deadline guard.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # safety margin before 120 s kill


def _radius(centres):
    """Compute the common radius for a set of centres in [-1/2, 1/2]^2."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Boundary margin
    margins = np.minimum(0.5 - np.abs(centres), axis=1)
    r_boundary = margins.min()
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_boundary, r_pair)


def _objective(x, N):
    """Smooth objective: sum of inverse squared distances + boundary penalties."""
    c = x.reshape(N, 2)
    # Pairwise inverse-squared distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    # Avoid division by zero
    d = np.maximum(d, 1e-10)
    pair_term = np.sum(1.0 / (d * d))
    # Boundary penalties: push away from edges
    # For each coordinate, penalty = 1/(0.5 - |coord|)^2 if inside, large if outside
    for k in range(2):
        coords = c[:, k]
        margin = 0.5 - np.abs(coords)
        margin = np.maximum(margin, 1e-10)
        pair_term += np.sum(1.0 / (margin * margin))
    return pair_term


def _optimize_from(x0, time_budget):
    """Run L-BFGS-B from x0, return best centres found."""
    best_x = x0.copy()
    best_obj = _objective(x0, N)
    start = time.time()
    
    # Try multiple L-BFGS-B runs with different tolerances
    for tol in [1e-8, 1e-10, 1e-12]:
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(
                _objective, x0, args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': tol, 'gtol': tol, 'maxfun': 5000}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass
        if time.time() - start > time_budget:
            break
        x0 = best_x  # continue from improved point
    
    return best_x


def _grid_init(n_cols=None, n_rows=None):
    """Square grid initialisation."""
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(N)))
    if n_rows is None:
        n_rows = int(np.ceil(N / n_cols))
    ticks_x = (np.arange(n_cols) + 0.5) / n_cols - 0.5
    ticks_y = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    pts = []
    for y in ticks_y:
        for x in ticks_x:
            pts.append([x, y])
            if len(pts) == N:
                break
        if len(pts) == N:
            break
    return np.array(pts[:N])


def _hex_init():
    """Hexagonal (close-packed) initialisation."""
    # Approximate hexagonal packing: rows offset by half spacing
    # For 50 circles, use ~7 rows of 7-8 circles
    rows = []
    # Try different row counts
    for n_rows in range(5, 12):
        n_per_row = int(np.ceil(N / n_rows))
        # Spacing
        dx = 1.0 / (n_per_row + 1)
        dy = 1.0 / (n_rows + 1)
        # Adjust for hexagonal offset
        dx = 1.0 / (n_per_row + 0.5)
        dy = 1.0 / (n_rows + 1)
        pts = []
        for i in range(n_rows):
            offset = (i % 2) * dx / 2
            for j in range(n_per_row):
                x = (j + 0.5) * dx - 0.5 + offset
                y = (i + 0.5) * dy - 0.5
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
            if len(pts) >= N:
                break
        if len(pts) >= N:
            return np.array(pts[:N])
    # Fallback to grid
    return _grid_init()


def _random_init(rng):
    """Random initialisation within the square."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _push_apart(centres, iters=200, strength=0.01):
    """Simple iterative push-apart to resolve overlaps."""
    c = centres.copy()
    for _ in range(iters):
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        # For each pair that's too close, push apart
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 0.15:  # threshold
                    diff = c[i] - c[j]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-10:
                        diff = diff / norm
                    else:
                        diff = rng_uniform_vec()
                    push = strength * (0.15 - d[i, j])
                    c[i] += push * diff
                    c[j] -= push * diff
        # Clamp to square
        c = np.clip(c, -0.49, 0.49)
    return c


def rng_uniform_vec():
    return np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])


def construct():
    global DEADLINE
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_centres = None
    best_radius = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Square grid
    inits.append(_grid_init())
    
    # 2. Hexagonal
    inits.append(_hex_init())
    
    # 3. Several random inits
    for _ in range(5):
        inits.append(_random_init(rng))
    
    # 4. Grid with slight perturbation
    grid = _grid_init()
    for _ in range(3):
        perturbed = grid + rng.normal(0, 0.01, size=(N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)
    
    # Process each initial configuration
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 5:  # leave 5s safety margin
            break
        
        # Quick push-apart to get a feasible-ish starting point
        c_pushed = _push_apart(x0, iters=50, strength=0.005)
        
        # Evaluate radius after push-apart
        r_pushed = _radius(c_pushed)
        if r_pushed > best_radius:
            best_radius = r_pushed
            best_centres = c_pushed.copy()
        
        # L-BFGS-B optimization
        time_budget = max(1.0, deadline - time.time() - 5)
        if time_budget < 0.5:
            continue
        
        x_opt = _optimize_from(c_pushed.ravel(), time_budget)
        c_opt = x_opt.reshape(N, 2)
        
        # Clamp to ensure validity
        c_opt = np.clip(c_opt, -0.5, 0.5)
        
        r_opt = _radius(c_opt)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centres = c_opt.copy()
        
        # Also try from the original x0
        if time.time() < deadline - 5:
            x_opt2 = _optimize_from(x0.ravel(), max(1.0, deadline - time.time() - 5))
            c_opt2 = x_opt2.reshape(N, 2)
            c_opt2 = np.clip(c_opt2, -0.5, 0.5)
            r_opt2 = _radius(c_opt2)
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_centres = c_opt2.copy()
    
    # Final: if we have time, do one more round of fine-tuning on the best
    if best_centres is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 3
        if remaining > 1.0:
            x_final = _optimize_from(best_centres.ravel(), remaining)
            c_final = x_final.reshape(N, 2)
            c_final = np.clip(c_final, -0.5, 0.5)
            r_final = _radius(c_final)
            if r_final > best_radius:
                best_radius = r_final
                best_centres = c_final.copy()
    
    if best_centres is None:
        # Fallback: grid
        best_centres = _grid_init()
    
    return best_centres
