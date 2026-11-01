"""Packing 50 equal circles in a unit square; maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _radius(c):
    """Compute the feasible common radius for centres c (N,2) in [-0.5,0.5]^2."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    # Boundary clearance
    r_boundary = 0.5 - np.max(np.abs(c), axis=1).min() if N > 0 else 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    r = min(r_boundary, r_pair)
    return max(0.0, float(r))


def _obj_and_grad(x, beta=50.0):
    """
    Objective: maximize min-pair-distance and boundary clearance.
    We minimize -r where r is approximated via a smooth min (soft-min) over
    all pair half-distances and boundary clearances.
    """
    c = x.reshape(N, 2)
    # Boundary clearances: 0.5 - |c_ik| for each point and each coordinate
    # We want these to be >= 0, and the radius is limited by min of these and pair half-dists.
    
    # Collect all "constraints" as values that should be large:
    # For pairs (i,j): |c_i - c_j| / 2
    # For boundaries: 0.5 - |c_ix| and 0.5 - |c_iy|
    
    # Number of pairs
    pairs = []
    idx_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs, dtype=int)
    
    # Pair distances
    diff = c[pairs[:, 0]] - c[pairs[:, 1]]
    pair_dists = np.sqrt((diff ** 2).sum(axis=1)) / 2.0
    
    # Boundary clearances
    bound_clear = 0.5 - np.abs(c)  # shape (N, 2)
    bound_flat = bound_clear.flatten()
    
    # All values
    all_vals = np.concatenate([pair_dists, bound_flat])
    
    # Soft min: -log(sum(exp(-beta * vals))) / beta
    # For numerical stability, subtract max
    shifted = all_vals - all_vals.max()
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) + beta * all_vals.max()
    soft_min = -log_sum / beta
    
    # Gradient of soft_min w.r.t. each value
    # d(soft_min)/d(vals_k) = exp(-beta*vals_k) / sum(exp(-beta*vals))
    exp_neg = np.exp(-beta * shifted)
    weights = exp_neg / exp_neg.sum()  # sum to 1
    
    # Gradient w.r.t. x
    grad = np.zeros_like(x)
    
    # Pair part
    # d(pair_dists_m)/dx: for pair (i,j), d(|c_i-c_j|/2)/d c_i = (c_i - c_j)/|c_i-c_j|
    #                        d(|c_i-c_j|/2)/d c_j = (c_j - c_i)/|c_i-c_j|
    for k in range(len(pairs)):
        i, j = pairs[k]
        d = c[i] - c[j]
        dist = np.sqrt((d ** 2).sum())
        if dist < 1e-12:
            continue
        w = weights[k]
        grad[2 * i] += w * d / dist
        grad[2 * j] -= w * d / dist
    
    # Boundary part
    # bound_clear[i,0] = 0.5 - |c[i,0]|, d/d c[i,0] = -sign(c[i,0])
    # bound_clear[i,1] = 0.5 - |c[i,1]|, d/d c[i,1] = -sign(c[i,1])
    for i in range(N):
        for coord in range(2):
            idx = 2 * N + i * 2 + coord
            w = weights[idx]
            val = c[i, coord]
            if val > 0:
                grad[2 * i + coord] += w * (-1.0)
            elif val < 0:
                grad[2 * i + coord] += w * (1.0)
            else:
                grad[2 * i + coord] += 0.0
    
    # We minimize -soft_min, so objective = -soft_min, gradient = -grad
    obj = -soft_min
    return obj, -grad


def _exact_obj_and_grad(x):
    """
    Exact (non-smooth) objective: minimize -min(all_vals).
    Gradient is subgradient from the argmin set.
    """
    c = x.reshape(N, 2)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    pairs = np.array(pairs, dtype=int)
    
    diff = c[pairs[:, 0]] - c[pairs[:, 1]]
    pair_dists = np.sqrt((diff ** 2).sum(axis=1)) / 2.0
    
    bound_clear = 0.5 - np.abs(c)
    bound_flat = bound_clear.flatten()
    
    all_vals = np.concatenate([pair_dists, bound_flat])
    min_val = all_vals.min()
    
    # Subgradient: average over argmin indices
    tol = 1e-8
    argmin_mask = all_vals <= min_val + tol
    if argmin_mask.sum() == 0:
        argmin_mask[0] = True
    
    grad = np.zeros_like(x)
    # Pair part
    for k in range(len(pairs)):
        if not argmin_mask[k]:
            continue
        i, j = pairs[k]
        d = c[i] - c[j]
        dist = np.sqrt((d ** 2).sum())
        if dist < 1e-12:
            continue
        w = 1.0 / argmin_mask.sum()
        grad[2 * i] += w * d / dist
        grad[2 * j] -= w * d / dist
    
    # Boundary part
    for i in range(N):
        for coord in range(2):
            idx = 2 * N + i * 2 + coord
            if not argmin_mask[idx]:
                continue
            val = c[i, coord]
            w = 1.0 / argmin_mask.sum()
            if val > 0:
                grad[2 * i + coord] += w * (-1.0)
            elif val < 0:
                grad[2 * i + coord] += w * (1.0)
    
    obj = -min_val
    return obj, -grad


def _hexagonal_init():
    """Generate a hexagonal close-packed initial arrangement."""
    # Try to fit as many in hexagonal pattern as possible
    # For 50 circles, try various spacings
    best_x = None
    best_r = 0
    for cols in range(6, 12):
        for rows in range(6, 12):
            if cols * rows < N:
                continue
            # Hexagonal offset
            points = []
            for row in range(rows):
                for col in range(cols):
                    x = (col + (0.5 if row % 2 == 1 else 0)) / cols - 0.5
                    y = (row + 0.5) / rows - 0.5
                    # Adjust for hexagonal spacing
                    x = (col + (0.5 if row % 2 == 1 else 0)) * 1.0 / cols - 0.5
                    points.append([x, y])
            if len(points) < N:
                continue
            points = np.array(points[:N])
            r = _radius(points)
            if r > best_r:
                best_r = r
                best_x = points.copy()
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N].copy()
    return best_x


def construct():
    t_start = time.time()
    
    # Multiple initializations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Hexagonal
    inits.append(_hexagonal_init())
    
    # 3. Random perturbations of grid
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = inits[0] + rng.normal(0, 0.02, size=(N, 2))
        # Clip to square
        perturbed = np.clip(perturbed, -0.48, 0.48)
        inits.append(perturbed)
    
    # 4. Random
    for _ in range(2):
        random_pts = rng.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(random_pts)
    
    best_x = None
    best_r = 0.0
    
    for init in inits:
        if time.time() - t_start > DEADLINE - 5:
            break
        
        x0 = init.flatten()
        
        # Use L-BFGS with soft-min objective
        try:
            result = minimize(
                _obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.x is not None:
                r = _radius(result.x.reshape(N, 2))
                if r > best_r:
                    best_r = r
                    best_x = result.x.copy()
        except Exception:
            pass
        
        if time.time() - t_start > DEADLINE - 10:
            break
    
    # Polish with exact objective
    if best_x is not None and time.time() - t_start < DEADLINE - 10:
        try:
            result2 = minimize(
                _exact_obj_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result2.x is not None:
                r2 = _radius(result2.x.reshape(N, 2))
                if r2 > best_r:
                    best_r = r2
                    best_x = result2.x.copy()
        except Exception:
            pass
    
    # Final polish with soft-min again
    if best_x is not None and time.time() - t_start < DEADLINE - 5:
        try:
            result3 = minimize(
                _obj_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result3.x is not None:
                r3 = _radius(result3.x.reshape(N, 2))
                if r3 > best_r:
                    best_r = r3
                    best_x = result3.x.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N].flatten()
    
    result = best_x.reshape(N, 2)
    
    # Final validation and clipping
    result = np.clip(result, -0.5, 0.5)
    
    return result
