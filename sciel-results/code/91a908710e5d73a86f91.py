"""Pack 50 equal circles in a unit square; maximize common radius."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 115.0  # safety margin under 120s wall-clock limit


def _radius(c):
    """Compute the packing radius for centres c of shape (N,2)."""
    c = np.asarray(c)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    min_wall = np.min(np.minimum(0.5 - np.abs(c), 0.5))
    return min(min_pair, min_wall)


def _negative_radius(c):
    """Negative radius for minimization."""
    return -_radius(c)


def _gradient_radius(c):
    """
    Compute gradient of the radius function.
    The radius is min over all constraints of (distance/2 or wall distance).
    We use the constraint that is currently most active (smallest).
    """
    c = np.asarray(c).copy()
    n = len(c)
    
    # Compute all pair distances and wall distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances for each point
    wall_dists = np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0])  # x-direction
    wall_dists = np.minimum(wall_dists, np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
    
    # Find the most constraining pair or wall
    min_pair_val = np.min(dists) / 2.0
    min_wall_val = np.min(wall_dists)
    
    grad = np.zeros_like(c)
    
    if min_pair_val <= min_wall_val:
        # The active constraint is a pair
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx
        if i == j:
            return grad
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist < 1e-15:
            return grad
        # d/dx_i (dist/2) = d / (2*dist)
        # d/dx_j (dist/2) = -d / (2*dist)
        grad[i] = d / (2.0 * dist)
        grad[j] = -d / (2.0 * dist)
    else:
        # The active constraint is a wall
        idx = np.argmin(wall_dists)
        pt = c[idx]
        # Determine which wall is closest
        dx_pos = 0.5 - pt[0]
        dx_neg = 0.5 + pt[0]
        dy_pos = 0.5 - pt[1]
        dy_neg = 0.5 + pt[1]
        min_d = min(dx_pos, dx_neg, dy_pos, dy_neg)
        if min_d == dx_pos:
            grad[idx, 0] = -1.0
        elif min_d == dx_neg:
            grad[idx, 0] = 1.0
        elif min_d == dy_pos:
            grad[idx, 1] = -1.0
        else:
            grad[idx, 1] = 1.0
    
    return grad


def _soft_radius(c, beta=50.0):
    """
    Smooth approximation of the radius using soft-min.
    This gives a differentiable objective.
    """
    c = np.asarray(c)
    n = len(c)
    
    # Pair constraints: distance/2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    # Wall constraints
    wall_vals = np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0])
    wall_vals = np.minimum(wall_vals, np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
    
    # Combine all constraints into one vector
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    
    # Soft min: -log(sum(exp(-beta*x)))/beta
    # For numerical stability
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min


def _soft_radius_grad(c, beta=50.0):
    """Gradient of the soft radius."""
    c = np.asarray(c)
    n = len(c)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0])
    wall_vals = np.minimum(wall_vals, np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    
    min_val = np.min(all_vals)
    shifted = all_vals - min_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # weights for soft min
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    pair_weights = weights[:n * n].reshape(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = pair_weights[i, j]
            if w < 1e-12:
                continue
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            # d(pair_val[i,j])/d(c[i]) = d/(2*dist)
            # d(pair_val[i,j])/d(c[j]) = -d/(2*dist)
            grad[i] += w * d / (2.0 * dist)
            grad[j] -= w * d / (2.0 * dist)
    
    # Wall contributions
    wall_weights = weights[n * n:]
    for i in range(n):
        w = wall_weights[i]
        if w < 1e-12:
            continue
        pt = c[i]
        dx_pos = 0.5 - pt[0]
        dx_neg = 0.5 + pt[0]
        dy_pos = 0.5 - pt[1]
        dy_neg = 0.5 + pt[1]
        min_d = min(dx_pos, dx_neg, dy_pos, dy_neg)
        if min_d == dx_pos:
            grad[i, 0] -= w
        elif min_d == dx_neg:
            grad[i, 0] += w
        elif min_d == dy_pos:
            grad[i, 1] -= w
        else:
            grad[i, 1] += w
    
    return -grad  # negative because we minimize negative soft radius


def construct():
    start_time = time.time()
    best_c = None
    best_r = -np.inf
    
    def _check_and_update(c):
        nonlocal best_c, best_r
        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        return r
    
    # Initial guess: grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
    _check_and_update(grid)
    
    # Try multiple random initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate random initial positions
        if attempt == 0:
            c = grid.copy()
        else:
            # Random jitter around grid or fully random
            if attempt < 10:
                c = grid + rng.randn(N, 2) * 0.02
            else:
                c = rng.uniform(-0.45, 0.45, size=(N, 2))
        
        # Clip to stay inside
        c = np.clip(c, -0.49, 0.49)
        
        # Run L-BFGS-B on the soft radius
        beta = 30.0
        res = minimize(
            _soft_radius,
            c,
            jac=_soft_radius_grad,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if res.success or res.fun < 0:
            _check_and_update(res.x)
        
        # Also try a second pass with higher beta for sharper min
        if time.time() - start_time > DEADLINE - 10:
            break
        beta2 = 100.0
        res2 = minimize(
            _soft_radius,
            res.x,
            jac=_soft_radius_grad,
            args=(beta2,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() - start_time > DEADLINE - 5:
            break
        
        _check_and_update(res2.x)
    
    # Try differential evolution for global search if time permits
    if time.time() - start_time < DEADLINE - 30:
        bounds = [(-0.49, 0.49)] * (2 * N)
        
        def _de_obj(x):
            c = x.reshape(N, 2)
            return -_radius(c)
        
        de_res = differential_evolution(
            _de_obj,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='deferred'
        )
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if de_res.success or de_res.fun < 0:
            c_de = de_res.x.reshape(N, 2)
            _check_and_update(c_de)
            
            # Polish with L-BFGS-B
            if time.time() - start_time < DEADLINE - 10:
                res3 = minimize(
                    _soft_radius,
                    c_de,
                    jac=_soft_radius_grad,
                    args=(50.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                _check_and_update(res3.x)
    
    # Final polish on best solution
    if best_c is not None and time.time() - start_time < DEADLINE - 5:
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > DEADLINE - 5:
                break
            res_final = minimize(
                _soft_radius,
                best_c,
                jac=_soft_radius_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            _check_and_update(res_final.x)
    
    # Ensure output is valid
    if best_c is None:
        return grid[:N]
    
    # Clip to ensure inside square
    best_c = np.clip(best_c, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_c
