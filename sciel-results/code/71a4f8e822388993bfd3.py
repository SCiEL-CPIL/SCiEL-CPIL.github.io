"""
Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy:
  1. Multi-start from diverse initializations (grid, FCC, BCC, Halton, random).
  2. Use SCA (Sequential Convex Approximation) with Lagrangian relaxation.
  3. Conjugate gradient refinement for high-precision polish.
  4. Adaptive time budget: more time for search diversity, less for single refinement.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 100
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    # Wall distances: min over axes of (0.5 - |c_ik|)
    wall_dists = np.min(CUBE_HALF - np.abs(centres), axis=1)  # (N,)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pair distances
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _sca_objective_and_grad(x, r_target):
    """
    SCA objective: minimize sum of squared violations for target radius r_target.
    
    Violations:
      - Wall: max(0, r_target - (0.5 - |c_ik|))^2 for each sphere and axis
      - Pair: max(0, 2*r_target - ||c_i - c_j||)^2 for each pair within range
    
    This is a smooth (piecewise quadratic) function that we minimize.
    """
    c = x.reshape(N, 3)
    
    # Wall violations
    wall_dists = CUBE_HALF - np.abs(c)  # (N, 3)
    wall_viol = np.maximum(0.0, r_target - wall_dists)  # (N, 3)
    
    # Pair violations
    tree = cKDTree(c)
    pairs = tree.query_pairs(2 * r_target + 0.1, output_type='ndarray')
    
    grad = np.zeros_like(c)
    obj = np.sum(wall_viol ** 2)
    
    # Wall gradient
    for axis in range(3):
        viol = wall_viol[:, axis]
        mask = viol > 0
        if np.any(mask):
            grad[mask, axis] -= 2 * viol[mask] * np.sign(c[mask, axis])
    
    if len(pairs) > 0:
        diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        pair_viol = np.maximum(0.0, 2 * r_target - dists)
        
        for idx in range(len(pairs)):
            if pair_viol[idx] <= 0:
                continue
            i, j = pairs[idx]
            d = dists[idx]
            if d < 1e-12:
                continue
            factor = 2 * pair_viol[idx] / d
            grad[i] -= factor * diffs[idx]
            grad[j] += factor * diffs[idx]
            obj += pair_viol[idx] ** 2
    
    return obj, grad.ravel()


def _sca_optimize(centres, deadline, max_sca_iters=20, tol=1e-8):
    """
    SCA (Sequential Convex Approximation) optimization.
    
    Iteratively:
      1. Compute current radius r.
      2. Set target r_target slightly above r.
      3. Minimize the SCA objective (sum of squared violations) using L-BFGS-B.
      4. Update r from the result.
      5. Repeat until convergence or time limit.
    """
    from scipy.optimize import minimize
    
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    bounds = [(-CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)] * (N * 3)
    
    r = best_r
    for it in range(max_sca_iters):
        if time.time() > deadline:
            break
        
        # Target slightly above current radius
        r_target = r + max(0.001, 0.05 * r)
        
        def obj_grad(x):
            return _sca_objective_and_grad(x, r_target)
        
        try:
            res = minimize(
                obj_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            
            new_c = np.clip(res.x.reshape(N, 3), -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
            
            # Check convergence
            if abs(new_r - r) < tol:
                c = new_c
                r = new_r
                break
            
            c = new_c
            r = new_r
        except Exception:
            break
    
    return best_c, best_r


def _conjugate_gradient_refinement(centres, deadline, max_iters=200):
    """
    Conjugate gradient method for high-precision refinement.
    
    Uses the SCA gradient but with CG line search for faster convergence.
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    r = best_r
    d = np.zeros(N * 3)  # search direction
    p = np.zeros(N * 3)  # previous gradient
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update target radius
        r_target = r + max(0.0005, 0.02 * r)
        
        f, g = _sca_objective_and_grad(c.flatten(), r_target)
        
        if it == 0:
            d = -g
        else:
            # Polak-Ribiere beta
            num = np.dot(g - p, g)
            den = np.dot(p, p)
            if den > 1e-20:
                beta = max(0.0, num / den)
            else:
                beta = 0.0
            d = -g + beta * d
        
        # Line search (backtracking)
        alpha = 0.01
        f_old = f
        for ls_iter in range(30):
            x_new = c.flatten() + alpha * d
            x_new = np.clip(x_new, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
            f_new, _ = _sca_objective_and_grad(x_new, r_target)
            if f_new < f_old - 1e-4 * alpha * np.dot(g, d):
                break
            alpha *= 0.5
        
        c = np.clip(c.flatten() + alpha * d, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6).reshape(N, 3)
        p = g.copy()
        
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Convergence check
        if np.linalg.norm(g) < 1e-8:
            break
    
    return best_c, best_r


def _multi_start_search(centres, deadline, n_starts=5):
    """
    Multi-start search: perturb the current solution in different directions
    and refine each with SCA + CG.
    """
    best_c = centres.copy()
    best_r = _compute_radius(best_c)
    
    # Generate diverse perturbations
    scales = [0.01, 0.02, 0.03, 0.05, 0.08]
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        scale = scales[i % len(scales)]
        perturbed = centres + np.random.randn(N, 3) * scale
        perturbed = np.clip(perturbed, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
        
        # SCA refinement
        c1, r1 = _sca_optimize(perturbed, deadline=deadline - 3)
        if r1 > best_r:
            best_r = r1
            best_c = c1.copy()
        
        # CG refinement
        if time.time() < deadline - 2:
            c2, r2 = _conjugate_gradient_refinement(c1, deadline=deadline - 1)
            if r2 > best_r:
                best_r = r2
                best_c = c2.copy()
    
    return best_c, best_r


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc_like():
    """FCC lattice initialization."""
    a = 1.0 / 5.0
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i * a - 0.5
                y = j * a - 0.5
                z = k * a - 0.5
                # FCC: add face-center offsets
                points.append([x, y, z])
                points.append([x + a/2, y + a/2, z])
                points.append([x + a/2, y, z + a/2])
                points.append([x, y + a/2, z + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= CUBE_HALF - 0.01, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_bcc_like():
    """BCC lattice initialization."""
    a = 1.0 / 5.0
    points = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = i * a - 0.5
                y = j * a - 0.5
                z = k * a - 0.5
                points.append([x, y, z])
                points.append([x + a/2, y + a/2, z + a/2])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= CUBE_HALF - 0.01, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_halton():
    """Halton sequence initialization for good space-filling."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        points.append([x - 0.5, y - 0.5, z - 0.5])
    
    return np.array(points)


def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _refine_single(centres, deadline):
    """
    Refine a single configuration using SCA + CG.
    """
    c = centres.copy().astype(np.float64)
    
    # Phase 1: SCA
    c1, r1 = _sca_optimize(c, deadline=deadline - 5)
    
    # Phase 2: CG refinement
    if time.time() < deadline - 3:
        c2, r2 = _conjugate_gradient_refinement(c1, deadline=deadline - 1)
        if r2 > r1:
            return c2, r2
    
    return c1, r1


def _final_polish(centres, deadline):
    """
    Final high-precision polish using multiple CG iterations with decreasing step.
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Run CG with tight tolerance
    c1, r1 = _conjugate_gradient_refinement(c, deadline=deadline, max_iters=100)
    if r1 > best_r:
        best_r = r1
        best_c = c1.copy()
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    initializations = [
        _init_fcc_like,
        _init_bcc_like,
        _init_grid,
        _init_halton,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    # Phase 1: Try each initialization with SCA + CG refinement
    for init_idx, init_fn in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        init = init_fn()
        init = np.clip(init, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
        
        c1, r1 = _refine_single(init, deadline=deadline - 25.0)
        
        if r1 > best_r:
            best_r = r1
            best_centres = c1.copy()
    
    # Phase 2: Multi-start search from best solution
    if best_centres is not None and time.time() < deadline - 20.0:
        c2, r2 = _multi_start_search(best_centres, deadline=deadline - 15.0, n_starts=4)
        if r2 > best_r:
            best_r = r2
            best_centres = c2.copy()
    
    # Phase 3: Final polish
    if best_centres is not None and time.time() < deadline - 10.0:
        c3, r3 = _final_polish(best_centres, deadline=deadline - 2.0)
        if r3 > best_r:
            best_r = r3
            best_centres = c3.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
