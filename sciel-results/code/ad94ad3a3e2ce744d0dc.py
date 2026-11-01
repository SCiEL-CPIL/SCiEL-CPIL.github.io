import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 105.0  # Safety margin for 120s evaluator limit

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def electrostatic_energy(x, k=5.0, wall_k=100.0):
    """
    Smooth electrostatic (jellium) energy:
    - Soft Riesz repulsion between all pairs (squared inverse distance, softened)
    - Harmonic confinement to cube walls
    This is fully differentiable and avoids the nonsmooth min-distance objective.
    """
    pts = x.reshape(N, 3)
    
    # Soft repulsion: sum over pairs of 1/(d^2 + eps)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    eps = 1e-4
    repulsion = np.sum(1.0 / (dist_sq + eps))
    
    # Harmonic wall confinement: penalize proximity to walls
    # Use (0.5 - |coord|) as distance to nearest wall, square it for smoothness
    wall_dists = 0.5 - np.abs(pts)
    # Penalize small wall distances: 1/(d + eps)^2
    wall_energy = np.sum(1.0 / (wall_dists + eps) ** 2)
    
    return k * repulsion + wall_k * wall_energy

def electrostatic_grad(x, k=5.0, wall_k=100.0):
    """
    Gradient of the electrostatic energy.
    """
    pts = x.reshape(N, 3)
    eps = 1e-4
    
    # Repulsion gradient: for each pair i,j, force on i is -2*(pi-pj)/(d^2+eps)^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # shape (N,N,3)
    dist_sq = np.sum(diff * diff, axis=2)  # shape (N,N)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Gradient contribution from repulsion
    # d/dpi [1/(|pi-pj|^2 + eps)] = -2*(pi-pj) / (|pi-pj|^2 + eps)^2
    denom = (dist_sq + eps) ** 2
    denom[np.diag_indices(N)] = np.inf
    rep_grad = -2.0 * diff / denom[:, :, np.newaxis]
    rep_grad_sum = np.sum(rep_grad, axis=1)  # shape (N,3)
    
    # Wall gradient: d/dpi [1/(0.5-|pi|+eps)^2] = -2*(sign(pi))/(0.5-|pi|+eps)^3
    wall_dists = 0.5 - np.abs(pts)
    signs = np.sign(pts)
    wall_grad = -2.0 * signs / (wall_dists + eps) ** 3
    
    grad = k * rep_grad_sum + wall_k * wall_grad
    return grad.flatten()

def electrostatic_smooth_optimize(x0, k=5.0, wall_k=100.0, max_iter=200):
    """
    Optimize using L-BFGS-B on the smooth electrostatic energy.
    Returns optimized positions.
    """
    x = x0.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    try:
        res = minimize(
            electrostatic_energy, x,
            jac=electrostatic_grad,
            method='L-BFGS-B',
            bounds=bounds,
            args=(k, wall_k),
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
        )
        x = res.x
    except Exception:
        pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def get_552_init():
    """
    Generate a 5x5x2 grid initialization (exactly 50 points).
    """
    nx, ny, nz = 5, 5, 2
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small jitter to break symmetry
    pts += np.random.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def get_body_centered_init():
    """
    Generate a body-centered cubic like initialization.
    """
    pts = []
    grid_size = 4
    cell = 1.0 / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x0 = (i + 0.5) * cell - 0.5
                y0 = (j + 0.5) * cell - 0.5
                z0 = (k + 0.5) * cell - 0.5
                pts.append([x0, y0, z0])
                # Body center
                bx = (i + 0.5) * cell - 0.5 + cell / 2
                by = (j + 0.5) * cell - 0.5 + cell / 2
                bz = (k + 0.5) * cell - 0.5 + cell / 2
                if abs(bx) <= 0.5 and abs(by) <= 0.5 and abs(bz) <= 0.5:
                    pts.append([bx, by, bz])
    
    pts = np.array(pts)
    # Select N points with maximum minimum distance (greedy)
    if len(pts) > N:
        selected = [pts[0]]
        remaining = list(pts[1:])
        for _ in range(N - 1):
            if not remaining:
                break
            best_idx = 0
            best_dist = -1
            for idx, pt in enumerate(remaining):
                min_d = min(np.linalg.norm(pt - s) for s in selected)
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        pts = np.array(selected)
    
    if len(pts) < N:
        n_pad = N - len(pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
        
    return pts[:N]

def get_body_centered_init():
    """
    Generate a body-centered cubic like initialization.
    """
    # 4x4x4 grid with body centers
    pts = []
    grid_size = 4
    cell = 1.0 / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x0 = (i + 0.5) * cell - 0.5
                y0 = (j + 0.5) * cell - 0.5
                z0 = (k + 0.5) * cell - 0.5
                pts.append([x0, y0, z0])
                # Body center
                bx = (i + 0.5) * cell - 0.5 + cell / 2
                by = (j + 0.5) * cell - 0.5 + cell / 2
                bz = (k + 0.5) * cell - 0.5 + cell / 2
                if abs(bx) <= 0.5 and abs(by) <= 0.5 and abs(bz) <= 0.5:
                    pts.append([bx, by, bz])
    
    pts = np.array(pts)
    # Select N points with maximum minimum distance (greedy)
    if len(pts) > N:
        selected = [pts[0]]
        remaining = list(pts[1:])
        for _ in range(N - 1):
            if not remaining:
                break
            best_idx = 0
            best_dist = -1
            for idx, pt in enumerate(remaining):
                min_d = min(np.linalg.norm(pt - s) for s in selected)
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        pts = np.array(selected)
    
    if len(pts) < N:
        n_pad = N - len(pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
        
    return pts[:N]

def get_fcc_init():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cell_size, j * cell_size, k * cell_size
                fcc_pts.append([x0, y0, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0 + 0.5 * cell_size, z0])
                fcc_pts.append([x0 + 0.5 * cell_size, y0, z0 + 0.5 * cell_size])
                fcc_pts.append([x0, y0 + 0.5 * cell_size, z0 + 0.5 * cell_size])
    
    fcc_pts = np.array(fcc_pts)
    fcc_pts -= 0.5
    
    mask = np.all(np.abs(fcc_pts) <= 0.5, axis=1)
    fcc_pts = fcc_pts[mask]
    
    if len(fcc_pts) > N:
        selected = []
        remaining = fcc_pts.copy()
        for _ in range(N):
            if not remaining.size:
                break
            pt = remaining[0]
            selected.append(pt)
            d = np.linalg.norm(remaining - pt, axis=1)
            remaining = remaining[d > 0.12]
        fcc_pts = np.array(selected)
        
    if len(fcc_pts) < N:
        n_pad = N - len(fcc_pts)
        pad = np.random.uniform(-0.45, 0.45, size=(n_pad, 3))
        fcc_pts = np.vstack([fcc_pts, pad])
        
    return fcc_pts[:N]

def get_jellium_init():
    """
    Generate initial positions using a short electrostatic relaxation.
    Start from random positions and run a few L-BFGS-B steps.
    """
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    # Quick electrostatic relaxation
    pts = electrostatic_smooth_optimize(pts, k=3.0, wall_k=50.0, max_iter=50)
    return pts

def local_refine(pts):
    """
    Local refinement using Nelder-Mead on the exact objective.
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Nelder-Mead on the exact objective
    try:
        res = minimize(
            neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    
    # Phase 2: Another round with tighter tolerances
    try:
        res = minimize(
            neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def electrostatic_polish(pts, k_values=None, wall_k_values=None):
    """
    Multi-scale electrostatic polishing: run L-BFGS-B with varying stiffness
    to gradually tighten the packing.
    """
    if k_values is None:
        k_values = [2.0, 5.0, 10.0, 20.0]
    if wall_k_values is None:
        wall_k_values = [50.0, 100.0, 200.0, 400.0]
    
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    for k, wk in zip(k_values, wall_k_values):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        try:
            res = minimize(
                electrostatic_energy, x,
                jac=electrostatic_grad,
                method='L-BFGS-B',
                bounds=bounds,
                args=(k, wk),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def local_refine_smooth(pts, beta_start=30.0, beta_end=150.0):
    """
    Smooth surrogate refinement followed by exact refinement.
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Smooth phase with increasing beta
    for beta in [beta_start, beta_start * 2, beta_start * 4, beta_end]:
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        try:
            def obj(x):
                p = x.reshape(N, 3)
                if np.any(np.abs(p) > 0.5 + 1e-9):
                    return 1e6
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                pair_dists = dists / 2.0
                wall_dists = 0.5 - np.abs(p)
                wall_flat = wall_dists.flatten()
                pair_flat = pair_dists[np.triu_indices(N, k=1)]
                all_dists = np.concatenate([pair_flat, wall_flat])
                d_min = np.min(all_dists)
                exp_terms = np.exp(-beta * (all_dists - d_min))
                soft_min_val = d_min - np.log(np.sum(exp_terms)) / beta
                return -soft_min_val
            
            res = minimize(
                obj, x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    # Exact phase
    x_2d = x.reshape(N, 3)
    if time.time() - START_TIME > TIME_LIMIT - 10.0:
        return x_2d
    try:
        res = minimize(
            neg_radius, x_2d,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        x_2d = res.x
    except Exception:
        pass
    
    return x_2d

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Generate diverse initializations
    inits = [
        get_552_init(),
        get_body_centered_init(),
        get_fcc_init(),
        get_jellium_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    # Add more random restarts while we have time
    while time.time() - START_TIME < TIME_LIMIT - 30.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Try each initialization
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Use electrostatic polish for most, fall back to exact refinement
        if i % 3 == 0:
            # Full electrostatic polish pipeline
            refined = electrostatic_polish(init_pts)
            # Then exact Nelder-Mead polish
            if time.time() - START_TIME < TIME_LIMIT - 8.0:
                refined = local_refine(refined)
        elif i % 3 == 1:
            # Smooth surrogate then exact
            refined = local_refine_smooth(init_pts)
        else:
            # Direct exact refinement
            refined = local_refine(init_pts)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
        
        # If we have a good result, do a final high-precision polish
        if best_r > 0.135 and time.time() - START_TIME < TIME_LIMIT - 5.0:
            try:
                res = minimize(
                    neg_radius, best_pts,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                r2 = compute_radius(res.x)
                if r2 > best_r:
                    best_r = r2
                    best_pts = res.x.copy()
            except Exception:
                pass
    
    # If no valid points found, return grid
    if best_pts is None:
        nx, ny, nz = 5, 5, 2
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        ys = (np.arange(ny) + 0.5) / ny - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid[:N]
    
    return best_pts
