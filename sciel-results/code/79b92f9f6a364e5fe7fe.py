import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin for 150s evaluator limit

def _compute_radius(pts):
    """Compute the maximum radius for the given centers."""
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def _neg_radius(pts):
    return -_compute_radius(pts)

def _soft_min_radius(pts, beta):
    """
    Smooth approximation of the minimum clearance using log-sum-exp.
    The clearance is min(min_pair_dist/2, min_wall_dist).
    We approximate min over all clearances with a soft-min.
    """
    p = pts.reshape(N, 3)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    wall_dists = 0.5 - np.abs(p)
    
    # Gather all clearances
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    wall_flat = wall_dists.flatten()
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    d_min = np.min(all_dists)
    # Soft min: d_min - (1/beta) * log(sum(exp(-beta*(d_i - d_min))))
    exp_terms = np.exp(-beta * (all_dists - d_min))
    soft_min_val = d_min - np.log(np.sum(exp_terms)) / beta
    return -soft_min_val

def _soft_min_radius_grad(pts, beta):
    """
    Gradient of the soft-min radius objective (negative soft-min).
    """
    p = pts.reshape(N, 3)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    wall_dists = 0.5 - np.abs(p)
    
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    wall_flat = wall_dists.flatten()
    all_dists = np.concatenate([pair_flat, wall_flat])
    
    d_min = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_min))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of soft_min w.r.t. each distance
    # d(soft_min)/d(d_i) = -weights[i]
    # We want gradient of -soft_min, so d(-soft_min)/d(d_i) = weights[i]
    
    # Now we need to propagate back to pts
    # This is complex, so we use numerical gradient or just use L-BFGS-B without explicit grad
    # For simplicity, return None to let scipy use numerical gradients
    return None

def _repulsion_anneal(pts, temp_start=0.1, temp_end=0.01, n_steps=200):
    """
    Simulated annealing with repulsive forces.
    """
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        temp = temp_start * (temp_end / temp_start) ** (step / n_steps)
        
        # Compute forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive force: k * diff / d^3 (inverse square law in 3D for points)
        # But we want to maximize min distance, so we push apart close points
        force = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 0.3:  # Only interact with close points
                    f = 0.01 / (d ** 2 + 1e-6)
                    force[i] += f * (pts[i] - pts[j]) / d
        
        # Wall forces
        wall_dists = 0.5 - np.abs(pts)
        for i in range(N):
            for axis in range(3):
                wd = wall_dists[i, axis]
                if wd < 0.15:
                    sign = np.sign(pts[i, axis])
                    force[i, axis] += 0.01 / (wd + 1e-6) * sign
        
        # Random perturbation
        noise = np.random.normal(0, temp, size=(N, 3))
        new_pts = pts + force + noise
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = _compute_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
        
        # Accept with probability
        if new_r >= best_r or np.random.random() < np.exp((new_r - best_r) / max(temp, 1e-6)):
            pts = new_pts
    
    return best_pts

def _lloyd_step(pts):
    """
    One step of Lloyd's algorithm: assign each point to its nearest neighbor,
    then move toward the centroid of its cluster.
    """
    # This is a heuristic; for sphere packing, we can use a different approach
    # Instead, let's do a simple repulsion-based relaxation
    force = np.zeros_like(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dists[i, j]
            if d < 0.25:
                f = 0.005 / (d ** 2 + 1e-6)
                force[i] += f * (pts[i] - pts[j]) / d
    
    # Wall forces
    wall_dists = 0.5 - np.abs(pts)
    for i in range(N):
        for axis in range(3):
            wd = wall_dists[i, axis]
            if wd < 0.12:
                sign = np.sign(pts[i, axis])
                force[i, axis] += 0.005 / (wd + 1e-6) * sign
    
    new_pts = pts + force
    new_pts = np.clip(new_pts, -0.499, 0.499)
    return new_pts

def _lloyd_optimize(pts, n_steps=100):
    """
    Run Lloyd's algorithm for a fixed number of steps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        pts = _lloyd_step(pts)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts

def _local_search(pts, n_iter=50):
    """
    Local search: try small random perturbations and keep improvements.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    step_size = 0.01
    
    for _ in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        # Try random perturbation
        noise = np.random.normal(0, step_size, size=(N, 3))
        new_pts = best_pts + noise
        new_pts = np.clip(new_pts, -0.499, 0.499)
        new_r = _compute_radius(new_pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
        else:
            # Shrink step size occasionally
            if np.random.random() < 0.1:
                step_size *= 0.9
    
    return best_pts

def _init_grid():
    """5x5x2 grid."""
    nx, ny, nz = 5, 5, 2
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts += np.random.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _init_fcc():
    """FCC lattice initialization."""
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

def _init_bcc():
    """Body-centered cubic initialization."""
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
                bx = (i + 0.5) * cell - 0.5 + cell / 2
                by = (j + 0.5) * cell - 0.5 + cell / 2
                bz = (k + 0.5) * cell - 0.5 + cell / 2
                if abs(bx) <= 0.5 and abs(by) <= 0.5 and abs(bz) <= 0.5:
                    pts.append([bx, by, bz])
    
    pts = np.array(pts)
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

def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _smooth_polish(pts, beta_start=30, beta_end=200, n_phases=4):
    """
    Smooth surrogate refinement using L-BFGS-B with increasing beta.
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    for i in range(n_phases):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        beta = beta_start * (beta_end / beta_start) ** (i / n_phases)
        
        try:
            res = minimize(
                _soft_min_radius, x,
                method='L-BFGS-B',
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def _refine_exact(pts, max_iter=100):
    """
    Exact refinement using Nelder-Mead on the raw objective.
    """
    x = pts.flatten().copy()
    try:
        res = minimize(
            _neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def _final_polish(pts):
    """
    Final high-precision polish.
    """
    x = pts.flatten().copy()
    try:
        res = minimize(
            _neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(),
        _init_fcc(),
        _init_bcc(),
        _init_random(),
    ]
    
    # Add more random restarts while we have time
    while time.time() - START_TIME < TIME_LIMIT - 40:
        inits.append(_init_random())
    
    # Try each initialization
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Phase 1: Repulsion annealing for global exploration
        if time.time() - START_TIME < TIME_LIMIT - 25:
            pts1 = _repulsion_anneal(init_pts, temp_start=0.08, temp_end=0.005, n_steps=150)
        else:
            pts1 = init_pts
        
        # Phase 2: Lloyd's algorithm for local refinement
        if time.time() - START_TIME < TIME_LIMIT - 20:
            pts2 = _lloyd_optimize(pts1, n_steps=80)
        else:
            pts2 = pts1
        
        # Phase 3: Smooth surrogate polish
        if time.time() - START_TIME < TIME_LIMIT - 15:
            pts3 = _smooth_polish(pts2, beta_start=30, beta_end=150, n_phases=3)
        else:
            pts3 = pts2
        
        # Phase 4: Exact refinement
        if time.time() - START_TIME < TIME_LIMIT - 10:
            pts4 = _refine_exact(pts3, max_iter=80)
        else:
            pts4 = pts3
        
        # Phase 5: Local search
        if time.time() - START_TIME < TIME_LIMIT - 8:
            pts5 = _local_search(pts4, n_iter=30)
        else:
            pts5 = pts4
        
        refined = np.clip(pts5, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
        
        # If we have a good result, do a final high-precision polish
        if best_r > 0.135 and time.time() - START_TIME < TIME_LIMIT - 5:
            try:
                final_pts = _final_polish(best_pts)
                r2 = _compute_radius(final_pts)
                if r2 > best_r:
                    best_r = r2
                    best_pts = final_pts.copy()
            except Exception:
                pass
    
    # If no valid points found, return grid
    if best_pts is None:
        return _init_grid()
    
    return best_pts
