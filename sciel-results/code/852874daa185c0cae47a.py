"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(pts):
    """Compute the actual minimum radius (exact, not soft)."""
    n = len(pts)
    # Wall distance: min over all points and all 6 walls
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    if r_wall < 0:
        return -1.0
    # Pair distances
    # Use a chunked approach to save memory
    min_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        if len(d) > 0:
            min_pair = min(min_pair, np.min(d))
    r_pair = min_pair / 2.0
    return min(r_wall, r_pair)

def _soft_min_radius(pts, beta=200.0):
    """Soft-min surrogate for the radius using log-sum-exp."""
    n = len(pts)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    pair_vals = []
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1) / 2.0
        if len(d) > 0:
            pair_vals.append(d)
    if pair_vals:
        pair_arr = np.concatenate(pair_vals)
    else:
        pair_arr = np.array([1.0])
    
    all_vals = np.concatenate([wall_dists, pair_arr])
    
    neg_beta_vals = -beta * all_vals
    max_val = np.max(neg_beta_vals)
    if max_val > 700:
        max_val = 700
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
    soft_min = -log_sum / beta
    return soft_min

def _soft_min_radius_grad(pts, beta=200.0):
    """Gradient of the soft-min radius surrogate."""
    n = len(pts)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    
    pair_vals = []
    pair_indices = []
    for i in range(n):
        d = np.linalg.norm(pts[i+1:] - pts[i], axis=1) / 2.0
        if len(d) > 0:
            pair_vals.append(d)
            for j in range(i+1, n):
                pair_indices.append((i, j))
    if pair_vals:
        pair_arr = np.concatenate(pair_vals)
    else:
        pair_arr = np.array([1.0])
    
    all_vals = np.concatenate([wall_dists, pair_arr])
    
    neg_beta_vals = -beta * all_vals
    max_val = np.max(neg_beta_vals)
    if max_val > 700:
        max_val = 700
    exp_vals = np.exp(neg_beta_vals - max_val)
    sum_exp = np.sum(exp_vals)
    
    weights = -exp_vals / sum_exp
    
    grad = np.zeros_like(pts)
    
    for i in range(n):
        w = weights[i]
        if abs(w) < 1e-15:
            continue
        abs_pts = np.abs(pts[i])
        max_idx = np.argmax(abs_pts)
        s = np.sign(pts[i, max_idx])
        if s == 0:
            s = 1.0
        grad[i, max_idx] += w * (-s)
    
    for m, (i, j) in enumerate(pair_indices):
        w = weights[n + m]
        if abs(w) < 1e-15:
            continue
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            continue
        unit = diff / dist
        grad[i] += w * unit
        grad[j] += w * (-unit)
    
    return grad

def _fcc_lattice_init():
    """Initialize with FCC-like lattice structure for better starting point."""
    m = 5
    xs = np.linspace(-0.4, 0.4, m)
    ys = np.linspace(-0.4, 0.4, m)
    zs = np.linspace(-0.45, 0.45, 4)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    rng = np.random.default_rng(42)
    grid = grid + rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.45, 0.45)
    return grid

def _md_relax(pts, time_budget=30.0):
    """Molecular dynamics relaxation with exponential repulsive potential."""
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    
    # Compute initial radius to use as reference
    r0 = _compute_radius(pts)
    if r0 <= 0:
        r0 = 0.05
    
    # MD parameters
    dt = 0.001
    k_rep = 50.0  # Repulsion strength
    damping = 0.98  # Velocity damping
    max_steps = int(time_budget / (dt * 0.001))  # Approximate steps
    max_steps = min(max_steps, 50000)
    
    velocities = np.zeros_like(pts)
    best_pts = pts.copy()
    best_r = r0
    
    for step in range(max_steps):
        if time.time() > start_time + time_budget:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Wall forces: push points away from walls
        for i in range(n):
            for k in range(3):
                dist_to_wall = 0.5 - abs(pts[i, k])
                if dist_to_wall < r0:
                    # Exponential repulsion from wall
                    force_mag = k_rep * np.exp(-dist_to_wall / (0.5 * r0))
                    if pts[i, k] > 0:
                        forces[i, k] -= force_mag
                    else:
                        forces[i, k] += force_mag
        
        # Pair forces: repulsive between all pairs
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 2.0 * r0:
                    # Exponential repulsion
                    force_mag = k_rep * np.exp(-dist / (0.5 * r0))
                    if dist > 1e-10:
                        unit = diff / dist
                        forces[i] += force_mag * unit
                        forces[j] -= force_mag * unit
                    else:
                        # Random kick for coincident points
                        forces[i] += np.random.randn(3) * 0.1
                        forces[j] -= np.random.randn(3) * 0.1
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        pts += velocities * dt
        
        # Keep points inside cube
        pts = np.clip(pts, -0.499, 0.499)
        
        # Check improvement
        if step % 100 == 0:
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        
        # Stop if forces are small
        if np.max(np.abs(forces)) < 1e-8:
            break
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial point: FCC-inspired lattice
    pts = _fcc_lattice_init()
    
    # Compute initial radius
    r0 = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = r0
    
    # Phase 1: Gradient-based optimization with soft-min
    beta_values = [100.0, 200.0, 500.0]
    
    for beta in beta_values:
        if time.time() > deadline - 40.0:
            break
        
        def objective(x):
            pts_flat = x.reshape(-1, 3)
            return -_soft_min_radius(pts_flat, beta=beta)
        
        def gradient(x):
            pts_flat = x.reshape(-1, 3)
            return _soft_min_radius_grad(pts_flat, beta=beta).reshape(-1)
        
        x0 = pts.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                objective, x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < objective(x0):
                pts = result.x.reshape(-1, 3)
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 35.0:
            break
    
    # Phase 2: MD relaxation to escape local minima
    if time.time() < deadline - 20.0:
        remaining = min(25.0, deadline - time.time() - 5.0)
        if remaining > 5.0:
            try:
                md_pts = _md_relax(best_pts, time_budget=remaining)
                r_md = _compute_radius(md_pts)
                if r_md > best_r:
                    best_r = r_md
                    best_pts = md_pts.copy()
            except Exception:
                pass
    
    # Phase 3: Final gradient polish
    if time.time() < deadline - 5.0:
        beta = 1000.0
        def objective2(x):
            pts_flat = x.reshape(-1, 3)
            return -_soft_min_radius(pts_flat, beta=beta)
        
        def gradient2(x):
            pts_flat = x.reshape(-1, 3)
            return _soft_min_radius_grad(pts_flat, beta=beta).reshape(-1)
        
        x0 = best_pts.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            result = minimize(
                objective2, x0,
                jac=gradient2,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < objective2(x0):
                pts2 = result.x.reshape(-1, 3)
                r2 = _compute_radius(pts2)
                if r2 > best_r:
                    best_r = r2
                    best_pts = pts2.copy()
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        return _fcc_lattice_init()
    if np.any(np.abs(best_pts) > 0.5):
        best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
