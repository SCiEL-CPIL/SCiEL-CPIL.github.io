"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    """
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [6, 6, 7, 7, 7, 6, 7, 4],
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
            
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 != 0) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _force_directed(pts, max_iters=20000, dt_init=0.002, dt_min=1e-7, time_limit=100.0):
    """
    Force-directed method to maximize minimum distance.
    Uses repulsive forces between all pairs and attraction to center.
    """
    t0 = time.time()
    pts = pts.copy().astype(float)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    dt = dt_init
    stuck_count = 0
    patience = 500
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: F_ij = (1/d^2) * unit_vec_ij for d < threshold
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Only apply strong repulsion when close
        threshold = 0.3  # Only interact within this distance
        mask = d < threshold
        mask.fill_diagonal(False)
        
        for i in range(N):
            for j in range(N):
                if mask[i, j] and d[i, j] > 1e-12:
                    # Repulsive force, stronger when closer
                    f = 1.0 / (d[i, j] ** 2)
                    forces[i] += f * diff[i, j] / d[i, j]
                    forces[j] -= f * diff[i, j] / d[i, j]
        
        # Wall repulsion: push points away from walls
        for i in range(N):
            # Left wall
            d_left = pts[i, 0] + 0.5
            if d_left < 0.15:
                forces[i, 0] += 1.0 / (d_left ** 2) * 0.5
            # Right wall
            d_right = 0.5 - pts[i, 0]
            if d_right < 0.15:
                forces[i, 0] -= 1.0 / (d_right ** 2) * 0.5
            # Bottom wall
            d_bottom = pts[i, 1] + 0.5
            if d_bottom < 0.15:
                forces[i, 1] += 1.0 / (d_bottom ** 2) * 0.5
            # Top wall
            d_top = 0.5 - pts[i, 1]
            if d_top < 0.15:
                forces[i, 1] -= 1.0 / (d_top ** 2) * 0.5
        
        # Normalize forces
        f_norm = np.linalg.norm(forces)
        if f_norm > 1e-12:
            forces = forces / f_norm
        
        # Update positions
        pts += dt * forces
        
        # Clamp to square (with small margin to avoid exact boundary)
        eps = 1e-9
        pts = np.clip(pts, -0.5 + eps, 0.5 - eps)
        
        # Check score
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > patience:
                dt *= 0.9
                stuck_count = 0
                if dt < dt_min:
                    break
        
        # Gradually reduce step size
        if it % 100 == 0:
            dt *= 0.98
    
    return best_pts, best_score

def _local_refine(pts, time_limit=20.0):
    """
    Local refinement using scipy minimize with BFGS on a smoothed objective.
    """
    t0 = time.time()
    
    def smooth_score(x):
        if time.time() - t0 > time_limit:
            return 0.0
        p = x.reshape(N, 2)
        # Compute minimum distance with a smooth approximation
        diff = p[:, None, :] - p[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        wall = np.minimum(
            np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0]),
            np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
        )
        
        # Use a soft-min approximation
        alpha = 100.0
        min_d = -np.logaddexp.reduce(np.log(np.exp(-alpha * d)) / alpha, axis=None)
        # Actually, let's just use the raw min for simplicity in the objective
        # but add a small penalty for being too close
        min_pair = d.min()
        wall_min = wall.min()
        
        return -min(wall_min, 0.5 * min_pair)
    
    try:
        res = minimize(smooth_score, pts.flatten(), method='Nelder-Mead',
                       options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 5000, 'adaptive': True})
        new_pts = res.x.reshape(N, 2)
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        return new_pts, _score(new_pts)
    except Exception:
        return pts, _score(pts)

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin before 150s kill
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal lattice
    hex_pts = _make_hex_initial()
    if hex_pts is not None:
        inits.append(hex_pts)
    
    # Grid-like initial
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid_pts) >= N:
        inits.append(grid_pts[:N])
    
    # Random initial with minimum distance
    rng = np.random.default_rng(42)
    for trial in range(5):
        rand_pts = rng.uniform(-0.45, 0.45, size=(N, 2))
        # Simple relaxation to avoid overlaps
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(rand_pts[i] - rand_pts[j])
                    if d < 0.15:
                        vec = rand_pts[i] - rand_pts[j]
                        if d > 1e-12:
                            vec = vec / d
                            rand_pts[i] += 0.5 * vec * (0.15 - d)
                            rand_pts[j] -= 0.5 * vec * (0.15 - d)
            rand_pts = np.clip(rand_pts, -0.45, 0.45)
        inits.append(rand_pts)
    
    # Run force-directed on each initial
    best_pts = None
    best_score = -1
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        fd_pts, fd_score = _force_directed(init, max_iters=15000, dt_init=0.002, time_limit=remaining)
        if fd_score > best_score:
            best_score = fd_score
            best_pts = fd_pts
    
    if best_pts is None:
        return inits[0] if inits else np.zeros((N, 2))
    
    # Local refinement
    if time.time() < deadline - 15:
        lr_pts, lr_score = _local_refine(best_pts, time_limit=15)
        if lr_score > best_score:
            best_score = lr_score
            best_pts = lr_pts
    
    # Final force-directed polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            fd_pts, fd_score = _force_directed(best_pts, max_iters=5000, dt_init=0.0005, time_limit=remaining)
            if fd_score > best_score:
                best_score = fd_score
                best_pts = fd_pts
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
