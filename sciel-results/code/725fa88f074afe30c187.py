"""Packing 50 equal circles in unit disk - gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _compute_score(points):
    """Return the feasible common radius for a set of centres."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    d = np.linalg.norm(points, axis=1)
    margin = 1.0 - d
    if np.min(margin) < -1e-12:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, np.min(margin))
    return max(r, 0.0)


def _energy(points, r, beta=50.0):
    """
    Smooth energy: sum of violations.
    For each pair i<j: max(0, 2r - dist_ij)^2
    For each i: max(0, r + dist_i - 1)^2
    """
    pts = points
    n = len(pts)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Pair violations
    pair_viol = np.maximum(0, 2 * r - dists)
    pair_energy = np.sum(pair_viol ** 2)
    
    # Wall violations
    radii = np.linalg.norm(pts, axis=1)
    wall_viol = np.maximum(0, r + radii - 1.0)
    wall_energy = np.sum(wall_viol ** 2)
    
    return pair_energy + wall_energy


def _neg_objective(params, r, beta=50.0):
    """Negative of the score approximation via energy minimization."""
    pts = params.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    e = _energy(pts, r, beta)
    # We want to maximize r such that energy is small
    # Use: -r + penalty * energy
    penalty = 100.0
    return -r + penalty * e


def _refine_local(points, max_iters=2000, tol=1e-12):
    """Local refinement using gradient-based approach on full 2D positions."""
    pts = points.copy().astype(float)
    
    best_r = _compute_score(pts)
    
    # Try to increase r and fix violations
    for attempt in range(50):
        # Current radius
        d = np.linalg.norm(pts, axis=1)
        margin = 1.0 - d
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_margin = np.min(margin)
        r = min(min_pair, min_margin)
        
        if r <= best_r + 1e-14:
            break
        
        best_r = r
        
        # Try to push r up slightly and minimize energy
        r_trial = r * 1.001
        if r_trial > 0.13:
            r_trial = 0.13
        
        def obj(x):
            p = x.reshape(N, 2)
            return _energy(p, r_trial, beta=100.0)
        
        try:
            res = minimize(obj, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 1e-10:
                pts = res.x.reshape(N, 2)
                new_r = _compute_score(pts)
                if new_r > best_r:
                    best_r = new_r
                    continue
        except Exception:
            pass
        break
    
    # Final polish: iterative constraint projection
    for _ in range(max_iters):
        d = np.linalg.norm(pts, axis=1)
        margin = 1.0 - d
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_margin = np.min(margin)
        r = min(min_pair / 2.0, min_margin)
        
        if min_pair / 2.0 < min_margin - 1e-12:
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i == j:
                continue
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-14:
                continue
            di /= dist
            push = (2 * r - dist) / 2.0
            if push > 1e-14:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(margin))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-14:
                continue
            di = pi / dist
            push = (1.0 - r - dist)
            if push > 1e-14:
                pts[i] += push * di
        
        if r >= best_r - 1e-12:
            if r > best_r:
                best_r = r
            break
    
    return pts


def _ring_initialization(ring_counts, ring_radii):
    """Generate initial points from ring structure."""
    pts = []
    for count, radius in zip(ring_counts, ring_radii):
        if count == 1:
            pts.append(np.array([0.0, 0.0]))
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            pts.append(np.column_stack([xs, ys]))
    return np.vstack(pts)


def _hex_initialization():
    """Hexagonal lattice initialization."""
    # Start with a hex grid and select best 50 points
    spacing = 0.22
    points = []
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < 1.0:
                points.append([x, y])
    points = np.array(points)
    
    # Select 50 points that maximize minimum distance
    if len(points) < N:
        # Increase spacing
        spacing = 0.2
        points = []
        for j in range(-5, 6):
            for i in range(-5, 6):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x*x + y*y < 1.0:
                    points.append([x, y])
        points = np.array(points)
    
    if len(points) >= N:
        # Greedy selection
        selected = [0]  # Start with center-ish point
        for _ in range(N - 1):
            best_idx = -1
            best_min_dist = -1
            for idx in range(len(points)):
                if idx in selected:
                    continue
                min_d = min(np.linalg.norm(points[idx] - points[sel]) for sel in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        return points[selected]
    
    return np.zeros((N, 2))


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_MARGIN
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Ring-based initialization with optimization
    ring_counts = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.22, 0.44, 0.66, 0.88]
    
    for trial in range(8):
        if time.time() > deadline:
            break
        
        # Vary ring radii slightly
        rng = np.random.RandomState(trial * 123 + 45)
        radii = np.array(ring_radii) * (0.9 + 0.2 * rng.random(len(ring_radii)))
        radii[0] = 0.0  # Keep center
        
        pts = _ring_initialization(ring_counts, radii)
        
        # Optimize using L-BFGS-B on full 2D positions
        def objective(x):
            p = x.reshape(N, 2)
            # Compute current r
            d = np.linalg.norm(p, axis=1)
            margin = 1.0 - d
            diff = p[:, None, :] - p[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            min_margin = np.min(margin)
            r = min(min_pair, min_margin)
            
            # Energy-based penalty
            e = _energy(p, r, beta=50.0)
            return -r + 10.0 * e
        
        try:
            res = minimize(objective, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            pts_opt = res.x.reshape(N, 2)
            s = _compute_score(pts_opt)
            
            if s > best_score:
                best_score = s
                best_points = pts_opt.copy()
                
                # Local refinement
                refined = _refine_local(pts_opt, max_iters=1000)
                s_ref = _compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_points = refined.copy()
        except Exception:
            pass
    
    # Strategy 2: Hexagonal initialization
    if time.time() < deadline - 20:
        pts_hex = _hex_initialization()
        
        def objective2(x):
            p = x.reshape(N, 2)
            d = np.linalg.norm(p, axis=1)
            margin = 1.0 - d
            diff = p[:, None, :] - p[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            min_margin = np.min(margin)
            r = min(min_pair, min_margin)
            e = _energy(p, r, beta=50.0)
            return -r + 10.0 * e
        
        try:
            res = minimize(objective2, pts_hex.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            pts_opt = res.x.reshape(N, 2)
            s = _compute_score(pts_opt)
            
            if s > best_score:
                best_score = s
                best_points = pts_opt.copy()
                
                refined = _refine_local(pts_opt, max_iters=1000)
                s_ref = _compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_points = refined.copy()
        except Exception:
            pass
    
    # Strategy 3: Random restarts with optimization
    if time.time() < deadline - 30:
        for trial in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(trial * 777 + 13)
            # Random points in disk
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0, 0.95, N)
            pts_rand = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            def objective3(x):
                p = x.reshape(N, 2)
                d = np.linalg.norm(p, axis=1)
                margin = 1.0 - d
                diff = p[:, None, :] - p[None, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2.0
                min_margin = np.min(margin)
                r = min(min_pair, min_margin)
                e = _energy(p, r, beta=50.0)
                return -r + 10.0 * e
            
            try:
                res = minimize(objective3, pts_rand.flatten(), method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
                pts_opt = res.x.reshape(N, 2)
                s = _compute_score(pts_opt)
                
                if s > best_score:
                    best_score = s
                    best_points = pts_opt.copy()
                    
                    refined = _refine_local(pts_opt, max_iters=500)
                    s_ref = _compute_score(refined)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_points = refined.copy()
            except Exception:
                pass
    
    # Final fallback if nothing worked
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Simple concentric rings
        ring_counts = [1, 6, 12, 18, 13]
        ring_radii = [0.0, 0.22, 0.44, 0.66, 0.88]
        best_points = _ring_initialization(ring_counts, ring_radii)
        best_score = _compute_score(best_points)
    
    return best_points
