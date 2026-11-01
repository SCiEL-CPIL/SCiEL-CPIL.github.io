"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess, optimized for N=50."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        
        spacing *= 0.98
        if spacing < 0.05:
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 0.81, N))
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init(seed):
    """Generate a random initial configuration with a given seed."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 0.85, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """Generate a ring-based initial configuration."""
    pts = np.zeros((N, 2))
    # Center point
    pts[0] = [0.0, 0.0]
    # Ring 1: 6 points
    r1 = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # Ring 2: 12 points
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        pts[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    # Ring 3: 18 points
    r3 = 0.78
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    # Remaining 14 points: place in gaps
    remaining = 14
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining + np.pi / 18
        r = 0.65
        pts[37 + i] = [r * np.cos(angle), r * np.sin(angle)]
    
    # Project to unit disk
    radii = np.hypot(pts[:, 0], pts[:, 1])
    mask = radii > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / radii[mask, np.newaxis]
    
    return pts


def _repulsion_optimize(pts0, max_time=120.0):
    """
    Vectorized repulsion optimization with momentum and adaptive step size.
    """
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    step_size = 0.02
    momentum = 0.5
    v = np.zeros_like(pts)
    stuck_count = 0
    
    start_time = time.time()
    
    while time.time() < min(start_time + max_time, _DEADLINE - 5.0):
        # 1. Compute Forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (N, N)
        np.fill_diagonal(dists, 1.0)
        
        # Weight: 1/d^2 for strong short-range repulsion
        weights = 1.0 / (dists * dists)
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j (p_i - p_j) * w_ij
        forces = np.sum(diff * weights[..., np.newaxis], axis=1)
        
        # Boundary force: push points back if they're too close to the edge
        dists_origin = np.linalg.norm(pts, axis=1)
        # Gentle inward pull for points near the boundary
        boundary_margin = 0.1
        near_boundary = dists_origin > (1.0 - boundary_margin)
        if np.any(near_boundary):
            boundary_forces = -0.1 * pts[near_boundary] / (dists_origin[near_boundary, np.newaxis] + 1e-10)
            forces[near_boundary] += boundary_forces
        
        # 2. Update Velocity and Position
        v = momentum * v + (1 - momentum) * forces
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        v = v / v_norms
        
        new_pts = pts + step_size * v
        
        # 3. Project to Unit Disk
        new_radii = np.linalg.norm(new_pts, axis=1)
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
            
        # 4. Evaluate
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            stuck_count = 0
            # Slightly increase step size if improving
            step_size = min(0.05, step_size * 1.02)
        else:
            stuck_count += 1
            step_size *= 0.99
            
            # If stuck for a while, try a random kick to escape local minimum
            if stuck_count > 40:
                kick_strength = 0.03
                pts = best_pts + np.random.normal(0, kick_strength, (N, 2))
                radii = np.linalg.norm(pts, axis=1)
                mask = radii > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / radii[mask, np.newaxis]
                v = np.zeros_like(pts)
                stuck_count = 0
    
    return best_pts


def _nelder_mead_polish(pts0, max_time=120.0):
    """
    High-precision polish using Nelder-Mead on the exact objective.
    Uses a multi-stage approach with decreasing tolerance.
    """
    def exact_obj(x):
        pts = x.reshape(N, 2)
        dists = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dists > 1.0):
            return 1e6 + np.sum(dists[dists > 1.0] - 1.0)
        r_cont = 1.0 - np.max(dists)
        if N < 2:
            return -r_cont
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, np.inf)
        r_pair = 0.5 * np.min(dist_mat)
        return -min(r_cont, r_pair)
    
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    # Multi-stage polishing with different initial simplex sizes
    stages = [
        {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'initial_simplex_size': 0.05},
        {'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'initial_simplex_size': 0.01},
        {'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex_size': 0.001},
    ]
    
    for stage in stages:
        if time.time() > _DEADLINE - 5.0:
            break
        
        try:
            # Create initial simplex
            x0 = best_pts.flatten()
            res = minimize(
                exact_obj,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': stage['maxiter'],
                    'xatol': stage['xatol'],
                    'fatol': stage['fatol'],
                    'adaptive': True,
                    'initial_simplex': x0 + np.random.normal(0, stage['initial_simplex_size'], (N, 2)).flatten()
                }
            )
            
            candidate_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            radii = np.hypot(candidate_pts[:, 0], candidate_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                candidate_pts[mask] = candidate_pts[mask] / radii[mask, np.newaxis]
            
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
                
        except Exception:
            pass
    
    return best_pts


def _perturbation_search(pts0, max_time=120.0):
    """
    Perturbation-based local search: try small random perturbations and keep improvements.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    start_time = time.time()
    perturbation_sizes = [0.05, 0.02, 0.01, 0.005]
    
    for size in perturbation_sizes:
        if time.time() > min(start_time + max_time, _DEADLINE - 5.0):
            break
        
        # Try multiple perturbations at this scale
        for _ in range(50):
            if time.time() > min(start_time + max_time, _DEADLINE - 5.0):
                break
            
            # Random perturbation
            perturbed = pts + np.random.normal(0, size, (N, 2))
            
            # Project to unit disk
            radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                perturbed[mask] = perturbed[mask] / radii[mask, np.newaxis]
            
            score = _score(perturbed)
            if score > best_score:
                best_score = score
                best_pts = perturbed.copy()
                pts = perturbed.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal lattice
    hex_pts = _hex_lattice_init()
    inits.append(hex_pts)
    
    # 2. Ring-based
    ring_pts = _ring_init()
    inits.append(ring_pts)
    
    # 3-5. Random with different seeds
    for seed in [42, 123, 456]:
        if time.time() > _DEADLINE - 30.0:
            break
        rand_pts = _random_init(seed)
        inits.append(rand_pts)
    
    # Optimize each initialization separately and keep the best
    best_pts = None
    best_score = 0.0
    
    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Repulsion optimization
        try:
            opt_pts = _repulsion_optimize(init_pts, max_time=30.0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass
        
        # Perturbation search
        if time.time() > _DEADLINE - 10.0:
            break
        try:
            pert_pts = _perturbation_search(best_pts if best_pts is not None else init_pts, max_time=15.0)
            pert_score = _score(pert_pts)
            if pert_score > best_score:
                best_score = pert_score
                best_pts = pert_pts
        except Exception:
            pass
    
    # Final high-precision polish
    if best_pts is not None and time.time() < _DEADLINE - 5.0:
        try:
            polish_pts = _nelder_mead_polish(best_pts, max_time=30.0)
            polish_score = _score(polish_pts)
            if polish_score > best_score:
                best_score = polish_score
                best_pts = polish_pts
        except Exception:
            pass
    
    # Final safety check
    if best_pts is None:
        best_pts = hex_pts
    
    final_score = _score(best_pts)
    if final_score <= 0:
        return hex_pts
    
    return best_pts
