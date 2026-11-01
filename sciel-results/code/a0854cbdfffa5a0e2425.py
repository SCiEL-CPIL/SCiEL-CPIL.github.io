"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

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
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _repulsion_optimize(pts0, max_time=120.0):
    """Optimize using vectorized repulsion forces between circles."""
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    step_size = 0.02
    n_iter = 0
    stuck_count = 0
    
    while time.time() < _DEADLINE - 10.0:
        n_iter += 1
        
        # Vectorized pairwise repulsion
        # Compute all pairwise distances
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist_mat, np.inf)
        
        # Force magnitude: strong repulsion when close
        # Use 1/d^2 for close pairs, 0 otherwise (smooth cutoff)
        cutoff = 0.5
        force_mag = np.where(dist_mat < cutoff, 1.0 / (dist_mat**2 + 1e-6), 0.0)
        np.fill_diagonal(force_mag, 0.0)
        
        # Direction vectors (unit vectors from j to i)
        dir_x = dx / (dist_mat + 1e-10)
        dir_y = dy / (dist_mat + 1e-10)
        np.fill_diagonal(dir_x, 0.0)
        np.fill_diagonal(dir_y, 0.0)
        
        # Sum forces on each point
        forces_x = np.sum(force_mag * dir_x, axis=1)
        forces_y = np.sum(force_mag * dir_y, axis=1)
        forces = np.column_stack([forces_x, forces_y])
        
        # Attraction toward boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        outward_strength = np.where(radii < 0.9, 0.15, 0.0)
        forces += outward_strength[:, np.newaxis] * pts / (radii[:, np.newaxis] + 1e-10)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update positions
        new_pts = pts + step_size * forces
        
        # Keep points inside the container
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_radii > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / new_radii[mask, np.newaxis]
        
        # Check if valid
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > 30:
                step_size *= 0.85
                stuck_count = 0
        
        # Occasionally add random perturbation
        if n_iter % 100 == 0:
            perturb = np.random.uniform(-0.03, 0.03, (N, 2))
            perturbed_pts = pts + perturb
            pert_radii = np.hypot(perturbed_pts[:, 0], perturbed_pts[:, 1])
            p_mask = pert_radii > 1.0
            if np.any(p_mask):
                perturbed_pts[p_mask] = perturbed_pts[p_mask] / pert_radii[p_mask, np.newaxis]
            p_score = _score(perturbed_pts)
            if p_score > best_score:
                best_score = p_score
                best_pts = perturbed_pts.copy()
        
        pts = new_pts
    
    return best_pts


def _objective(x):
    """Objective function: minimize negative radius with smooth soft-min."""
    pts = x.reshape(N, 2)
    
    # Check boundary constraint
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 1.0
    
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 1.0
    
    if N < 2:
        return -r_cont
    
    # Pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Soft-min of pairwise distances / 2 (smooth approximation to min)
    # Use log-sum-exp style: -log(sum(exp(-d/tau))) * tau approximates min(d)
    # For stability, use a moderate temperature
    tau = 0.05
    half_dists = 0.5 * dist_mat
    # Only consider finite values
    finite_mask = np.isfinite(half_dists)
    if not np.any(finite_mask):
        return -r_cont
    
    # Soft min: -tau * log(sum(exp(-d/tau)))
    # Shift for numerical stability
    shifted = half_dists[finite_mask]
    max_val = np.max(shifted)
    exp_vals = np.exp(-(shifted - max_val) / tau)
    soft_min = max_val - tau * np.log(np.sum(exp_vals))
    
    r = min(r_cont, soft_min)
    return -r


def _de_optimize(pts0, max_time=120.0):
    """
    Optimize using differential evolution with smooth objective.
    """
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def callback(xk, convergence):
        return time.time() > _DEADLINE - 15.0
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=80, 
            popsize=12, 
            tol=1e-5, 
            atol=1e-5, 
            seed=42, 
            callback=callback,
            polish=False
        )
        best_pts = res.x.reshape(N, 2)
        
        # Project back to unit disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis]
            
        return best_pts
        
    except Exception:
        return pts0


def _nelder_mead_polish(pts0, max_time=120.0):
    """Final high-precision polish using Nelder-Mead on the exact objective."""
    from scipy.optimize import minimize
    
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
    
    def callback_polish(xk):
        return time.time() > _DEADLINE - 5.0
    
    try:
        res = minimize(
            exact_obj,
            pts0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            callback=callback_polish
        )
        best_pts = res.x.reshape(N, 2)
        
        # Project back to unit disk
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            best_pts[mask] = best_pts[mask] / radii[mask, np.newaxis]
        
        return best_pts
    except Exception:
        return pts0


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run differential evolution optimization
    try:
        if time.time() < _DEADLINE - 20.0:
            de_pts = _de_optimize(pts0)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_pts = de_pts
                best_score = de_score
    except Exception:
        pass
    
    # Repulsion optimization for local refinement
    try:
        if time.time() < _DEADLINE - 15.0:
            opt_pts = _repulsion_optimize(best_pts)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # Final Nelder-Mead polish for high precision
    try:
        if time.time() < _DEADLINE - 5.0:
            polish_pts = _nelder_mead_polish(best_pts)
            polish_score = _score(polish_pts)
            if polish_score > best_score:
                best_pts = polish_pts
                best_score = polish_score
    except Exception:
        pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
