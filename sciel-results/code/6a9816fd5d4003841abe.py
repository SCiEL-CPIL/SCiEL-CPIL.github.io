"""
Multi-start violation-energy optimization with cKDTree-accelerated radius
computation, iterative radius increase, and contact-graph refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C using cKDTree."""
    if len(C) < 2:
        return 0.5 - np.max(np.abs(C))
    
    # Boundary constraint
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    # Pairwise constraint using cKDTree
    tree = cKDTree(C)
    min_dist, _ = tree.query(C, k=2)  # k=2: self + nearest neighbor
    min_dist = min_dist[:, 1]  # exclude self
    r_pair = np.min(min_dist) / 2.0
    
    return min(r_bound, r_pair)

def _energy_and_grad(flat_pos, r_target):
    """
    Violation energy: penalize pair overlaps and wall violations.
    E = sum_{i<j} max(0, 2*r - d_ij)^2 + sum_{i,k} max(0, r - (HALF - |x_ik|))^2
    """
    pos = flat_pos.reshape(N, 3)
    
    # Use cKDTree for efficient pairwise distance computation
    tree = cKDTree(pos)
    # Query all pairs within distance 2*r_target
    pairs = tree.query_pairs(2.0 * r_target)
    
    energy = 0.0
    grad_pos = np.zeros_like(pos)
    
    # Pair violations
    for i, j in pairs:
        d = np.linalg.norm(pos[i] - pos[j])
        viol = max(0.0, 2.0 * r_target - d)
        if viol > 0:
            energy += viol * viol
            if d > 1e-14:
                vec = (pos[i] - pos[j]) / d
                grad_pos[i] += 2.0 * viol * vec
                grad_pos[j] -= 2.0 * viol * vec
    
    # Wall violations
    d_walls = HALF - np.abs(pos)
    wall_viol = np.maximum(0.0, r_target - d_walls)
    energy += np.sum(wall_viol**2)
    
    # Wall gradient
    mask = wall_viol > 0
    if np.any(mask):
        signs = np.sign(pos)
        signs[signs == 0] = 1.0
        grad_pos[mask] += 2.0 * wall_viol[mask] * signs[mask]
    
    return energy, -grad_pos.flatten()

def _optimize_bfgs(pos, r_target, max_iter=300, ftol=1e-14, gtol=1e-12):
    """Minimize violation energy for a fixed target radius using L-BFGS-B."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_and_grad(x, r_target)[0],
            pos.flatten(),
            jac=lambda x: _energy_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        new_pos = res.x.reshape(N, 3)
        if np.all(np.isfinite(new_pos)):
            return new_pos
    except Exception:
        pass
    return pos

def _contact_graph_refine(pos, r_target, n_iter=50):
    """
    Refine positions by optimizing only on the contact graph (pairs close to 2*r).
    This is faster than full pairwise and helps escape local minima.
    """
    for _ in range(n_iter):
        if _check_time():
            break
        
        tree = cKDTree(pos)
        # Get pairs within slightly more than 2*r
        pairs = tree.query_pairs(2.0 * r_target * 1.05)
        
        if len(pairs) == 0:
            break
        
        # Build sparse gradient from contact pairs only
        grad_pos = np.zeros_like(pos)
        for i, j in pairs:
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 1e-14:
                continue
            viol = max(0.0, 2.0 * r_target - d)
            if viol > 0:
                vec = (pos[i] - pos[j]) / d
                grad_pos[i] += 2.0 * viol * vec
                grad_pos[j] -= 2.0 * viol * vec
        
        # Wall violations
        d_walls = HALF - np.abs(pos)
        wall_viol = np.maximum(0.0, r_target - d_walls)
        mask = wall_viol > 0
        if np.any(mask):
            signs = np.sign(pos)
            signs[signs == 0] = 1.0
            grad_pos[mask] += 2.0 * wall_viol[mask] * signs[mask]
        
        # Gradient step with adaptive step size
        norm_g = np.linalg.norm(grad_pos)
        if norm_g < 1e-15:
            break
        
        # Simple line search
        alpha = 0.01
        for _ in range(10):
            new_pos = pos - alpha * grad_pos
            new_pos = np.clip(new_pos, -HALF + 1e-12, HALF - 1e-12)
            
            # Check if energy decreased
            old_energy, _ = _energy_and_grad(pos.flatten(), r_target)
            new_energy, _ = _energy_and_grad(new_pos.flatten(), r_target)
            
            if new_energy < old_energy:
                pos = new_pos
                break
            else:
                alpha *= 0.5
        
        else:
            break
    
    return pos

def _generate_grid(m):
    """Generate a simple grid of m^3 points, take first N."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]

def _generate_bcc(n_max=4):
    """Generate BCC lattice points, take first N by proximity to center."""
    points = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    center = points.mean(axis=0)
    points = points - center
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def construct():
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # Grid
    inits.append(_generate_grid(5))
    
    # BCC
    bcc = _generate_bcc(4)
    # Scale to fit
    bcc_extent = np.max(bcc.max(axis=0) - bcc.min(axis=0))
    bcc_scaled = bcc / (bcc_extent * 1.1 + 0.1)
    inits.append(bcc_scaled)
    
    # Random with noise
    for _ in range(3):
        rand = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rand)
    
    best_C = None
    best_r = 0.0
    
    for init in inits:
        if _check_time():
            break
        
        # Initial BFGS polish at a modest radius
        r_start = _compute_radius(init) * 0.95
        pos = _optimize_bfgs(init, r_start, max_iter=200)
        r = _compute_radius(pos)
        
        # Iterative radius increase
        for _ in range(20):
            if _check_time():
                break
            r_new_target = r * 1.005
            pos = _optimize_bfgs(pos, r_new_target, max_iter=100)
            r_check = _compute_radius(pos)
            if r_check > r:
                r = r_check
            else:
                break
        
        # Contact graph refinement
        pos = _contact_graph_refine(pos, r, n_iter=30)
        r = _compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_C = pos.copy()
    
    # Final aggressive polish
    if best_C is not None and not _check_time():
        # Try to push radius further
        for _ in range(10):
            if _check_time():
                break
            r_target = best_r * 1.002
            pos = _optimize_bfgs(best_C, r_target, max_iter=150)
            r_check = _compute_radius(pos)
            if r_check > best_r:
                best_r = r_check
                best_C = pos.copy()
            else:
                break
        
        # Contact graph refinement again
        best_C = _contact_graph_refine(best_C, best_r, n_iter=50)
        best_r = _compute_radius(best_C)
    
    # Safety check
    if best_C is None or not np.all(np.isfinite(best_C)):
        best_C = _generate_grid(5)
    
    best_C = np.clip(best_C, -HALF + 1e-12, HALF - 1e-12)
    return best_C
