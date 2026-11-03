"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(centres):
    """Compute the packing score (largest common radius)."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    if min_boundary < -1e-12:
        return 0.0
    d = pdist(centres)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, min_boundary)
    return r

def _compute_radius(centres):
    """Compute radius and identify critical pairs."""
    dist_boundary = 0.5 - np.abs(centres)
    min_boundary = np.min(dist_boundary)
    d = pdist(centres)
    min_pair = np.min(d)
    r = min(min_pair / 2.0, min_boundary)
    return r

def _soft_min_value_and_grad(d, beta):
    """Compute soft-min value and gradient weights."""
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    value = d_min - np.log(s) / beta
    weights = terms / s
    return value, weights

def _objective_and_grad(x, beta):
    """
    Objective: minimize -soft_min(min_pair/2, min_face)
    Returns (value, gradient). Fully vectorized.
    """
    c = x.reshape(N, 3)
    
    # Pairwise distances (vectorized)
    i, j = np.triu_indices(N, 1)
    diff = c[i] - c[j]
    dist_sq = np.sum(diff**2, axis=1)
    dist = np.sqrt(dist_sq + 1e-15)
    
    # Soft min of pairwise distances
    min_pair_soft, w_pair = _soft_min_value_and_grad(dist, beta)
    
    # Boundary distances (flatten to 1D)
    face_dists = 0.5 - np.abs(c)  # shape (N, 3)
    face_flat = face_dists.flatten()
    min_face_soft, w_face_flat = _soft_min_value_and_grad(face_flat, beta)
    
    # Outer soft min: r_soft = soft_min([min_pair/2, min_face])
    a = min_pair_soft / 2.0
    b = min_face_soft
    outer_beta = beta * 0.5
    r_soft, w_outer = _soft_min_value_and_grad(np.array([a, b]), outer_beta)
    
    value = -r_soft
    
    # Gradient of min_pair_soft w.r.t. c (vectorized)
    # w_pair[k] is d(min_pair_soft)/d(dist_k)
    # d(dist_k)/d(c_i) = -diff_k / dist_k, d(dist_k)/d(c_j) = diff_k / dist_k
    w_scaled = w_pair / dist  # shape (n_pairs,)
    # For each pair k: grad[i_k] -= w_scaled[k] * diff[k], grad[j_k] += w_scaled[k] * diff[k]
    grad_pair = np.zeros((N, 3))
    np.add.at(grad_pair, i, -w_scaled[:, None] * diff)
    np.add.at(grad_pair, j, w_scaled[:, None] * diff)
    
    # Gradient of min_face_soft w.r.t. c
    # d(face_dists[n,k])/d(c[n,k]) = -sign(c[n,k])
    signs = np.sign(c)
    grad_face = -w_face_flat.reshape(N, 3) * signs
    
    # Total gradient
    grad = w_outer[0] * 0.5 * grad_pair + w_outer[1] * grad_face
    
    return value, grad.flatten()

def _generate_bcc_lattice(N_local):
    """Generate BCC lattice points scaled to fit N_local spheres in unit cube."""
    # BCC: 8 corners + 1 body center per cell
    # For 50 spheres, use a 3x3x3 grid of BCC cells = 27*2 = 54 points, take 50
    m = int(np.ceil((N_local / 2.0) ** (1.0 / 3.0)))
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                points.append([(ix + 0.0) / m - 0.5, (iy + 0.0) / m - 0.5, (iz + 0.0) / m - 0.5])
                # Body center
                points.append([(ix + 0.5) / m - 0.5, (iy + 0.5) / m - 0.5, (iz + 0.5) / m - 0.5])
    
    pts = np.array(points[:N_local])
    return pts

def _build_initial_grid(N_local):
    """Build an initial grid configuration."""
    m = int(np.ceil(N_local ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N_local]
    return grid

def _optimize_from_init(init, beta_schedule, max_iter_per_beta=300, deadline=None):
    """
    Optimize from an initial configuration using a beta schedule.
    Uses vectorized analytic gradient for speed.
    """
    current = init.copy()
    
    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break
        
        x0 = current.flatten()
        
        def obj_and_grad(x):
            return _objective_and_grad(x, beta)
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 40}
            )
            current = result.x.reshape(N, 3)
        except Exception:
            pass
    
    return current

def _generate_fcc_lattice(N_local):
    """Generate FCC lattice points scaled to fit N_local spheres in unit cube."""
    # FCC: 8 corners + 6 face centers per cell
    # For 50 spheres, use a 3x3x3 grid of FCC cells = 27 * (8+6)/8 = 27*1.75 ~ 47, need to adjust
    m = 3
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner
                points.append([(ix + 0.0) / m - 0.5, (iy + 0.0) / m - 0.5, (iz + 0.0) / m - 0.5])
                # Face centers
                if ix < m - 1:
                    points.append([(ix + 0.5) / m - 0.5, (iy + 0.0) / m - 0.5, (iz + 0.0) / m - 0.5])
                if iy < m - 1:
                    points.append([(ix + 0.0) / m - 0.5, (iy + 0.5) / m - 0.5, (iz + 0.0) / m - 0.5])
                if iz < m - 1:
                    points.append([(ix + 0.0) / m - 0.5, (iy + 0.0) / m - 0.5, (iz + 0.5) / m - 0.5])
    
    # Deduplicate
    pts = np.array(points)
    # Remove duplicates
    if len(pts) > 0:
        _, idx = np.unique(np.round(pts, 10), axis=0, return_index=True)
        pts = pts[np.sort(idx)]
    
    pts = pts[:N_local]
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = np.random.RandomState(42)
    
    # Best solution found so far
    best_centres = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = []
    
    # BCC lattice
    bcc_init = _generate_bcc_lattice(N)
    bcc_init = np.clip(bcc_init, -0.49, 0.49)
    inits.append(bcc_init)
    
    # FCC lattice
    fcc_init = _generate_fcc_lattice(N)
    fcc_init = np.clip(fcc_init, -0.49, 0.49)
    inits.append(fcc_init)
    
    # Grid
    grid_init = _build_initial_grid(N)
    inits.append(grid_init)
    
    # Beta schedule - finer continuation
    beta_schedule = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    # Optimize from each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        init_score = _score(init)
        if init_score > best_score:
            best_centres = init.copy()
            best_score = init_score
        
        refined = _optimize_from_init(init, beta_schedule, max_iter_per_beta=500, deadline=deadline)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_centres = refined.copy()
            best_score = refined_score
    
    # Random restarts from BCC with jitter
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        init_r = _generate_bcc_lattice(N)
        jitter = rng.uniform(-0.04, 0.04, size=init_r.shape)
        init_r += jitter
        init_r = np.clip(init_r, -0.49, 0.49)
        
        refined_r = _optimize_from_init(init_r, beta_schedule, max_iter_per_beta=400, deadline=deadline)
        refined_r_score = _score(refined_r)
        if refined_r_score > best_score:
            best_centres = refined_r.copy()
            best_score = refined_r_score
    
    # Final high-precision polish
    if time.time() < deadline - 10 and best_centres is not None:
        polish_betas = [2000.0, 5000.0, 10000.0]
        final = _optimize_from_init(best_centres, polish_betas, max_iter_per_beta=800, deadline=deadline)
        final_score = _score(final)
        if final_score > best_score:
            best_centres = final.copy()
            best_score = final_score
    
    # Ensure validity
    if best_centres is None:
        return _build_initial_grid(N)
    
    # Final clip to ensure inside cube
    best_centres = np.clip(best_centres, -0.499, 0.499)
    
    return best_centres
# EVOLVE-BLOCK-END
