# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 300

def _hexagonal_lattice(r_target):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2.0 * r_target * 1.001
    sqrt3 = np.sqrt(3.0)
    
    points = []
    k_max = int(1.0 / spacing) + 5
    
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points well inside
    margin = r_target * 0.5
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    # Select best N points (closest to center first for stability)
    if len(points) >= N:
        idx = np.argsort(radii)
        return points[idx[:N]]
    else:
        # If not enough, return what we have (shouldn't happen for reasonable r)
        return points

def _build_spring_graph(centers, r_target, k=6):
    """Build spring connections for nearest neighbors in hexagonal lattice."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Find k nearest neighbors for each point
    edges = set()
    for i in range(n):
        nearest = np.argsort(dists[i])[:k]
        for j in nearest:
            if j > i:
                edges.add((i, j))
    
    return np.array(list(edges), dtype=int)

def _spring_energy(centers, edges, rest_length, stiffness=100.0):
    """Compute spring energy and gradient."""
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    if len(edges) == 0:
        return energy, grad
    
    for i, j in edges:
        d = centers[i] - centers[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        force = stiffness * (dist - rest_length) / dist
        grad[i] += force * d
        grad[j] -= force * d
        energy += 0.5 * stiffness * (dist - rest_length) ** 2
    
    return energy, grad

def _wall_penalty(centers, r_target, weight=50.0):
    """Compute penalty for circles too close to wall."""
    radii = np.linalg.norm(centers, axis=1)
    # We want radii <= 1 - r_target
    # Penalty: sum max(0, radii - (1-r_target))^2
    excess = np.maximum(0, radii - (1.0 - r_target))
    energy = weight * np.sum(excess ** 2)
    
    safe_radii = np.where(radii < 1e-12, 1e-12, radii)
    grad = 2.0 * weight * excess[:, np.newaxis] * centers / safe_radii[:, np.newaxis]
    
    return energy, grad

def _overlap_penalty(centers, r_target, weight=1000.0):
    """Compute penalty for overlapping circles."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    overlap = np.maximum(0, 2.0 * r_target - pair_dists)
    energy = weight * np.sum(overlap ** 2)
    
    # Gradient
    w = 2.0 * weight * overlap
    W_mat = np.zeros((n, n))
    W_mat[mask] = w
    W_mat = W_mat + W_mat.T
    
    normalized_diff = diff / (dists[:, :, np.newaxis] + 1e-12)
    grad = np.sum(W_mat[:, :, np.newaxis] * normalized_diff, axis=1)
    
    return energy, grad

def _total_energy(centers, r_target, edges, spring_stiffness=100.0):
    """Total energy: springs + wall + overlap."""
    rest_length = 2.0 * r_target
    
    e_spring, g_spring = _spring_energy(centers, edges, rest_length, spring_stiffness)
    e_wall, g_wall = _wall_penalty(centers, r_target)
    e_overlap, g_overlap = _overlap_penalty(centers, r_target)
    
    energy = e_spring + e_wall + e_overlap
    grad = g_spring + g_wall + g_overlap
    
    return energy, grad

def _optimize_with_springs(centers, r_target, max_iter=200):
    """Optimize using spring mechanics."""
    edges = _build_spring_graph(centers, r_target)
    rest_length = 2.0 * r_target
    
    x0 = centers.flatten()
    
    def obj(x):
        c = x.reshape(-1, 2)
        e, g = _total_energy(c, r_target, edges)
        return e, g
    
    result = minimize(
        obj,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return result.x.reshape(-1, 2)

def _boundary_rotation_optimization(centers, r_target):
    """Optimize boundary circles via rotation and radial adjustment."""
    n = len(centers)
    radii = np.linalg.norm(centers, axis=1)
    
    # Boundary circles: outermost ring
    boundary_mask = radii > np.percentile(radii, 70)
    boundary_idx = np.where(boundary_mask)[0]
    interior_idx = np.where(~boundary_mask)[0]
    
    if len(boundary_idx) < 3:
        return centers
    
    # Optimize: rotation angle + radial displacements for boundary circles
    # Parameterize: theta (rotation), delta_r (radial offset per boundary circle)
    
    def boundary_obj(params):
        theta = params[0]
        delta_r = params[1:]
        
        # Apply rotation
        c = centers.copy()
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        c[boundary_idx] = (c[boundary_idx] @ rot.T)
        
        # Apply radial displacements
        for i, idx in enumerate(boundary_idx):
            if radii[idx] > 1e-12:
                c[idx] += delta_r[i] * c[idx] / radii[idx]
        
        # Compute energy
        e, g = _total_energy(c, r_target, _build_spring_graph(c, r_target))
        
        # Gradient w.r.t params
        grad = np.zeros_like(params)
        # Approximate gradient via finite difference for rotation
        eps = 1e-4
        c_plus = centers.copy()
        cos_p, sin_p = np.cos(theta + eps), np.sin(theta + eps)
        rot_p = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
        c_plus[boundary_idx] = (c_plus[boundary_idx] @ rot_p.T)
        for i, idx in enumerate(boundary_idx):
            if radii[idx] > 1e-12:
                c_plus[idx] += delta_r[i] * c_plus[idx] / radii[idx]
        e_plus, _ = _total_energy(c_plus, r_target, _build_spring_graph(c_plus, r_target))
        grad[0] = (e_plus - e) / eps
        
        # Radial gradients via finite difference
        for i in range(len(delta_r)):
            c_r = c.copy()
            c_r[boundary_idx[i]] += eps * c_r[boundary_idx[i]] / max(radii[boundary_idx[i]], 1e-12)
            e_r, _ = _total_energy(c_r, r_target, _build_spring_graph(c_r, r_target))
            grad[i+1] = (e_r - e) / eps
        
        return e, grad
    
    x0 = np.zeros(1 + len(boundary_idx))
    result = minimize(
        boundary_obj,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
    )
    
    theta_opt = result.x[0]
    delta_r_opt = result.x[1:]
    
    # Apply optimal transformation
    c = centers.copy()
    cos_t, sin_t = np.cos(theta_opt), np.sin(theta_opt)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    c[boundary_idx] = (c[boundary_idx] @ rot.T)
    
    for i, idx in enumerate(boundary_idx):
        if radii[idx] > 1e-12:
            c[idx] += delta_r_opt[i] * c[idx] / radii[idx]
    
    # Safety: keep inside container
    radii_new = np.linalg.norm(c, axis=1)
    max_allowed = 1.0 - r_target
    mask_out = radii_new > max_allowed
    if np.any(mask_out):
        c[mask_out] *= (max_allowed / radii_new[mask_out])[:, np.newaxis]
    
    return c

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)

def construct():
    """
    Construct 300 equal circles in a unit circle using spring mechanics approach.
    """
    # Start with a reasonable radius estimate
    r_est = 1.0 / np.sqrt(N * 0.9069)  # Hexagonal packing density
    
    # Generate initial hexagonal lattice
    centers = _hexagonal_lattice(r_est)
    
    # Ensure we have exactly N points
    if len(centers) < N:
        # Generate more points and select
        spacing = 2.0 * r_est * 1.001
        sqrt3 = np.sqrt(3.0)
        points = []
        k_max = int(1.0 / spacing) + 10
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt3 / 2.0
                points.append([x, y])
        points = np.array(points)
        radii = np.hypot(points[:, 0], points[:, 1])
        mask = radii <= 1.0 - r_est * 0.5
        points = points[mask]
        radii = radii[mask]
        idx = np.argsort(radii)
        centers = points[idx[:N]]
    elif len(centers) > N:
        centers = centers[:N]
    
    # Phase 1: Spring optimization to find stable configuration
    current_r = r_est * 0.95
    centers = _optimize_with_springs(centers, current_r, max_iter=100)
    
    # Phase 2: Increase radius and re-optimize
    for r_target in [0.052, 0.0525, 0.053, 0.05315]:
        # Rebuild lattice for new radius
        new_centers = _hexagonal_lattice(r_target)
        if len(new_centers) >= N:
            centers = new_centers
        centers = _optimize_with_springs(centers, r_target, max_iter=150)
    
    # Phase 3: Boundary optimization
    centers = _boundary_rotation_optimization(centers, 0.05315)
    
    # Phase 4: Final polish
    centers = _optimize_with_springs(centers, 0.05315, max_iter=100)
    
    # Safety check
    final_score = _compute_score(centers)
    for i in range(len(centers)):
        r = np.linalg.norm(centers[i])
        if r > 1.0 - 1e-10:
            centers[i] *= (1.0 - 1e-10) / r
    
    return centers
# EVOLVE-BLOCK-END
