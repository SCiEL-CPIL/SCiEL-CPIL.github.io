"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import scipy.linalg as sla


def _compute_radius(pts):
    """
    Compute the packing radius for given points using KD-tree for efficiency.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    if len(pts) < 2:
        return 0.0
    tree = cKDTree(pts)
    pairs = tree.query_pairs(0.2)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d < min_pair_dist:
                min_pair_dist = d
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _init_hex(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _spectral_gradient(pts, lam_target=0.02, pair_threshold=0.25):
    """
    Compute gradient of a spectral objective based on Delaunay graph Laplacian.
    Maximizes algebraic connectivity (Fiedler eigenvalue) for structural rigidity.
    Returns gradient to add to pts to increase spectral objective.
    """
    n = len(pts)
    if n < 3:
        return np.zeros_like(pts)
    
    try:
        # Build Delaunay triangulation
        tri = Delaunay(pts)
        edges = tri.simplices
        # Convert to edge list (unique edges)
        edge_set = set()
        for simplex in edges:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edge_set.add((a, b))
        
        if len(edge_set) == 0:
            return np.zeros_like(pts)
        
        edge_list = list(edge_set)
        m = len(edge_list)
        
        # Build sparse Laplacian
        rows = []
        cols = []
        vals = []
        
        for a, b in edge_list:
            diff = pts[a] - pts[b]
            dist = np.hypot(diff[0], diff[1])
            if dist > 1e-12:
                w = 1.0 / (dist ** 2 + 1e-10)  # Weight inversely with distance squared
                # L[a,a] += w, L[b,b] += w, L[a,b] -= w, L[b,a] -= w
                rows.extend([a, b, a, b])
                cols.extend([a, b, b, a])
                vals.extend([w, w, -w, -w])
        
        L = csr_matrix((vals, (rows, cols)), shape=(n, n))
        
        # Compute Fiedler eigenvalue (second smallest) and eigenvector
        # Use shift-invert to find eigenvalues near 0
        try:
            vals_eig, vecs_eig = eigs(L, k=2, sigma=0.0, which='LM', tol=1e-6)
            # Sort by eigenvalue
            idx = np.argsort(vals_eig.real)
            vals_eig = vals_eig[idx].real
            vecs_eig = vecs_eig[:, idx]
            
            # Fiedler eigenvalue is the second smallest
            if len(vals_eig) < 2:
                return np.zeros_like(pts)
            
            lambda_2 = vals_eig[1]
            v2 = vecs_eig[:, 1]
            
            # Gradient of lambda_2 w.r.t. positions
            # d(lambda_2)/d(x_p) = 2 * sum_{(a,b) in E} w_ab * (v2[a]-v2[b]) * d(dist_ab)/d(x_p)
            # Actually, for Laplacian, the gradient of eigenvalue lambda w.r.t. entry L[i,j] is (v_i*v_j)
            # And dL[i,j]/d(x_p) for p in {i,j}
            
            grad = np.zeros_like(pts)
            for a, b in edge_list:
                diff = pts[a] - pts[b]
                dist = np.hypot(diff[0], diff[1])
                if dist < 1e-12:
                    continue
                w = 1.0 / (dist ** 2 + 1e-10)
                unit_vec = diff / dist
                
                # dL[a,a]/dx_p and dL[a,b]/dx_p
                # L[a,a] = sum of w for edges from a, L[a,b] = -w for edge (a,b)
                # dL[a,a]/dx_a = -2*w*unit_vec (since w = 1/dist^2, dw/dx_a = -2w*unit_vec/dist... let me be careful)
                
                # w = 1/(dist^2), dist = |x_a - x_b|
                # dw/dx_a = d(1/dist^2)/dx_a = -2/dist^3 * d(dist)/dx_a = -2/dist^3 * unit_vec
                # dL[a,a]/dx_a = dw/dx_a = -2/dist^3 * unit_vec
                # dL[a,b]/dx_a = -dw/dx_a = 2/dist^3 * unit_vec
                
                # Gradient contribution from entry (a,a): v2[a]^2 * dL[a,a]/dx_a
                # Gradient contribution from entry (a,b): v2[a]*v2[b] * dL[a,b]/dx_a
                # Similarly for x_b
                
                coeff_a = v2[a] ** 2 * (-2.0 / dist ** 3) + v2[a] * v2[b] * (2.0 / dist ** 3)
                coeff_b = v2[b] ** 2 * (-2.0 / dist ** 3) + v2[a] * v2[b] * (2.0 / dist ** 3)
                
                grad[a] += coeff_a * unit_vec
                grad[b] -= coeff_b * unit_vec
            
            # Scale gradient
            grad_norm = np.hypot(grad[:, 0], grad[:, 1])
            max_grad = np.max(grad_norm) if len(grad_norm) > 0 else 0.0
            if max_grad > 1e-12:
                grad /= max_grad
            
            return grad
            
        except Exception:
            return np.zeros_like(pts)
            
    except Exception:
        return np.zeros_like(pts)


def _soft_min_gradient(pts, eps, pair_threshold=0.25):
    """
    Compute gradient of soft-min surrogate using KD-tree for efficiency.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(pair_threshold)
    
    for i, j in pairs:
        diff = pts[i] - pts[j]
        dist = np.hypot(diff[0], diff[1])
        if dist > 1e-12:
            w = np.exp(-eps * dist)
            unit_vec = diff / dist
            grad[i] += w * unit_vec
            grad[j] -= w * unit_vec
    
    radii = np.hypot(pts[:, 0], pts[:, 1])
    b_dists = 1.0 - radii
    safe_norms = np.where(radii > 1e-12, radii, 1.0)
    unit_radii = pts / safe_norms[:, np.newaxis]
    w_bound = np.exp(-eps * b_dists)
    grad += w_bound[:, np.newaxis] * unit_radii
    
    return grad


def _soft_min_optimize(pts, deadline, eps_start=5.0, eps_end=200.0,
                       lr_start=0.001, lr_end=1e-6, max_iter=5000, pair_threshold=0.25,
                       spectral_weight=0.1):
    """
    Gradient ascent on soft-min surrogate with optional spectral regularization.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress
        
        grad = _soft_min_gradient(best_pts, eps, pair_threshold)
        
        # Add spectral gradient for structural rigidity
        if spectral_weight > 0 and it % 50 == 0:
            spec_grad = _spectral_gradient(best_pts, pair_threshold=pair_threshold)
            grad += spectral_weight * spec_grad
        
        grad_norm = np.hypot(grad[:, 0], grad[:, 1])
        grad_norm[grad_norm < 1e-12] = 1.0
        grad /= grad_norm[:, np.newaxis]
        
        candidate = best_pts + lr * grad
        
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]
        
        cand_r = _compute_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.95
    
    return best_pts, best_r


def _hertzian_dynamics(pts, max_steps=3000, dt=0.001, k_repel=50.0, k_boundary=30.0,
                       damping=0.98, deadline=None, target_margin=0.005, pair_threshold=0.15):
    """
    Hertzian contact dynamics simulation using Verlet integration.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    v = np.zeros_like(pts)
    pos = pts.copy()
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(pos)
        
        tree = cKDTree(pos)
        pairs = tree.query_pairs(pair_threshold)
        
        for i, j in pairs:
            diff = pos[i] - pos[j]
            dist = np.hypot(diff[0], diff[1])
            if dist > 1e-12:
                overlap = max(0.0, pair_threshold - dist)
                if overlap > 0:
                    force_mag = k_repel * (overlap ** 4)
                    unit_vec = diff / dist
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        
        norms = np.hypot(pos[:, 0], pos[:, 1])
        margin = 1.0 - norms
        boundary_active = margin < target_margin
        if np.any(boundary_active):
            safe_norms = np.where(norms[boundary_active] > 1e-12, norms[boundary_active], 1.0)
            inward_dir = -pos[boundary_active] / safe_norms[:, np.newaxis]
            boundary_force_mag = k_boundary * ((target_margin - margin[boundary_active]) / target_margin) ** 2
            forces[boundary_active] += inward_dir * boundary_force_mag[:, np.newaxis]
        
        v = damping * v + dt * forces
        pos = pos + dt * v
        
        norms_cand = np.hypot(pos[:, 0], pos[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            pos[too_far] *= 0.9999 / norms_cand[too_far]
            v[too_far] *= 0.5
        
        if step % 50 == 0 or step == max_steps - 1:
            cand_r = _compute_radius(pos)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = pos.copy()
        
        force_norm = np.hypot(forces[:, 0], forces[:, 1])
        if np.max(force_norm) < 1e-8:
            break
    
    return best_pts, best_r


def _polish(pts, deadline):
    """
    Final high-precision polish with very small steps.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]:
        if time.time() > deadline:
            break
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            eps = 100.0
            grad = _soft_min_gradient(best_pts, eps, pair_threshold=0.2)
            
            grad_norm = np.hypot(grad[:, 0], grad[:, 1])
            grad_norm[grad_norm < 1e-12] = 1.0
            grad /= grad_norm[:, np.newaxis]
            
            candidate = best_pts + lr * grad
            
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 0.99999
            if np.any(too_far):
                candidate[too_far] *= 0.99999 / norms_cand[too_far]
            
            cand_r = _compute_radius(candidate)
            if cand_r > best_r + 1e-15:
                best_r = cand_r
                best_pts = candidate.copy()
    
    return best_pts, best_r


def _random_perturb(pts, magnitude=0.005, rng=None):
    """Add small random perturbation to escape local minima."""
    if rng is None:
        rng = np.random.default_rng()
    perturbed = pts + rng.normal(0, magnitude, size=pts.shape)
    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
    too_far = norms > 0.99
    if np.any(too_far):
        perturbed[too_far] *= 0.99 / norms[too_far]
    return perturbed


def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    # Phase 0: Search over hexagonal spacings for good initialization
    best_pts = None
    best_r = 0.0
    
    spacings = np.linspace(0.09, 0.13, 30)
    for spacing in spacings:
        if time.time() > deadline:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius(best_pts)
                break
            spacing *= 0.98
    
    # Phase 1: Hertzian dynamics simulation
    if time.time() < deadline - 50.0:
        best_pts, best_r = _hertzian_dynamics(
            best_pts,
            max_steps=3000,
            dt=0.001,
            k_repel=50.0,
            k_boundary=30.0,
            damping=0.98,
            deadline=deadline - 50.0,
            target_margin=0.005,
            pair_threshold=0.15
        )
    
    # Phase 2: Soft-min gradient optimization with spectral regularization
    if time.time() < deadline - 25.0:
        best_pts, best_r = _soft_min_optimize(
            best_pts,
            deadline=deadline - 25.0,
            eps_start=5.0,
            eps_end=200.0,
            lr_start=0.001,
            lr_end=1e-6,
            max_iter=5000,
            pair_threshold=0.25,
            spectral_weight=0.15
        )
    
    # Phase 3: Random perturbations + re-optimize to escape local minima
    if time.time() < deadline - 12.0:
        for attempt in range(8):
            if time.time() > deadline - 12.0:
                break
            perturbed = _random_perturb(best_pts, magnitude=0.004, rng=rng)
            pert_r = _compute_radius(perturbed)
            if pert_r >= best_r * 0.995:
                p_pts, p_r = _hertzian_dynamics(
                    perturbed,
                    max_steps=1500,
                    dt=0.0008,
                    k_repel=40.0,
                    k_boundary=25.0,
                    damping=0.985,
                    deadline=deadline - 12.0,
                    target_margin=0.003,
                    pair_threshold=0.12
                )
                p_pts, p_r = _soft_min_optimize(
                    p_pts,
                    deadline=deadline - 12.0,
                    eps_start=50.0,
                    eps_end=300.0,
                    lr_start=0.0008,
                    lr_end=5e-7,
                    max_iter=1500,
                    pair_threshold=0.2,
                    spectral_weight=0.1
                )
                if p_r > best_r:
                    best_pts = p_pts
                    best_r = p_r
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1.0:
        best_pts, best_r = _polish(best_pts, deadline - 1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
