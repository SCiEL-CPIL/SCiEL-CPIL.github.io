import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of hull facets) in degrees.
    Returns (radius_in_degrees, worst_facet_index, worst_circumcenter, hull).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, -1, None, None

    max_radius = 0.0
    worst_facet_idx = -1
    worst_center = None

    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)

        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)

        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C

        R = math.acos(cos_R)

        if R > max_radius:
            max_radius = R
            worst_center = C
            worst_facet_idx = i

    return math.degrees(max_radius), worst_facet_idx, worst_center, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def _fibonacci_spiral(n):
    """Generate a Fibonacci spiral configuration on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _repulsion_step(points, alpha):
    """Perform one repulsion step to spread points out (vectorized)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_dist_cubed = 1.0 / (dist_sq ** 1.5)
    np.fill_diagonal(inv_dist_cubed, 0.0)
    
    forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
    
    forces_norm = np.linalg.norm(forces, axis=1, keepdims=True)
    forces = forces / np.where(forces_norm == 0, 1, forces_norm)
    
    new_points = points + alpha * forces
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    return new_points

def _lloyd_iteration(points, max_iter=200):
    """
    Lloyd's algorithm: replace each point with the centroid of its Voronoi cell.
    """
    points = points.copy()
    
    for _ in range(max_iter):
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                new_points[i] = points[i]
                continue
            
            verts = vor.vertices[region]
            verts = verts[np.all(np.isfinite(verts), axis=1)]
            
            if len(verts) == 0:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(verts, axis=0)
            norm = np.linalg.norm(centroid)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / norm
        
        diff = np.linalg.norm(new_points - points, axis=1)
        if np.max(diff) < 1e-8:
            points = new_points
            break
        
        points = new_points
    
    return points

def _spectral_optimization(points, max_iter=300, time_budget=30.0):
    """
    Spectral optimization: construct a PSD matrix from the point configuration,
    then iteratively update points as the top eigenvectors of a modified matrix
    that encourages uniform distribution.
    """
    points = points.copy()
    start_time = time.time()
    
    # Build initial Gram matrix
    G = points @ points.T
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Compute the Laplacian-like matrix that penalizes clustering
        # We want to maximize the minimum eigenvalue of the Gram matrix
        # while keeping points on the sphere
        
        # Use the matrix sign function approach to find the optimal PSD matrix
        # that is close to the current Gram matrix but has better spectral properties
        
        # Simple approach: add a small multiple of identity to encourage spread
        # and then take the top eigenvectors
        
        # Compute eigenvalues and eigenvectors of G
        try:
            eigvals, eigvecs = np.linalg.eigh(G)
        except np.linalg.LinAlgError:
            break
        
        # The top eigenvectors should correspond to well-spread points
        # We use a modified objective: maximize the sum of the smallest
        # eigenvalues (which correspond to the most "clustered" directions)
        
        # Gradient-based update on the Gram matrix
        # dG = -lambda_min * v_min * v_min^T + small regularization
        min_idx = np.argmin(eigvals)
        v_min = eigvecs[:, min_idx]
        
        # Update the Gram matrix to push up the minimum eigenvalue
        G_new = G + 0.1 * np.outer(v_min, v_min)
        
        # Project back to PSD by clipping negative eigenvalues
        eigvals_new, eigvecs_new = np.linalg.eigh(G_new)
        eigvals_new = np.maximum(eigvals_new, 0)
        G_new = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
        
        # Extract new points from the top eigenvectors of G_new
        # Actually, we need to extract the points themselves
        # The points are the rows of the matrix, so we need to
        # find a factorization G_new = P_new @ P_new^T
        
        # Use Cholesky-like decomposition
        try:
            L = np.linalg.cholesky(G_new + 1e-10 * np.eye(N))
        except np.linalg.LinAlgError:
            # Fallback: use SVD
            U, s, Vt = np.linalg.svd(G_new)
            L = U * np.sqrt(s)
        
        # The new points are the rows of L, normalized
        new_points = L
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / np.where(norms < 1e-10, 1, norms)
        
        # Check if we improved
        old_radius, _, _, _ = _compute_covering_radius(points)
        new_radius, _, _, _ = _compute_covering_radius(new_points)
        
        if new_radius < old_radius:
            points = new_points
        else:
            # Try smaller step
            G_new = G + 0.01 * np.outer(v_min, v_min)
            eigvals_new, eigvecs_new = np.linalg.eigh(G_new)
            eigvals_new = np.maximum(eigvals_new, 0)
            G_new = eigvecs_new @ np.diag(eigvals_new) @ eigvecs_new.T
            
            try:
                L = np.linalg.cholesky(G_new + 1e-10 * np.eye(N))
            except np.linalg.LinAlgError:
                U, s, Vt = np.linalg.svd(G_new)
                L = U * np.sqrt(s)
            
            new_points = L
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / np.where(norms < 1e-10, 1, norms)
            
            new_radius, _, _, _ = _compute_covering_radius(new_points)
            if new_radius < old_radius:
                points = new_points
                G = G_new
            else:
                # No improvement, break
                break
        
        G = G_new
    
    return points

def _gradient_ascent_on_worst_facet(points, hull, center, alpha):
    """
    Perform a gradient-ascent-like step: move the 3 vertices of the worst facet
    towards the circumcenter. Returns new points and whether it improved.
    """
    radius_before, facet_idx, _, _ = _compute_covering_radius(points)
    if facet_idx == -1 or center is None:
        return points, False
    
    simplex = hull.simplices[facet_idx]
    v0, v1, v2 = simplex
    
    new_points = points.copy()
    new_points[v0] = _move_point_towards(points[v0], center, alpha)
    new_points[v1] = _move_point_towards(points[v1], center, alpha)
    new_points[v2] = _move_point_towards(points[v2], center, alpha)
    
    radius_after, _, _, _ = _compute_covering_radius(new_points)
    
    if radius_after < radius_before:
        return new_points, True
    else:
        return points, False

def _soft_max_covering_radius(points, temperature=0.05):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    This provides gradient information for all facets simultaneously.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, None, None
    
    radii = []
    centers = []
    facets = []
    
    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]

        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)

        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])

        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue

        xT_M_x = np.dot(x, M @ x)

        if xT_M_x <= 1e-10:
            continue

        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        R = math.acos(cos_R)
        
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        radii.append(R)
        centers.append(C)
        facets.append(simplex)
    
    if not radii:
        return 90.0, None, None
    
    radii = np.array(radii)
    
    # Log-sum-exp for smooth max
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, None, None
    
    exp_terms = np.exp((radii - max_r) / temperature)
    sum_exp = np.sum(exp_terms)
    soft_max = max_r + temperature * np.log(sum_exp)
    
    # Weighted average of centers based on exp weights
    weights = exp_terms / sum_exp
    centers_arr = np.array(centers)
    weighted_center = np.sum(weights[:, np.newaxis] * centers_arr, axis=0)
    weighted_center = weighted_center / np.linalg.norm(weighted_center)
    
    return math.degrees(soft_max), weighted_center, hull

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_radius = 90.0
    
    # Phase 1: Global optimization via repulsion from multiple starts
    for seed in range(10):
        if time.time() > deadline - 70:
            break
        
        np.random.seed(seed * 7919 + 42)
        
        # Start with Fibonacci spiral plus jitter
        points = _fibonacci_spiral(N).copy()
        noise = np.random.randn(N, 3) * (0.02 + 0.01 * seed)
        points = points + noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Global repulsion to spread points
        alpha = 0.03
        for step in range(200):
            if time.time() > deadline - 70:
                break
            
            points = _repulsion_step(points, alpha)
            alpha *= 0.995
            
            # Every 20 steps, check the radius
            if step % 20 == 0 or step == 199:
                radius, _, _, _ = _compute_covering_radius(points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    # Phase 2: Spectral optimization to improve distribution
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()
    
    if time.time() < deadline - 60:
        spectral_budget = min(30.0, deadline - 60 - time.time())
        points = _spectral_optimization(points, max_iter=300, time_budget=spectral_budget)
        radius, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 3: Lloyd's algorithm to improve distribution
    if time.time() < deadline - 40:
        points = best_points.copy()
        points = _lloyd_iteration(points, max_iter=80)
        radius, _, _, _ = _compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
    
    # Phase 4: Local search from the best point found
    if best_points is not None:
        points = best_points.copy()
    else:
        points = _fibonacci_spiral(N)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        best_radius, _, _, _ = _compute_covering_radius(points)
        best_points = points.copy()
    
    alpha = 0.08
    stagnation_counter = 0
    max_stagnation = 50
    
    while time.time() < deadline - 25:
        radius, facet_idx, center, hull = _compute_covering_radius(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if center is None or facet_idx == -1:
            break
        
        simplex = hull.simplices[facet_idx]
        v0, v1, v2 = simplex
        
        improved = False
        
        # Try moving all 3 vertices of the worst facet towards its circumcenter
        new_points, imp = _gradient_ascent_on_worst_facet(points, hull, center, alpha)
        if imp:
            points = new_points
            improved = True
        else:
            # Try moving vertices individually
            for v_idx in [v0, v1, v2]:
                test_points = points.copy()
                test_points[v_idx] = _move_point_towards(points[v_idx], center, alpha)
                test_radius, _, _, _ = _compute_covering_radius(test_points)
                if test_radius < radius:
                    points = test_points
                    improved = True
                    break
        
        if not improved:
            alpha *= 0.8
            if alpha < 0.003:
                # Perturb the worst facet's vertices to escape local minimum
                noise = np.random.randn(3, 3) * 0.05
                new_points = points.copy()
                new_points[v0] = points[v0] + noise[0]
                new_points[v1] = points[v1] + noise[1]
                new_points[v2] = points[v2] + noise[2]
                
                for idx in [v0, v1, v2]:
                    new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
                
                pert_radius, _, _, _ = _compute_covering_radius(new_points)
                
                if pert_radius < best_radius * 1.02:
                    points = new_points
                    alpha = 0.06
                    stagnation_counter = 0
                else:
                    if stagnation_counter > max_stagnation:
                        break
                    else:
                        alpha = 0.02
    
    # Phase 5: High-precision polish pass with adaptive step sizes
    if time.time() < deadline - 10 and best_points is not None:
        points = best_points.copy()
        polish_alpha = 0.02
        for polish_iter in range(100):
            if time.time() > deadline - 10:
                break
            
            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break
            
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            improved = False
            # Try smaller steps in polish phase
            for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                step = polish_alpha * step_mult
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                # Try individual vertex moves with smaller steps
                for v_idx in [v0, v1, v2]:
                    for step_mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                        step = polish_alpha * step_mult
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                polish_alpha *= 0.65
                if polish_alpha < 0.0003:
                    break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    # Phase 6: Final fine-tuning with very small steps
    if time.time() < deadline - 3 and best_points is not None:
        points = best_points.copy()
        for final_iter in range(60):
            if time.time() > deadline - 3:
                break
            
            radius, facet_idx, center, hull = _compute_covering_radius(points)
            if center is None or facet_idx == -1:
                break
            
            simplex = hull.simplices[facet_idx]
            v0, v1, v2 = simplex
            
            improved = False
            for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                new_points = points.copy()
                new_points[v0] = _move_point_towards(points[v0], center, step)
                new_points[v1] = _move_point_towards(points[v1], center, step)
                new_points[v2] = _move_point_towards(points[v2], center, step)
                
                new_radius, _, _, _ = _compute_covering_radius(new_points)
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                # Try individual moves
                for v_idx in [v0, v1, v2]:
                    for step in [0.002, 0.001, 0.0005, 0.0002]:
                        test_points = points.copy()
                        test_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                        test_radius, _, _, _ = _compute_covering_radius(test_points)
                        if test_radius < radius:
                            points = test_points
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
            else:
                if radius < best_radius:
                    best_radius = radius
                    best_points = points.copy()
    
    return best_points
