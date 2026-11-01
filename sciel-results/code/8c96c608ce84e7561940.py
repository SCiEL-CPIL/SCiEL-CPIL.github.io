import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the given points on the unit sphere.
    Returns the radius in radians.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        n_theta, n_phi = 60, 120
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        x = np.sin(T) * np.cos(P)
        y = np.sin(T) * np.sin(P)
        z = np.cos(T)
        grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        min_dists = np.min(np.arccos(np.clip(grid @ points.T, -1, 1)), axis=1)
        return np.max(min_dists)

    worst_radius = 0.0
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        if r > worst_radius:
            worst_radius = r
    
    return worst_radius

def _get_worst_facet(points):
    """
    Get the worst facet (largest circumradius) and its circumcenter.
    Returns (radius, circumcenter, vertex_indices) or (inf, None, None) on failure.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.inf, None, None

    worst_radius = 0.0
    worst_center = None
    worst_vertices = None

    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        if r > worst_radius:
            worst_radius = r
            worst_center = n
            worst_vertices = simplex.copy()
    
    if worst_center is None:
        return np.inf, None, None
    
    return worst_radius, worst_center, worst_vertices

def _local_search(points, time_limit=20.0, start_time=None):
    """
    Local search using gradient-based optimization on the sphere.
    Uses exact facet-based covering radius with analytic gradients.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    best = current.copy()
    best_radius = _covering_radius(current)
    
    def _facet_radii_and_normals(pts):
        """Compute all facet circumradii and outward normals."""
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except Exception:
            return None, None
        radii = []
        normals = []
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                radii.append(0.0)
                normals.append(np.zeros(3))
                continue
            n = n / norm_n
            if np.dot(n, a) < 0:
                n = -n
            r = np.arccos(np.clip(np.dot(n, a), -1, 1))
            radii.append(r)
            normals.append(n)
        return np.array(radii), np.array(normals)
    
    def _objective_and_grad(flat_params):
        """Compute soft-max of facet radii and its gradient w.r.t. flat params."""
        pts = flat_params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        radii, normals = _facet_radii_and_normals(pts)
        if radii is None:
            return 1e10, np.zeros_like(flat_params)
        
        # Soft-max with adaptive beta
        beta = 60.0
        scaled = -beta * radii
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        obj = -(max_scaled + np.log(sum_exp)) / beta
        
        # Gradient of soft-max w.r.t. radii
        weights = exp_scaled / sum_exp  # softmax weights
        grad_radii = -beta * weights  # d(obj)/d(radii)
        
        # Gradient of each facet radius w.r.t. its three vertices
        # For facet with vertices i,j,k and normal n: r = arccos(n·a)
        # dr/d(a) = -n / sqrt(1-(n·a)^2) projected to tangent plane
        # dr/d(b) and dr/d(c) similar via n = cross(b-a, c-a)
        
        grad_flat = np.zeros_like(flat_params)
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except Exception:
            return obj, grad_flat
        
        for fi, simplex in enumerate(hull.simplices):
            w = grad_radii[fi]
            if abs(w) < 1e-12:
                continue
            i, j, k = simplex
            a = pts[i]
            b = pts[j]
            c = pts[k]
            n = normals[fi]
            
            # dr/da = -n / sqrt(1-(n·a)^2), projected to tangent plane of a
            dot_na = np.dot(n, a)
            denom = np.sqrt(max(1e-15, 1.0 - dot_na * dot_na))
            da = -n / denom
            da = da - np.dot(da, a) * a  # project to tangent plane
            
            # For dr/db and dr/dc, we need derivative of n w.r.t. b and c
            # n = cross(b-a, c-a) / |cross(b-a, c-a)|
            # This is complex; use finite difference for these
            eps = 1e-7
            for vi, v in [(j, b), (k, c)]:
                for d in range(3):
                    pert = pts.copy()
                    pert[vi][d] += eps
                    pert[vi] = pert[vi] / np.linalg.norm(pert[vi])
                    # Recompute just this facet's normal and radius
                    pa = pert[i]
                    pb = pert[j]
                    pc = pert[k]
                    npert = np.cross(pb - pa, pc - pa)
                    npert_norm = np.linalg.norm(npert)
                    if npert_norm < 1e-10:
                        continue
                    npert = npert / npert_norm
                    if np.dot(npert, pa) < 0:
                        npert = -npert
                    r_pert = np.arccos(np.clip(np.dot(npert, pa), -1, 1))
                    dr_dv = (r_pert - radii[fi]) / eps
                    # Project to tangent plane of pert[vi]
                    dv = dr_dv * np.ones(3)
                    dv[d] = 1.0  # only component d changed
                    # Actually, gradient w.r.t. pert[vi] in direction e_d
                    # Project: remove radial component
                    radial = np.dot(dv, pert[vi]) * pert[vi]
                    tangent = dv - radial
                    grad_flat[vi * 3 + d] = w * tangent[d]
            
            # Add contribution from vertex a
            grad_flat[i * 3:i * 3 + 3] += w * da
        
        # Project gradient to tangent spaces (remove radial components)
        for idx in range(N):
            radial = np.dot(grad_flat[idx * 3:idx * 3 + 3], pts[idx]) * pts[idx]
            grad_flat[idx * 3:idx * 3 + 3] -= radial
        
        return obj, grad_flat
    
    # Run L-BFGS-B optimization
    x0 = current.ravel()
    
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
        
        if restart > 0:
            rng = np.random.default_rng(42 + restart)
            noise = rng.normal(0, 0.003, size=(N, 3))
            x0 = (current + noise).ravel()
            norms = np.linalg.norm(x0.reshape(N, 3), axis=1, keepdims=True)
            x0 = (x0.reshape(N, 3) / norms).ravel()
        
        try:
            result = scipy.optimize.minimize(
                _objective_and_grad, x0, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-8}
            )
            
            new_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best = new_pts.copy()
                current = new_pts.copy()
        except Exception:
            pass
    
    return best, best_radius

def _lloyd_iteration(points):
    """
    One iteration of Lloyd's algorithm for spherical k-means.
    """
    # Assign each point on a grid to its nearest site
    n_theta, n_phi = 60, 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(T) * np.cos(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(T)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Find nearest site for each grid point
    dots = grid @ points.T
    dots = np.clip(dots, -1, 1)
    nearest = np.argmax(dots, axis=1)
    
    # Compute new centroids as normalized sum of assigned grid points
    new_points = np.zeros_like(points)
    for i in range(N):
        mask = nearest == i
        if np.sum(mask) > 0:
            new_points[i] = np.sum(grid[mask], axis=0)
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-10:
                new_points[i] = new_points[i] / norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    
    return new_points

def _repulsion_relax(points, max_iters=50, time_limit=10.0, start_time=None):
    """
    Relax points using repulsive forces to spread them out.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.arccos(np.clip(np.dot(current[i], current[j]), -1, 1))
                if d < 1e-6:
                    continue
                f = (1.0 / (d * d)) * (current[i] - current[j])
                forces[i] += f
        
        # Normalize forces to tangent plane
        for i in range(N):
            radial = np.dot(forces[i], current[i]) * current[i]
            forces[i] = forces[i] - radial
        
        # Apply forces with adaptive step
        step = 0.01 * (1.0 - it / max_iters)
        new_points = current.copy()
        for i in range(N):
            new_points[i] = current[i] + step * forces[i]
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-10:
                new_points[i] = new_points[i] / norm
        
        current = new_points
    
    return current

def _worst_facet_step(points, step_size=0.05, time_limit=10.0, start_time=None):
    """
    Move the vertices of the worst facet toward its circumcenter to reduce the covering radius.
    Vectorized where possible for speed.
    """
    if start_time is None:
        start_time = time.time()
    
    current = points.copy()
    best = current.copy()
    best_radius = _covering_radius(current)
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        radius, worst_center, worst_vertices = _get_worst_facet(current)
        if worst_vertices is None:
            break
        
        if radius < best_radius:
            best_radius = radius
            best = current.copy()
        
        # Move the three vertices of the worst facet toward the circumcenter
        for idx in worst_vertices:
            direction = worst_center - current[idx]
            radial = np.dot(direction, current[idx]) * current[idx]
            tangent = direction - radial
            norm_tangent = np.linalg.norm(tangent)
            if norm_tangent < 1e-10:
                continue
            new_point = current[idx] + step_size * tangent / norm_tangent
            norm_new = np.linalg.norm(new_point)
            if norm_new > 1e-10:
                new_point = new_point / norm_new
            current[idx] = new_point
        
        # Vectorized repulsion from worst facet vertices for nearby points
        wv = worst_vertices
        for j in wv:
            # Compute distances from all points to current[j]
            dots = np.clip(current @ current[j], -1, 1)
            dists = np.arccos(dots)
            # Only repel points that are close (d < 0.3 rad)
            close_mask = (dists < 0.3) & (np.arange(N) != j)
            if np.any(close_mask):
                # Directions from current[j] to nearby points
                dirs = current[close_mask] - current[j]
                # Project to tangent plane of each point
                radial_comps = np.sum(dirs * current[close_mask], axis=1, keepdims=True) * current[close_mask]
                tangents = dirs - radial_comps
                norms_t = np.linalg.norm(tangents, axis=1, keepdims=True)
                norms_t = np.where(norms_t < 1e-10, 1.0, norms_t)
                # Apply small repulsion
                step_rep = 0.01 * (0.3 - dists[close_mask])[:, None] / 0.3  # scale by closeness
                current[close_mask] += step_rep * tangents / norms_t
                # Renormalize
                norms_c = np.linalg.norm(current[close_mask], axis=1, keepdims=True)
                current[close_mask] = current[close_mask] / np.where(norms_c < 1e-10, 1.0, norms_c)
    
    return best, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_radius = np.inf
    
    # Phase 1: Multiple restarts with different initializations
    n_restarts = 12
    
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        rng = np.random.default_rng(seed=2000 + i * 13)
        
        # Different initialization strategies
        if i % 3 == 0:
            # Fibonacci with varying noise
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.03 + 0.015 * (i % 3), size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif i % 3 == 1:
            # Random uniform on sphere
            points = rng.normal(0, 1, size=(N, 3))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Fibonacci with larger perturbation
            j = np.arange(N) + 0.5
            z = 1.0 - 2.0 * j / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * j
            r = np.sqrt(1.0 - z * z)
            base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            noise = rng.normal(0, 0.08, size=(N, 3))
            points = base_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Step 1: Repulsion relaxation for initial spread
        remaining = deadline - time.time()
        if remaining < 25:
            break
        
        points = _repulsion_relax(points, max_iters=25, time_limit=min(6.0, remaining - 25), start_time=time.time())
        
        # Step 2: Lloyd's iterations
        for _ in range(4):
            if time.time() > deadline - 20:
                break
            points = _lloyd_iteration(points)
        
        # Step 3: Worst facet gradient step
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        pts, rad = _worst_facet_step(points, step_size=0.04, time_limit=min(10.0, remaining - 15), start_time=time.time())
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
        
        # Step 4: Local search with gradient-based optimization
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        pts, rad = _local_search(points, time_limit=min(20.0, remaining - 8), start_time=time.time())
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
    
    # Phase 2: Focused refinement on best configuration
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            # Multiple refinement passes
            for pass_i in range(3):
                remaining = deadline - time.time()
                if remaining < 8:
                    break
                pts, rad = _local_search(best_points, time_limit=min(15.0, remaining - 5), start_time=time.time())
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts
                else:
                    break
    
    if best_points is None:
        # Fallback: Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure all points are unit vectors
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
