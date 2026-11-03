"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 10000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r


def _fast_facets(points):
    """Compute facet circumcenters and radii efficiently."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    centers = []
    radii = []
    simplices_valid = []
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        centers.append(v)
        radii.append(r)
        simplices_valid.append(simplex)
    
    if not centers:
        return None, None, None
    
    return np.array(centers), np.array(radii), simplices_valid


def _fibonacci_points(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based_points(n):
    """Generate points based on icosahedron vertices expanded to n points."""
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    rng = np.random.RandomState(123)
    points = []
    for i in range(n):
        # Start from icosahedron vertex closest to random direction
        random_dir = rng.randn(3)
        random_dir /= np.linalg.norm(random_dir)
        dots = ico @ random_dir
        closest = ico[np.argmax(dots)]
        
        # Add small perturbation
        noise = rng.randn(3) * 0.3
        p = closest + noise
        p /= np.linalg.norm(p)
        points.append(p)
    
    return np.array(points)


def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _lloyd_iteration(points, max_iters=2000, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm: move each point to the centroid of its Voronoi cell."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _soft_max_objective_and_grad(x, temperature=0.1):
    """
    Compute soft-max approximation of covering radius and its gradient.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    centers, radii, simplices_valid = _fast_facets(pts)
    if centers is None:
        tree = cKDTree(pts)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        max_d = np.max(dists)
        max_r = 2.0 * np.arcsin(np.clip(max_d / 2.0, 0, 1))
        return max_r, np.zeros_like(x)
    
    T = temperature
    scaled = T * radii
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    obj = log_sum_exp / T
    
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pts = np.zeros_like(pts)
    for v, simplex, w in zip(centers, simplices_valid, weights):
        for idx in simplex:
            p = pts[idx]
            dot_vp = np.dot(v, p)
            sin_r = np.sqrt(np.maximum(0, 1.0 - dot_vp * dot_vp))
            if sin_r < 1e-12:
                continue
            dr_dp = -v / sin_r
            dr_dp_tangent = dr_dp - np.dot(dr_dp, p) * p
            grad_pts[idx] += w * dr_dp_tangent
    
    return obj, grad_pts.flatten()


def _riesz_energy_and_grad(x, s=2.0):
    """
    Compute Riesz s-energy and its gradient.
    Riesz energy: sum_{i!=j} |p_i - p_j|^(-s)
    Lower energy = more uniform distribution.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # Avoid division by zero
    
    dist = np.sqrt(np.maximum(dist2, 1e-15))
    energy = np.sum(dist ** (-s))
    
    # Gradient: d/dp_i sum_{j!=i} |p_i - p_j|^(-s) = -s * sum_{j!=i} (p_i - p_j) / |p_i - p_j|^(s+2)
    # But we need to project onto tangent plane
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = dist[i, j]
            if d < 1e-12:
                continue
            # d/dp_i (1/d^s) = -s * (p_i - p_j) / d^(s+2)
            grad[i] += -s * (pts[i] - pts[j]) / (d ** (s + 2))
        # Project onto tangent plane
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad.flatten()


def _gram_matrix_refine(points, time_limit=None, start_time=None, max_iters=100):
    """
    Refine using Gram-matrix based optimization.
    Maximizes minimum pairwise angular distance via projected gradient ascent.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    lr = 0.01
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute Gram matrix
        G = points @ points.T
        
        # Compute pairwise angular distances
        cos_angles = np.clip(G, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        # Find the minimum off-diagonal angle (worst pair)
        mask = ~np.eye(N, dtype=bool)
        min_angle = np.min(angles[mask])
        
        # Find all pairs with angle close to minimum (within tolerance)
        tol = 0.05
        close_pairs = np.argwhere(
            (angles < min_angle + tol) & mask
        )
        
        if len(close_pairs) == 0:
            break
        
        # Compute gradient: push apart the closest pairs
        grad = np.zeros_like(points)
        for i, j in close_pairs:
            if i == j:
                continue
            # Gradient of angle w.r.t. point i: -p_j / sin(angle)
            # Gradient of angle w.r.t. point j: -p_i / sin(angle)
            angle_ij = angles[i, j]
            sin_a = np.sin(angle_ij)
            if sin_a < 1e-12:
                continue
            # To maximize angle, we want to decrease cos(angle)
            # d(cos)/dp_i = p_j, d(cos)/dp_j = p_i
            # d(angle)/d(cos) = -1/sin(angle)
            # So d(angle)/dp_i = -p_j/sin(angle), d(angle)/dp_j = -p_i/sin(angle)
            # Project onto tangent planes
            grad_i = -points[j] / sin_a
            grad_i -= np.dot(grad_i, points[i]) * points[i]
            grad[i] += grad_i
            
            grad_j = -points[i] / sin_a
            grad_j -= np.dot(grad_j, points[j]) * points[j]
            grad[j] += grad_j
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm[grad_norm < 1e-15] = 1.0
        grad = grad / grad_norm
        
        # Update
        points = points + lr * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _lbfgs_refine(points, time_limit=None, start_time=None, temperature=0.1):
    """Refine points using L-BFGS-B with soft-max objective."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _soft_max_objective_and_grad(x, temperature)[0],
            x0,
            jac=lambda x: _soft_max_objective_and_grad(x, temperature)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and result.fun < best_cr:
            new_pts = result.x.reshape(N, 3)
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_cr


def _lbfgs_riesz(points, time_limit=None, start_time=None, s=2.0):
    """Refine points using L-BFGS-B with Riesz energy objective."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    x0 = points.flatten()
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    try:
        result = minimize(
            lambda x: _riesz_energy_and_grad(x, s)[0],
            x0,
            jac=lambda x: _riesz_energy_and_grad(x, s)[1],
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success:
            new_pts = result.x.reshape(N, 3)
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_points = new_pts
    except Exception:
        pass
    
    return best_points, best_cr


def _perturb_and_refine(points, n_perturbs=10, time_limit=None, start_time=None):
    """Perturb random points and refine with local optimization."""
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(42)
    
    for _ in range(n_perturbs):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 5)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * 0.08
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        refined, cr1 = _lloyd_iteration(perturbed, max_iters=500, time_limit=5, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        refined, cr2 = _lbfgs_refine(refined, time_limit=5, start_time=start_time, temperature=0.1)
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr


def _simulated_annealing(points, time_limit=None, start_time=None, T0=0.15, T1=0.01, n_iter=2000):
    """Simulated annealing to escape local minima."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Annealing schedule
        alpha = it / max(n_iter - 1, 1)
        T = T0 * (T1 / T0) ** alpha
        
        # Perturb random points
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        cr_new = _covering_radius(perturbed)
        
        # Accept if better or with probability exp(-(cr_new - current_cr) / T)
        if cr_new < current_cr:
            points = perturbed
            current_cr = cr_new
            if cr_new < best_cr:
                best_cr = cr_new
                best_points = points.copy()
        else:
            accept_prob = np.exp(-(cr_new - current_cr) / max(T, 1e-10))
            if rng.random() < accept_prob:
                points = perturbed
                current_cr = cr_new
    
    return best_points, best_cr


def _multi_start_refine(start_points_list, time_limit=None, start_time=None):
    """Try multiple starting points and keep the best result."""
    best_points = None
    best_cr = np.inf
    
    for sp in start_points_list:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Lloyd relaxation
        refined, cr1 = _lloyd_iteration(sp, max_iters=1000, time_limit=10, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # L-BFGS refinement
        refined, cr2 = _lbfgs_refine(refined, time_limit=10, start_time=start_time, temperature=0.1)
        
        # Gram matrix refinement
        refined, cr3 = _gram_matrix_refine(refined, time_limit=5, start_time=start_time)
        
        if cr3 < best_cr:
            best_cr = cr3
            best_points = refined.copy()
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate diverse starting points
    starts = [
        _fibonacci_points(N, offset=0.5),
        _fibonacci_points(N, offset=0.0),
        _fibonacci_points(N, offset=1.0),
        _icosahedron_based_points(N),
        _random_points(N, seed=42),
        _random_points(N, seed=123),
    ]
    
    # Phase 1: Multi-start with Lloyd + L-BFGS + Gram refinement
    remaining = deadline - time.time()
    if remaining > 30:
        best_points, best_cr = _multi_start_refine(starts, time_limit=remaining * 0.35, start_time=start_time)
        if best_points is None:
            best_points = _fibonacci_points(N)
            best_cr = _covering_radius(best_points)
    else:
        best_points = _fibonacci_points(N)
        best_cr = _covering_radius(best_points)
    
    # Phase 2: Riesz energy refinement to improve uniformity
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _lbfgs_riesz(best_points, time_limit=remaining * 0.2, start_time=start_time, s=2.0)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Perturb and refine from best solution
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _perturb_and_refine(best_points, n_perturbs=12, time_limit=remaining * 0.3, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 4: Simulated annealing to escape local minima
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _simulated_annealing(best_points, time_limit=remaining * 0.25, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 5: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 10:
        # Tighter Lloyd
        points, cr = _lloyd_iteration(best_points, max_iters=3000, tol=1e-12, time_limit=remaining * 0.15, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        
        remaining = deadline - time.time()
        if remaining > 5:
            # Very low temperature L-BFGS
            points, cr = _lbfgs_refine(best_points, time_limit=remaining * 0.25, start_time=start_time, temperature=0.02)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        
        remaining = deadline - time.time()
        if remaining > 3:
            # Final Gram refinement
            points, cr = _gram_matrix_refine(best_points, time_limit=remaining * 0.5, start_time=start_time, max_iters=200)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
