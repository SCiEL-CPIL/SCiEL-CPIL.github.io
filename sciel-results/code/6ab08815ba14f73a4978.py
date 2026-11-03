"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
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

def _repulsive_energy(points, power=2.0):
    """
    Calculate the repulsive energy of the configuration.
    Vectorized for speed.
    """
    points = _normalize(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    # Upper triangle only
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist_sq[mask]
    d = np.maximum(d, 1e-10)
    energy = np.sum(1.0 / (d ** (power / 2.0)))
    return energy

def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]

def _lloyd_iteration(points, max_iters=500, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = _normalize(points).copy()
    
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

def _sa_optimize(points, T_start=0.5, T_end=1e-5, n_iter=10000, time_limit=None, start_time=None):
    """
    Simulated Annealing to minimize repulsive energy.
    """
    points = _normalize(points).copy()
    current_energy = _repulsive_energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Cooling schedule
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        T = T_start * (cooling ** i)
        
        # Select a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a new position by adding random noise and normalizing
        # Step size decreases as T decreases
        step_size = 0.1 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        # Update points
        points[idx] = new_point
        
        new_energy = _repulsive_energy(points)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            # Reject move
            points[idx] = original_point
            
    return best_points

def _worst_facet_projection(points, max_iters=500, lr_init=0.02, lr_min=1e-6, time_limit=None, start_time=None):
    """
    Geometric worst-facet projection method.
    Iteratively identifies the convex hull facet with the largest spherical circumradius
    and projects its three vertices along their negative spherical gradients to shrink that facet.
    """
    points = _normalize(points).copy()
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the worst facet (largest circumradius)
        worst_r = -1.0
        worst_simplex = None
        worst_center = None
        
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
            if r > worst_r:
                worst_r = r
                worst_simplex = simplex
                worst_center = v
        
        if worst_simplex is None:
            break
        
        # Project the three vertices of the worst facet toward the center
        step = np.zeros_like(points)
        for idx in worst_simplex:
            p = points[idx]
            d = worst_center - p
            d_tangent = d - np.dot(d, p) * p
            step[idx] += d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = points + lr * step
        new_points = _normalize(new_points)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.998, lr_min)
        else:
            lr = max(lr * 0.9, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: SA on repulsive energy + Lloyd + worst-facet projection."""
    # Stage 1: Simulated Annealing on repulsive energy (smooth surrogate)
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        sa_time = min(remaining * 0.35, 12)
    else:
        sa_time = 12
    
    # Use SA to explore the space using the smooth surrogate
    points_sa = _sa_optimize(points, T_start=0.3, T_end=1e-5, n_iter=5000, time_limit=sa_time, start_time=start_time)
    
    # Check if SA improved the covering radius
    cr_sa = _covering_radius(points_sa)
    cr_orig = _covering_radius(points)
    if cr_sa < cr_orig:
        points = points_sa
        current_cr = cr_sa
    else:
        current_cr = cr_orig

    # Stage 2: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, current_cr
        lloyd_time = min(remaining * 0.25, 8)
    else:
        lloyd_time = 8
    
    points, cr1 = _lloyd_iteration(points, max_iters=300, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 3: Worst-facet projection (geometric relaxation)
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr1
        wf_time = min(remaining * 0.6, 18)
    else:
        wf_time = 18
    
    points, cr2 = _worst_facet_projection(points, max_iters=800, lr_init=0.02, time_limit=wf_time, start_time=start_time)
    
    return points, cr2

def _jittered_fibonacci_points(n, seed=42, jitter_scale=0.3):
    """Generate Fibonacci spiral points with random jitter."""
    base = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    pts = base.copy()
    for i in range(n):
        noise = rng.randn(3) * jitter_scale
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=1, jitter_scale=0.2),
        _random_points(N, seed=1),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 25)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Final refinement with more time if available
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.8, 20)
        final_points, final_cr = _refine(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize final result
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
