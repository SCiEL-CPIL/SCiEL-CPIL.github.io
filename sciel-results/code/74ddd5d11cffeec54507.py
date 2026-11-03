"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

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


def _spherical_lloyd_iteration(points, max_iters=500, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
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


def _potential_energy_minimize(points, time_limit=None, start_time=None):
    """
    Formulate the problem as a spherical potential minimization using a repulsive
    inverse-power law combined with an attractive force toward the sphere center.
    Solved via a symplectic integrator with velocity Verlet integration.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Velocity Verlet parameters
    dt = 0.01
    gamma = 0.99  # Damping factor
    k_rep = 100.0  # Repulsive force constant
    k_att = 50.0   # Attractive force constant (toward center)
    
    # Initialize velocities to zero
    velocities = np.zeros_like(points)
    
    max_iters = 5000
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Repulsive force: inverse square law (1/r^2) for stronger effect at close range
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:
                    # Force magnitude: k_rep / dist^2
                    force_mag = k_rep / (dist * dist)
                    forces[i] += force_mag * (diff / dist)
            
            # Attractive force toward center (radial)
            # This keeps points on the sphere
            forces[i] -= k_att * points[i]
        
        # Velocity Verlet integration
        # Update velocities
        velocities = velocities * gamma + forces * dt
        
        # Update positions
        new_points = points + velocities * dt
        
        # Normalize to sphere
        new_norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_norms[new_norms < 1e-15] = 1.0
        new_points = new_points / new_norms
        
        # Check for improvement
        cr = _covering_radius(new_points)
        if cr < best_cr:
            best_cr = cr
            best_points = new_points.copy()
        
        # Update points and velocities
        points = new_points
        
        # Recompute forces for next step (already done in next iteration)
    
    return best_points, best_cr


def _gradient_descent(points, max_iters=2000, lr_init=0.005, lr_min=1e-8, time_limit=None, start_time=None):
    """Gradient descent to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
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
        
        facet_data = []
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
            facet_data.append((r, simplex, v))
        
        if not facet_data:
            break
        
        facet_data.sort(key=lambda x: -x[0])
        top_k = min(5, len(facet_data))
        
        step = np.zeros_like(points)
        for r, simplex, center in facet_data[:top_k]:
            weight = max(0.0, r - best_cr + 0.01)
            for idx in simplex:
                p = points[idx]
                d = center - p
                d_tangent = d - np.dot(d, p) * p
                step[idx] += weight * d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = points + lr * step
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.999, lr_min)
        else:
            lr = max(lr * 0.92, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr


def _simulated_annealing(points, time_limit=None, start_time=None, seed=42):
    """Simulated annealing to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    rng = np.random.RandomState(seed)
    T = 0.05
    cooling = 0.995
    T_min = 1e-6
    
    current_cr = best_cr
    current_points = points.copy()
    
    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = current_points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = current_points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = perturbed.copy()
        
        T *= cooling
    
    return best_points, best_cr


def _nelder_mead_polish(points, time_limit=None, start_time=None):
    """High-precision polish using Nelder-Mead on individual coordinates."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Coordinate-wise Nelder-Mead for high precision
    from scipy.optimize import minimize
    
    def _objective(x_flat):
        P = x_flat.reshape(N, 3)
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        return _covering_radius(P)
    
    # Try a few random perturbations and polish the best
    rng = np.random.RandomState(999)
    for trial in range(3):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Small perturbation
        perturbed = points.copy()
        mask = rng.rand(N) < 0.3
        perturbed[mask] += rng.randn(np.sum(mask), 3) * 0.01
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        try:
            result = minimize(
                _objective,
                perturbed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-14}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_cr = _covering_radius(new_points)
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
        except Exception:
            continue
    
    return best_points, best_cr


def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd + Potential Minimization + Gradient Descent + NM Polish."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    # Stage 1: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.2, 6)
    else:
        lloyd_time = 6
    
    points, cr1 = _spherical_lloyd_iteration(points, max_iters=200, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 2: Potential Energy Minimization (NEW)
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        pot_time = min(remaining * 0.3, 12)
    else:
        pot_time = 12
    
    points, cr2 = _potential_energy_minimize(points, time_limit=pot_time, start_time=start_time)
    
    # Stage 3: Gradient Descent Polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr2
        gd_time = min(remaining * 0.25, 8)
    else:
        gd_time = 8
    
    points, cr3 = _gradient_descent(points, max_iters=300, lr_init=0.003, time_limit=gd_time, start_time=start_time)
    
    # Stage 4: Nelder-Mead High-Precision Polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr3
        nm_time = min(remaining * 0.25, 10)
    else:
        nm_time = 10
    
    points, cr4 = _nelder_mead_polish(points, time_limit=nm_time, start_time=start_time)
    
    return points, cr4


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
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
    
    # Phase 2: Simulated Annealing from Best Configuration
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.3, 20)
        sa_points, sa_cr = _simulated_annealing(best_points, time_limit=sa_time, start_time=start_time, seed=123)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Phase 3: Final Refinement with Full Pipeline
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.7, 20)
        final_points, final_cr = _refine(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize Final Result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
