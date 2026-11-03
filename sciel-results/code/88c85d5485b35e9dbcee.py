"""Spherical Lloyd's algorithm with optimized facet circumradius computation and multi-strategy optimization."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_triangle_circumcenter(a, b, c):
    """Compute the spherical circumcenter of triangle (a,b,c) on unit sphere."""
    u = a - b
    v = a - c
    p_dir = np.cross(u, v)
    p_norm = np.linalg.norm(p_dir)
    
    if p_norm < 1e-12:
        centroid = a + b + c
        cn = np.linalg.norm(centroid)
        if cn < 1e-12:
            return None
        return centroid / cn
    
    p = p_dir / p_norm
    centroid = (a + b + c) / 3.0
    if np.dot(p, centroid) < 0:
        p = -p
    return p

def _facet_circumradius_fast(a, b, c):
    """Compute circumradius in radians using vectorized-friendly approach."""
    # The circumcenter is the point equidistant from a, b, c on the sphere
    # For a spherical triangle, the circumradius R satisfies:
    # cos(R) = (cos(c) + cos(a)cos(b)) / (sin(a)sin(b)) ... but we use geometric approach
    # 
    # Alternative: solve for p such that p·a = p·b = p·c = cos(R)
    # This means p is in the direction of a×(b-c) type construction
    # 
    # More direct: the circumcenter lies along the line through the centroid 
    # in the direction perpendicular to the plane of the triangle
    # 
    # Use the formula: circumcenter direction = (a×b×c) normalized won't work directly
    # 
    # Best approach: solve linear system
    # p · (a-b) = 0
    # p · (a-c) = 0
    # |p| = 1
    u = a - b
    v = a - c
    cross = np.cross(u, v)
    cn = np.linalg.norm(cross)
    
    if cn < 1e-14:
        # Degenerate - use centroid
        centroid = a + b + c
        c_norm = np.linalg.norm(centroid)
        if c_norm < 1e-14:
            return 0.0
        p = centroid / c_norm
    else:
        p = cross / cn
        # Ensure correct orientation
        centroid = (a + b + c) / 3.0
        if np.dot(p, centroid) < 0:
            p = -p
    
    dot_val = np.clip(np.dot(p, a), -1.0, 1.0)
    return np.arccos(dot_val)

def _compute_all_circumradii(points, hull):
    """Compute circumradii for all facets. Returns array of circumradii in radians."""
    simplices = hull.simplices
    n_facets = len(simplices)
    circumradii = np.zeros(n_facets)
    
    for i, simplex in enumerate(simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        circumradii[i] = _facet_circumradius_fast(a, b, c)
    
    return circumradii

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    circumradii = _compute_all_circumradii(points, hull)
    return np.degrees(np.max(circumradii))

def _spherical_lloyd_step(points, damping=0.3, alpha=2.0):
    """One Lloyd iteration: move each point to weighted centroid of its Voronoi cell.
    
    Weight each facet by (circumradius)^alpha to emphasize large-radius cells.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    n_points = len(points)
    simplices = hull.simplices
    circumradii = _compute_all_circumradii(points, hull)
    
    # For each point, collect its incident facets and compute weighted circumcenter
    point_targets = np.zeros_like(points)
    
    for i in range(n_points):
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for fi, simplex in enumerate(simplices):
            if i in simplex:
                cr = circumradii[fi]
                if cr > 1e-10:
                    weight = cr ** alpha
                    a = points[simplex[0]]
                    b = points[simplex[1]]
                    c = points[simplex[2]]
                    cc = _spherical_triangle_circumcenter(a, b, c)
                    if cc is not None:
                        weighted_sum += weight * cc
                        total_weight += weight
        
        if total_weight > 1e-14:
            target = weighted_sum / total_weight
            tn = np.linalg.norm(target)
            if tn > 1e-14:
                point_targets[i] = target / tn
            else:
                point_targets[i] = points[i]
        else:
            point_targets[i] = points[i]
    
    # Damped spherical interpolation
    new_points = np.zeros_like(points)
    for i in range(n_points):
        dot_val = np.clip(np.dot(points[i], point_targets[i]), -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        if angle < 1e-12:
            new_points[i] = points[i]
        else:
            move_angle = damping * angle
            sin_total = np.sin(angle)
            if sin_total < 1e-12:
                new_points[i] = points[i]
            else:
                coeff1 = np.cos(move_angle)
                coeff2 = np.sin(move_angle) / sin_total
                new_points[i] = coeff1 * points[i] + coeff2 * point_targets[i]
                nn = np.linalg.norm(new_points[i])
                if nn > 1e-12:
                    new_points[i] = new_points[i] / nn
                else:
                    new_points[i] = points[i]
    
    return new_points

def _aggressive_local_optimize(points, max_time=10.0):
    """Aggressively optimize the worst facet by moving its vertices toward circumcenter.
    
    Uses adaptive step size and tries multiple perturbations.
    """
    start_time = time.time()
    current = points.copy()
    current_radius = _compute_covering_radius(current)
    
    base_step = 0.1
    no_improve = 0
    best_local = current.copy()
    best_local_radius = current_radius
    
    while time.time() - start_time < max_time:
        if no_improve > 50:
            break
        
        try:
            hull = ConvexHull(current)
        except Exception:
            break
        
        circumradii = _compute_all_circumradii(current, hull)
        worst_idx = np.argmax(circumradii)
        worst_cr = circumradii[worst_idx]
        worst_simplex = hull.simplices[worst_idx]
        
        a = current[worst_simplex[0]]
        b = current[worst_simplex[1]]
        c = current[worst_simplex[2]]
        cc = _spherical_triangle_circumcenter(a, b, c)
        
        if cc is None:
            break
        
        # Try several step sizes
        improved = False
        for step_frac in [0.3, 0.5, 0.7, 1.0, 0.15, 0.25]:
            trial = current.copy()
            for idx in worst_simplex:
                dot_val = np.clip(np.dot(trial[idx], cc), -1.0, 1.0)
                angle = np.arccos(dot_val)
                if angle > 1e-12:
                    sin_angle = np.sin(angle)
                    if sin_angle > 1e-12:
                        move = step_frac * angle
                        coeff1 = np.cos(move)
                        coeff2 = np.sin(move) / sin_angle
                        new_pos = coeff1 * trial[idx] + coeff2 * cc
                        nn = np.linalg.norm(new_pos)
                        if nn > 1e-12:
                            trial[idx] = new_pos / nn
            
            # Check separation
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    if np.linalg.norm(trial[i] - trial[j]) < 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            
            if not ok:
                continue
            
            new_radius = _compute_covering_radius(trial)
            if new_radius < current_radius - 1e-9:
                current = trial
                current_radius = new_radius
                no_improve = 0
                improved = True
                if current_radius < best_local_radius:
                    best_local_radius = current_radius
                    best_local = current.copy()
                break
        
        if not improved:
            no_improve += 1
            # Try a small random perturbation of the worst facet vertices
            rng = np.random.default_rng()
            trial = current.copy()
            for idx in worst_simplex:
                perturbation = rng.normal(0, 0.01, 3)
                trial[idx] = trial[idx] + perturbation
                nn = np.linalg.norm(trial[idx])
                if nn > 1e-12:
                    trial[idx] = trial[idx] / nn
            
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    if np.linalg.norm(trial[i] - trial[j]) < 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok:
                new_radius = _compute_covering_radius(trial)
                if new_radius < current_radius - 1e-9:
                    current = trial
                    current_radius = new_radius
                    no_improve = 0
                    if current_radius < best_local_radius:
                        best_local_radius = current_radius
                        best_local = current.copy()
    
    return best_local

def _lloyd_relax(points, max_time=25.0):
    """Run Lloyd relaxation with adaptive damping and alpha."""
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_radius = _compute_covering_radius(best)
    
    # Phase 1: high alpha, moderate damping (focus on worst cells)
    # Phase 2: lower alpha, lower damping (fine-tune)
    phases = [
        (0.4, 3.0, 20),   # damping, alpha, iters
        (0.3, 2.5, 20),
        (0.2, 2.0, 20),
        (0.15, 1.5, 20),
        (0.1, 1.0, 20),
        (0.08, 0.8, 20),
        (0.05, 0.5, 20),
    ]
    
    for damping, alpha, iters in phases:
        if time.time() - start_time > max_time - 3.0:
            break
        
        for _ in range(iters):
            if time.time() - start_time > max_time - 3.0:
                break
            
            new_points = _spherical_lloyd_step(current, damping=damping, alpha=alpha)
            
            # Check separation
            ok = True
            for i in range(len(new_points)):
                for j in range(i + 1, len(new_points)):
                    if np.linalg.norm(new_points[i] - new_points[j]) < 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            
            if not ok:
                continue
            
            radius = _compute_covering_radius(new_points)
            
            if radius < best_radius:
                best_radius = radius
                best = new_points.copy()
            
            current = new_points
    
    return best, best_radius

def _fibonacci_initialization(n, offset=0.5):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _random_initialization(n, seed):
    """Generate n random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _so50_initialization():
    """Use a known good starting configuration close to the SO(50) result."""
    # Start from Fibonacci and do a quick Lloyd pass
    pts = _fibonacci_initialization(N, offset=0.5)
    # Quick 5-iteration Lloyd to get a decent start
    for _ in range(5):
        pts = _spherical_lloyd_step(pts, damping=0.3, alpha=2.0)
    return pts

def construct():
    """Construct 50 points on the unit sphere using spherical Lloyd's algorithm."""
    start_time = time.time()
    max_total_time = 115.0
    
    best_points = None
    best_radius = float('inf')
    
    # Strategy 1: Fibonacci + Lloyd
    if time.time() - start_time < max_total_time - 30.0:
        init = _fibonacci_initialization(N, offset=0.5)
        result, radius = _lloyd_relax(init, max_time=min(25.0, max_total_time - (time.time() - start_time) - 20.0))
        if radius < best_radius:
            best_radius = radius
            best_points = result.copy()
        
        remaining = max_total_time - (time.time() - start_time)
        if remaining > 10.0 and best_points is not None:
            local = _aggressive_local_optimize(best_points, max_time=min(12.0, remaining - 5.0))
            local_radius = _compute_covering_radius(local)
            if local_radius < best_radius:
                best_radius = local_radius
                best_points = local.copy()
    
    # Strategy 2: Different Fibonacci offset
    if time.time() - start_time < max_total_time - 25.0:
        init = _fibonacci_initialization(N, offset=0.3)
        result, radius = _lloyd_relax(init, max_time=min(20.0, max_total_time - (time.time() - start_time) - 15.0))
        if radius < best_radius:
            best_radius = radius
            best_points = result.copy()
        
        remaining = max_total_time - (time.time() - start_time)
        if remaining > 8.0 and best_points is not None:
            local = _aggressive_local_optimize(best_points, max_time=min(10.0, remaining - 4.0))
            local_radius = _compute_covering_radius(local)
            if local_radius < best_radius:
                best_radius = local_radius
                best_points = local.copy()
    
    # Strategy 3: Random initialization with good seed
    if time.time() - start_time < max_total_time - 20.0:
        init = _random_initialization(N, seed=42)
        result, radius = _lloyd_relax(init, max_time=min(18.0, max_total_time - (time.time() - start_time) - 12.0))
        if radius < best_radius:
            best_radius = radius
            best_points = result.copy()
        
        remaining = max_total_time - (time.time() - start_time)
        if remaining > 6.0 and best_points is not None:
            local = _aggressive_local_optimize(best_points, max_time=min(8.0, remaining - 3.0))
            local_radius = _compute_covering_radius(local)
            if local_radius < best_radius:
                best_radius = local_radius
                best_points = local.copy()
    
    # Strategy 4: Another random seed
    if time.time() - start_time < max_total_time - 15.0:
        init = _random_initialization(N, seed=123)
        result, radius = _lloyd_relax(init, max_time=min(15.0, max_total_time - (time.time() - start_time) - 10.0))
        if radius < best_radius:
            best_radius = radius
            best_points = result.copy()
        
        remaining = max_total_time - (time.time() - start_time)
        if remaining > 5.0 and best_points is not None:
            local = _aggressive_local_optimize(best_points, max_time=min(6.0, remaining - 2.0))
            local_radius = _compute_covering_radius(local)
            if local_radius < best_radius:
                best_radius = local_radius
                best_points = local.copy()
    
    # Final polish on the overall best
    remaining = max_total_time - (time.time() - start_time)
    if remaining > 5.0 and best_points is not None:
        final = _aggressive_local_optimize(best_points, max_time=min(10.0, remaining - 2.0))
        final_radius = _compute_covering_radius(final)
        if final_radius < best_radius:
            best_points = final
            best_radius = final_radius
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_initialization(N, offset=0.5)
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
