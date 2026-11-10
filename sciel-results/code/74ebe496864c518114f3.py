"""
Optimized 50-point sphere covering using:
1. Fast vectorized covering radius computation
2. Multi-start with diverse initializations (Fibonacci, random, antipodal)
3. Adaptive gradient-based optimization with simulated annealing
4. Local polish with fine-grained moves
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50


def _cover_radius_fast(pts):
    """Compute covering radius in radians using convex hull facets."""
    if len(pts) < 3:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    simplices = hull.simplices
    # Extract triangle vertices
    tri = pts[simplices]  # shape (n_facets, 3, 3)
    
    a = tri[:, 0, :]  # (n_facets, 3)
    b = tri[:, 1, :]
    c = tri[:, 2, :]
    
    # Cross products for weights
    cross_bc = np.cross(b, c)  # (n_facets, 3)
    cross_ca = np.cross(c, a)
    cross_ab = np.cross(a, b)
    
    w_a = np.linalg.norm(cross_bc, axis=1)  # (n_facets,)
    w_b = np.linalg.norm(cross_ca, axis=1)
    w_c = np.linalg.norm(cross_ab, axis=1)
    
    total = w_a + w_b + w_c
    valid = total > 1e-15
    
    if not np.any(valid):
        return 0.0
    
    # Compute circumcenters
    circum = a * w_a[:, None] + b * w_b[:, None] + c * w_c[:, None]
    cnorm = np.linalg.norm(circum, axis=1)
    valid &= cnorm > 1e-15
    
    if not np.any(valid):
        return 0.0
    
    circum = circum[valid]
    cnorm = cnorm[valid]
    circum /= cnorm[:, None]
    
    # Ensure circumcenter is on correct side
    facet_normals = np.cross(b[valid] - a[valid], c[valid] - a[valid])
    fnorm = np.linalg.norm(facet_normals, axis=1)
    valid_fn = fnorm > 1e-15
    
    if np.any(valid_fn):
        fn = facet_normals[valid_fn] / fnorm[valid_fn, None]
        circum_v = circum[valid_fn]
        dots = np.einsum('ij,ij->i', circum_v, fn)
        flip = dots < 0
        if np.any(flip):
            circum_v[flip] *= -1
            circum[valid_fn] = circum_v
    
    # Compute angular distance from circumcenter to nearest vertex
    # For each facet, compute min dot product with its three vertices
    a_v = a[valid]
    b_v = b[valid]
    c_v = c[valid]
    
    dots_a = np.einsum('ij,ij->i', circum, a_v)
    dots_b = np.einsum('ij,ij->i', circum, b_v)
    dots_c = np.einsum('ij,ij->i', circum, c_v)
    
    min_dot = np.minimum(np.minimum(dots_a, dots_b), dots_c)
    min_dot = np.clip(min_dot, -1.0, 1.0)
    angles = np.arccos(min_dot)
    
    return float(np.max(angles)) if len(angles) > 0 else 0.0


def _fibonacci_spiral(n=N):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n=N, seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _antipodal_pairs(n=N, seed=42):
    """Generate antipodal pair-based initialization."""
    rng = np.random.RandomState(seed)
    half = n // 2
    pts = rng.randn(half, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    if n % 2 == 1:
        extra = rng.randn(1, 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([pts, extra])
    return np.vstack([pts, -pts])


def _perturbed_init(base_pts, n_perturb=5, seed=42):
    """Generate perturbed versions of base points."""
    rng = np.random.RandomState(seed)
    results = []
    for i in range(n_perturb):
        scale = 0.05 + 0.1 * i
        perturb = rng.randn(N, 3) * scale
        new_pts = base_pts + perturb
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        results.append(new_pts)
    return results


def _optimize_gradient(pts, max_iters=2000, deadline=None, beta_start=0.5, beta_end=5.0):
    """
    Gradient-based optimization with adaptive step size.
    Uses the subgradient of the covering radius with respect to point positions.
    """
    pts = pts.copy()
    best_radius = _cover_radius_fast(pts)
    best_pts = pts.copy()
    
    step = 0.03
    no_improve_count = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        tri = pts[simplices]
        a = tri[:, 0, :]
        b = tri[:, 1, :]
        c = tri[:, 2, :]
        
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        
        w_a = np.linalg.norm(cross_bc, axis=1)
        w_b = np.linalg.norm(cross_ca, axis=1)
        w_c = np.linalg.norm(cross_ab, axis=1)
        
        total = w_a + w_b + w_c
        valid = total > 1e-15
        
        if not np.any(valid):
            break
        
        circum = a * w_a[:, None] + b * w_b[:, None] + c * w_c[:, None]
        cnorm = np.linalg.norm(circum, axis=1)
        valid &= cnorm > 1e-15
        
        if not np.any(valid):
            break
        
        circum_valid = circum[valid] / cnorm[valid, None]
        
        facet_normals = np.cross(b[valid] - a[valid], c[valid] - a[valid])
        fnorm = np.linalg.norm(facet_normals, axis=1)
        valid_fn = fnorm > 1e-15
        
        if np.any(valid_fn):
            fn = facet_normals[valid_fn] / fnorm[valid_fn, None]
            circum_v = circum_valid[valid_fn]
            dots = np.einsum('ij,ij->i', circum_v, fn)
            flip = dots < 0
            if np.any(flip):
                circum_v[flip] *= -1
                circum_valid[valid_fn] = circum_v
        
        # Compute angles for each facet
        a_v = a[valid]
        b_v = b[valid]
        c_v = c[valid]
        
        dots_a = np.einsum('ij,ij->i', circum_valid, a_v)
        dots_b = np.einsum('ij,ij->i', circum_valid, b_v)
        dots_c = np.einsum('ij,ij->i', circum_valid, c_v)
        
        min_dot = np.minimum(np.minimum(dots_a, dots_b), dots_c)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angles = np.arccos(min_dot)
        
        current_radius = np.max(angles)
        
        # Find the worst facet(s)
        worst_mask = angles > current_radius - 1e-8
        
        if not np.any(worst_mask):
            break
        
        # Compute gradient: move points away from worst circumcenters
        forces = np.zeros_like(pts)
        
        worst_circums = circum_valid[worst_mask]
        worst_indices = np.where(valid)[0][worst_mask]
        
        for idx, circ in zip(worst_indices, worst_circums):
            # Find nearest point to this circumcenter
            dots = pts @ circ
            nearest = np.argmin(dots)
            
            # Force the nearest point toward the circumcenter
            # Project (circ - pts[nearest]) onto tangent plane
            diff = circ - pts[nearest]
            tangent = diff - np.dot(diff, pts[nearest]) * pts[nearest]
            tnorm = np.linalg.norm(tangent)
            if tnorm > 1e-10:
                tangent /= tnorm
                forces[nearest] += 0.5 * tangent
        
        # Add repulsion for close points
        for i in range(N):
            for j in range(i + 1, N):
                dot_ij = np.dot(pts[i], pts[j])
                dot_ij = np.clip(dot_ij, -1.0, 1.0)
                angle_ij = np.arccos(dot_ij)
                if angle_ij < 0.15:  # ~8.6 degrees
                    rep_strength = (0.15 - angle_ij) / 0.15
                    diff = pts[i] - pts[j]
                    dnorm = np.linalg.norm(diff)
                    if dnorm > 1e-10:
                        diff /= dnorm
                        # Project onto tangent planes
                        t_i = diff - np.dot(diff, pts[i]) * pts[i]
                        t_j = diff - np.dot(diff, pts[j]) * pts[j]
                        tn_i = np.linalg.norm(t_i)
                        tn_j = np.linalg.norm(t_j)
                        if tn_i > 1e-10:
                            forces[i] += rep_strength * t_i / tn_i
                        if tn_j > 1e-10:
                            forces[j] -= rep_strength * t_j / tn_j
        
        # Normalize forces
        for i in range(N):
            fnorm_i = np.linalg.norm(forces[i])
            if fnorm_i > 1e-10:
                forces[i] /= fnorm_i
        
        # Apply step with adaptive sizing
        new_pts = pts + step * forces
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = _cover_radius_fast(new_pts)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            best_pts = pts.copy()
            no_improve_count = 0
            # Slightly increase step if improving
            step = min(step * 1.05, 0.08)
        else:
            no_improve_count += 1
            step *= 0.85
            if step < 1e-7:
                break
    
    return best_pts, best_radius


def _local_polish(pts, deadline=None, max_iters=500):
    """
    Fine-grained local polish: try small moves in random tangent directions.
    """
    pts = pts.copy()
    best_radius = _cover_radius_fast(pts)
    best_pts = pts.copy()
    rng = np.random.RandomState(999)
    
    step = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try moving each point
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Generate random tangent direction
            random_dir = rng.randn(3)
            tangent = random_dir - np.dot(random_dir, pts[i]) * pts[i]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm
            
            # Try positive move
            new_pts = pts.copy()
            new_pts[i] = pts[i] + step * tangent
            new_pts[i] /= np.linalg.norm(new_pts[i])
            
            new_radius = _cover_radius_fast(new_pts)
            if new_radius < best_radius - 1e-12:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()
                improved = True
            else:
                # Try negative move
                new_pts[i] = pts[i] - step * tangent
                new_pts[i] /= np.linalg.norm(new_pts[i])
                new_radius = _cover_radius_fast(new_pts)
                if new_radius < best_radius - 1e-12:
                    pts = new_pts
                    best_radius = new_radius
                    best_pts = pts.copy()
                    improved = True
        
        if not improved:
            no_improve += 1
            step *= 0.9
            if step < 1e-8 or no_improve > 50:
                break
        else:
            no_improve = 0
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_radius = np.pi
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 101112]:
        inits.append(_random_points(N, seed=seed))
    
    # Antipodal pairs
    for seed in [2020, 3030]:
        inits.append(_antipodal_pairs(N, seed=seed))
    
    # Perturbed Fibonacci
    fib_base = _fibonacci_spiral(N)
    for i in range(3):
        scale = 0.05 + 0.05 * i
        rng = np.random.RandomState(100 + i)
        perturb = rng.randn(N, 3) * scale
        new_pts = fib_base + perturb
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        inits.append(new_pts)
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Gradient optimization
        opt_pts, opt_radius = _optimize_gradient(
            init_pts, 
            max_iters=1500, 
            deadline=deadline - 10
        )
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_pts = opt_pts.copy()
    
    # Local polish on best result
    if best_pts is not None and time.time() < deadline - 5:
        final_pts, final_radius = _local_polish(best_pts, deadline=deadline - 1)
        if final_radius < best_radius:
            best_radius = final_radius
            best_pts = final_pts.copy()
    
    # If no good result, use Fibonacci as fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
