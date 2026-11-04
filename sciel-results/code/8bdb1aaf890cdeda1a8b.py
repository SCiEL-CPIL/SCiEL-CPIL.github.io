"""Covering the sphere with 130 equal caps - strong algorithm."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull
import math

N = 130
DEADLINE = 140.0  # safety margin before 150s kill

def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return pts / n

def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Circumradius of triangle on unit sphere = max angular distance from triangle center
            # The covering radius for a facet is the angular distance from the sphere point
            # farthest from the triangle, which is the circumradius of the spherical triangle.
            # For a planar triangle with vertices on unit sphere, the circumradius of the
            # spherical triangle equals the angle subtended by the circumcenter.
            # Simpler: compute the circumcenter of the 3 points in 3D, project to sphere.
            a, b, c = tri[0], tri[1], tri[2]
            # Circumcenter of triangle in 3D
            d = 2.0 * (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))
            if abs(d) < 1e-12:
                continue
            sa = (b[0]**2+b[1]**2+b[2]**2)*(c[0]**2+c[1]**2+c[2]**2) - (b@c)**2
            sb = (a[0]**2+a[1]**2+a[2]**2)*(c[0]**2+c[1]**2+c[2]**2) - (a@c)**2
            sc = (a[0]**2+a[1]**2+a[2]**2)*(b[0]**2+b[1]**2+b[2]**2) - (a@b)**2
            cx = (sa*(b[0]*c[2]-b[2]*c[1]) - sb*(a[0]*c[2]-a[2]*c[0]) + sc*(a[0]*b[2]-a[2]*b[0])) / d
            cy = (sa*(c[0]*b[1]-c[1]*b[0]) - sb*(c[0]*a[1]-c[1]*a[0]) + sc*(b[0]*a[1]-b[1]*a[0])) / d
            cz = (sa*(c[1]*b[2]-c[2]*b[1]) - sb*(c[1]*a[2]-c[2]*a[1]) + sc*(b[1]*a[2]-b[2]*a[1])) / d
            circ = np.array([cx, cy, cz])
            cn = np.linalg.norm(circ)
            if cn < 1e-15:
                continue
            circ = circ / cn
            # Angular distance from circ to each vertex should be equal = circumradius
            # The covering radius for this facet is the angle from circ to the sphere
            # Actually, the farthest point on the sphere from the triangle is in the direction
            # of the circumcenter if it's on the same side, or opposite.
            # The spherical circumradius is the angle between circ and any vertex.
            ang = np.arccos(np.clip(np.abs(circ @ a), -1.0, 1.0))
            # But we need the max over the sphere of distance to nearest point in triangle.
            # This is the spherical circumradius.
            max_r = max(max_r, ang)
        return np.degrees(max_r)
    except Exception:
        # Fallback: brute force
        max_r = 0.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0))
                max_r = max(max_r, d)
        return np.degrees(max_r)

def _min_pairwise_angle(pts):
    """Minimum angular distance between any pair of points."""
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(dots, 1.0)
    angles = np.arccos(dots)
    return np.min(angles)

def _soft_max_min_angle(pts, beta=50.0):
    """Smooth surrogate: -soft_min of pairwise angles. Minimizing this maximizes the min angle."""
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(dots, 1.0)
    angles = np.arccos(dots)
    # soft min via log-sum-exp
    flat = angles.ravel()
    # Remove diagonal (self-distances)
    mask = ~np.eye(N, dtype=bool)
    flat = angles[mask]
    m = np.min(flat)
    sm = m + np.log(np.sum(np.exp(beta * (flat - m)))) / beta
    return -sm

def _soft_max_min_grad(pts, beta=50.0):
    """Gradient of -soft_min pairwise angle."""
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    np.fill_diagonal(dots, 1.0)
    angles = np.arccos(dots)
    mask = ~np.eye(N, dtype=bool)
    flat = angles[mask]
    m = np.min(flat)
    exps = np.exp(beta * (flat - m))
    weights = exps / np.sum(exps)
    # d(-sm)/d(angles) = -weights
    # d(angles)/d(dots) = -1/sqrt(1-dots^2)
    # d(dots)/d(pts) = pts_j for (i,j) pair
    grad = np.zeros_like(pts)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            idx = i * N + j
            w = weights[idx]
            d = dots[i, j]
            dsda = -1.0 / np.sqrt(max(1.0 - d * d, 1e-15))
            # d(-sm)/d(pts_i) += w * dsda * pts_j
            grad[i] += w * dsda * pts[j]
            # d(-sm)/d(pts_j) += w * dsda * pts_i
            grad[j] += w * dsda * pts[i]
    return grad

def _voronoi_centroid_refine(pts, iters=50, tol=1e-10):
    """Move each point to the centroid of its Voronoi cell on the sphere."""
    pts = _normalize(pts.copy())
    for it in range(iters):
        # Compute nearest neighbor for a grid of test points
        # Use a fine icosahedral subdivision or uniform grid
        n_test = 5000
        # Generate test points: use Fibonacci spiral with more points
        i = np.arange(n_test) + 0.5
        z = 1.0 - 2.0 * i / n_test
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Find nearest point for each test point
        dots = test_pts @ pts.T  # (n_test, N)
        nearest = np.argmax(dots, axis=1)
        
        # Compute centroid for each point
        sums = np.zeros((N, 3))
        counts = np.zeros(N)
        for k in range(N):
            mask = nearest == k
            if np.any(mask):
                sums[k] = np.sum(test_pts[mask], axis=0)
                counts[k] = np.sum(mask)
        
        new_pts = pts.copy()
        for k in range(N):
            if counts[k] > 10:
                c = sums[k]
                cn = np.linalg.norm(c)
                if cn > 1e-10:
                    new_pts[k] = c / cn
        
        new_pts = _normalize(new_pts)
        diff = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = new_pts
        if diff < tol:
            break
    return pts

def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    t0 = time.time()
    best_pts = None
    best_cr = np.inf
    
    # Multiple initializations
    inits = []
    # Fibonacci spiral
    inits.append(_fibonacci_spiral(N))
    # Jittered Fibonacci
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3]:
        rng_s = np.random.default_rng(seed)
        base = _fibonacci_spiral(N)
        # Add small random perturbation
        noise = rng_s.normal(0, 0.05, (N, 3))
        pts = _normalize(base + noise)
        inits.append(pts)
    # Random start
    inits.append(_normalize(rng.normal(0, 1, (N, 3))))
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Step 1: Gradient optimization on soft-max-min objective
        try:
            res = minimize(
                _soft_max_min_angle,
                init_pts,
                jac=_soft_max_min_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_pts = _normalize(res.x)
        except Exception:
            opt_pts = init_pts
        
        # Step 2: Voronoi centroid refinement
        if time.time() - t0 > DEADLINE - 15:
            break
        refined = _voronoi_centroid_refine(opt_pts, iters=30)
        
        # Step 3: Another gradient pass on refined
        try:
            res2 = minimize(
                _soft_max_min_angle,
                refined,
                jac=_soft_max_min_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            final_pts = _normalize(res2.x)
        except Exception:
            final_pts = refined
        
        # Final Voronoi polish
        if time.time() - t0 > DEADLINE - 10:
            break
        final_pts = _voronoi_centroid_refine(final_pts, iters=20)
        
        # Evaluate
        cr = _covering_radius(final_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = inits[0]
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    # Check no coincident points
    min_ang = _min_pairwise_angle(best_pts)
    if min_ang < 1e-6:
        # Add tiny perturbation
        rng = np.random.default_rng(999)
        best_pts = _normalize(best_pts + rng.normal(0, 1e-4, (N, 3)))
    
    return best_pts
