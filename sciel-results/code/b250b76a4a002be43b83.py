"""Spectral + repulsion optimization for 130-point sphere covering."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130
DEADLINE_MARGIN = 15.0  # seconds before the 150s kill


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facets."""
    try:
        hull = ConvexHull(pts)
        # For each facet, find the circumcenter and check it's on the sphere
        # The covering radius is the max over facets of the angular distance
        # from the facet's circumcenter (pushed to sphere) to the nearest point
        max_ang = 0.0
        for simplex in hull.simplices:
            # Get the 3 (or 4) points of this facet
            facet_pts = pts[simplex]
            # Compute circumcenter of the facet (in 3D, for a triangle it's the
            # point equidistant from all vertices; for tetrahedron, centroid-ish)
            # For a triangle on the sphere, the circumcenter of the spherical triangle
            # is the normalized cross product of two edge vectors... 
            # Actually, for a facet of the convex hull inscribed in the unit sphere,
            # the circumradius of the facet (as a flat polygon) gives the angular cap.
            # The angular distance from the circumcenter (on sphere) to each vertex
            # is arccos(1/sqrt(1 + R^2)) where R is the flat circumradius... 
            # Simpler: the circumcenter of the flat facet, normalized to unit sphere,
            # has angular distance arccos(dot(circumcenter_norm, vertex)) to each vertex.
            
            if len(simplex) == 3:
                a, b, c = facet_pts
                # Circumcenter of triangle abc
                ab = b - a
                ac = c - a
                bc = c - b
                # Using the formula: circumcenter = a + cross(ab, ac) * something
                # Actually use the standard formula
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) -
                           a[1] * (b[0] * c[2] - c[0] * b[2]) +
                           a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-15:
                    continue
                ux = ((b @ b) * (c[2] * a[1] - a[2] * c[1]) -
                      (c @ c) * (b[2] * a[1] - a[2] * b[1])) / d
                uy = ((c @ c) * (b[0] * a[2] - a[0] * c[2]) -
                      (b @ b) * (c[0] * a[2] - a[0] * b[2])) / d
                uz = ((a @ a) * (c[0] * b[1] - b[0] * c[1]) -
                      (b @ b) * (c[0] * a[1] - a[0] * c[1])) / d
                cc = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(cc)
                if cc_norm < 1e-12:
                    continue
                cc_unit = cc / cc_norm
                # Angular distance from cc_unit to each vertex
                for v in facet_pts:
                    dot = np.dot(cc_unit, v)
                    dot = np.clip(dot, -1.0, 1.0)
                    ang = np.arccos(dot)
                    if ang > max_ang:
                        max_ang = ang
            else:
                # For tetrahedral facets (4 points), use centroid as approximation
                centroid = np.mean(facet_pts, axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm < 1e-12:
                    continue
                cc_unit = centroid / c_norm
                for v in facet_pts:
                    dot = np.dot(cc_unit, v)
                    dot = np.clip(dot, -1.0, 1.0)
                    ang = np.arccos(dot)
                    if ang > max_ang:
                        max_ang = ang
        return np.degrees(max_ang)
    except Exception:
        # Fallback: brute force on a grid
        lat = np.linspace(0, np.pi, 180)
        lon = np.linspace(0, 2 * np.pi, 360)
        max_d = 0.0
        for la in lat:
            for lo in lon:
                y = np.array([np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)])
                dists = np.arccos(np.clip(pts @ y, -1, 1))
                if dists.min() > max_d:
                    max_d = dists.min()
        return np.degrees(max_d)


def _repulsion_step(pts, step_size, iterations=50):
    """Perform repulsion-based local optimization."""
    pts = _normalize(pts).copy()
    for it in range(iterations):
        # Compute repulsive forces
        # For each pair, push apart if too close
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        mask = dists > 1e-10
        dists_safe = np.where(mask, dists, 1.0)
        
        # Repulsive force: proportional to 1/d^2 along the direction
        # We want to maximize minimum distance, so push points apart
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Push i away from j with strength ~ 1/d^2
                strength = 1.0 / (d * d)
                forces[i] += strength * (pts[i] - pts[j]) / d
        
        # Also add a weak centering force to keep points on sphere
        # (normalization handles this)
        
        pts += step_size * forces
        pts = _normalize(pts)
    
    return pts


def _spectral_init(rng, n_starts=8):
    """Generate initial configurations using spectral/eigenvalue approach."""
    configs = []
    for s in range(n_starts):
        # Start from a perturbed Fibonacci spiral
        i = np.arange(N) + 0.5 + rng.uniform(-0.3, 0.3, N)
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = _normalize(pts)
        
        # Add random perturbation
        pts += rng.normal(0, 0.05, (N, 3))
        pts = _normalize(pts)
        
        configs.append(pts)
    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate multiple initial configurations
    configs = _spectral_init(rng, n_starts=8)
    
    best_pts = None
    best_radius = np.inf
    
    # Phase 2: Repulsion optimization on each config
    for idx, pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Multiple repulsion passes with decreasing step size
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        iters_per_step = [50, 50, 50, 40, 40, 30, 30, 20]
        
        for ss, it in zip(step_sizes, iters_per_step):
            if time.time() > deadline:
                break
            pts = _repulsion_step(pts, ss, it)
        
        # Evaluate
        radius = _covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()
    
    # Phase 3: Final polish with very small steps
    if best_pts is not None and time.time() < deadline:
        for ss in [0.0005, 0.0002, 0.0001]:
            if time.time() > deadline:
                break
            best_pts = _repulsion_step(best_pts, ss, 30)
        final_radius = _covering_radius(best_pts)
        if final_radius < best_radius:
            best_radius = final_radius
    
    if best_pts is None:
        # Fallback: Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return _normalize(best_pts)
