"""Rewritten program: sphere covering with 130 points using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 130

# Deadline guard: must return before 150s wall-clock; use 140s as safety margin
_DEADLINE = 140.0


def _covering_radius_deg(pts):
    """Compute covering radius in degrees. pts is (N,3) unit vectors."""
    # Covering radius = max over sphere of min angular distance to any point.
    # This equals the maximum circumradius of the Delaunay triangulation facets.
    # A practical computation: for each Voronoi cell vertex (intersection of bisector planes),
    # compute its angular distance to the nearest site.
    # 
    # Efficient approach: the covering radius is the max over all pairs/triples of points
    # of the circumradius of the spherical triangle. But a simpler and correct method:
    # The covering radius is the maximum, over all points on the sphere, of the distance
    # to the nearest site. The worst point is always a vertex of the Voronoi diagram on the sphere,
    # which corresponds to a circumcenter of a spherical triangle formed by 3 sites (or 2 for degenerate cases).
    #
    # For N=130, we can compute the convex hull of the 3D points, then for each facet (triangle),
    # find the circumcenter of that triangle (in 3D, it's the center of the circumscribed circle),
    # project to the unit sphere, and compute its angular distance to the nearest point.
    # Actually, the circumcenter of a facet of the convex hull of points on the unit sphere
    # is not directly the Voronoi vertex. Let me think more carefully.
    #
    # The Voronoi diagram on the sphere: each Voronoi cell is a spherical polygon. The vertices
    # of the Voronoi diagram are the circumcenters of the Delaunay triangles. The Delaunay triangulation
    # on the sphere is the dual of the Voronoi diagram. For points on the unit sphere, the Delaunay
    # triangulation can be obtained from the convex hull: each face of the convex hull corresponds
    # to a Delaunay triangle. The circumcenter of a Delaunay triangle (in 3D) projected to the sphere
    # gives a Voronoi vertex.
    #
    # So: compute convex hull, for each triangular face, compute the circumcenter of the three vertices,
    # normalize to unit sphere, compute angular distance to nearest site, take max.
    
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    max_rad = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Circumcenter of triangle abc in 3D
        # The circumcenter is equidistant from a, b, c.
        # Solve: |x-a|^2 = |x-b|^2 = |x-c|^2
        # This gives 2 linear equations in x.
        # d_ab = b - a, d_ac = c - a
        # 2*x.d_ab = |b|^2 - |a|^2
        # 2*x.d_ac = |c|^2 - |a|^2
        d_ab = b - a
        d_ac = c - a
        A = np.array([d_ab, d_ac])  # 2x3
        rhs = np.array([np.dot(b, b) - np.dot(a, a), np.dot(c, c) - np.dot(a, a)]) / 2.0
        # Solve least squares (the triangle is not degenerate on the sphere)
        try:
            x = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        # Normalize to unit sphere
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            continue
        x = x / norm
        # Angular distance to nearest site
        dots = pts @ x
        dots = np.clip(dots, -1.0, 1.0)
        min_angle = np.arccos(1.0 - np.min(1.0 - dots))  # This is wrong
        # Actually, angular distance = arccos(dot). Min over sites.
        angles = np.arccos(dots)
        min_ang = np.min(angles)
        max_rad = max(max_rad, min_ang)
    return np.degrees(max_rad)


def _covering_radius_fast(pts):
    """Faster covering radius using vectorized operations where possible."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    simplices = hull.simplices
    # Get all triangle vertices
    tri_pts = pts[simplices]  # (F, 3, 3)
    a = tri_pts[:, 0, :]  # (F, 3)
    b = tri_pts[:, 1, :]
    c = tri_pts[:, 2, :]
    
    d_ab = b - a  # (F, 3)
    d_ac = c - a
    # Build A matrix: (F, 2, 3)
    A = np.stack([d_ab, d_ac], axis=1)  # (F, 2, 3)
    # RHS: (F, 2)
    rhs = np.stack([
        (np.sum(b * b, axis=1) - np.sum(a * a, axis=1)) / 2.0,
        (np.sum(c * c, axis=1) - np.sum(a * a, axis=1)) / 2.0
    ], axis=1)
    
    # Solve least squares for each row
    # x = A^+ rhs, where A is 2x3
    # A^+ = (A^T A)^{-1} A^T
    ATA = np.einsum('fij,fkj->fik', A, A)  # (F, 2, 2)
    # Invert 2x2 matrices
    det = ATA[:, 0, 0] * ATA[:, 1, 1] - ATA[:, 0, 1] * ATA[:, 1, 0]
    # Avoid division by zero
    det_safe = np.where(np.abs(det) < 1e-15, 1.0, det)
    ATA_inv = np.zeros_like(ATA)
    ATA_inv[:, 0, 0] = ATA[:, 1, 1] / det_safe
    ATA_inv[:, 1, 1] = ATA[:, 0, 0] / det_safe
    ATA_inv[:, 0, 1] = -ATA[:, 0, 1] / det_safe
    ATA_inv[:, 1, 0] = -ATA[:, 1, 0] / det_safe
    
    # A^T is (F, 3, 2)
    AT = np.transpose(A, (0, 2, 1))  # (F, 3, 2)
    # x = ATA_inv @ AT @ rhs: (F, 2, 2) @ (F, 3, 2) @ (F, 2)
    # First: AT @ rhs -> (F, 3)
    AT_rhs = np.einsum('fij,fj->fi', AT, rhs)  # (F, 3)
    # Then: ATA_inv @ (AT_rhs) -> but ATA_inv is (F,2,2), AT_rhs is (F,3)
    # We need to compute for each f: ATA_inv[f] @ (AT[f] @ rhs[f])
    # AT[f] @ rhs[f] is already computed as AT_rhs[f] which is (3,)
    # But ATA_inv is 2x2, so we need to project. Let me re-derive.
    # Actually, x = (A^T A)^{-1} A^T rhs. A^T rhs is (3,), (A^T A)^{-1} is (2,2).
    # This doesn't work dimensionally. Let me reconsider.
    # 
    # We have A (2x3), rhs (2,). We want x (3,) such that A x ≈ rhs.
    # Least squares: x = (A^T A)^{-1} A^T rhs. A^T A is 3x3, A^T rhs is 3x1.
    # But A^T A is singular (rank 2). So we use pseudo-inverse or solve the 2x2 system differently.
    # 
    # Alternative: the circumcenter lies in the plane of the triangle. We can parametrize
    # x = a + s*d_ab + t*d_ac and solve for s, t.
    # |a + s*d_ab + t*d_ac - a|^2 = |a + s*d_ab + t*d_ac - b|^2
    # |s*d_ab + t*d_ac|^2 = |s*d_ab + t*d_ac - d_ab|^2
    # Let v = s*d_ab + t*d_ac. |v|^2 = |v - d_ab|^2 = |v|^2 - 2 v.d_ab + |d_ab|^2
    # So 2 v.d_ab = |d_ab|^2
    # Similarly 2 v.d_ac = |d_ac|^2
    # v.d_ab = s|d_ab|^2 + t d_ab.d_ac = |d_ab|^2/2
    # v.d_ac = s d_ab.d_ac + t|d_ac|^2 = |d_ac|^2/2
    # So: s|d_ab|^2 + t d_ab.d_ac = |d_ab|^2/2
    #     s d_ab.d_ac + t|d_ac|^2 = |d_ac|^2/2
    
    max_rad = 0.0
    for i in range(len(simplices)):
        dab = d_ab[i]
        dac = d_ac[i]
        dab2 = np.dot(dab, dab)
        dac2 = np.dot(dac, dac)
        dab_dac = np.dot(dab, dac)
        if dab2 < 1e-15 or dac2 < 1e-15:
            continue
        det_i = dab2 * dac2 - dab_dac * dab_dac
        if abs(det_i) < 1e-15:
            continue
        s = (0.5 * dab2 * dac2 - 0.5 * dab_dac * dab_dac) / det_i
        t = (0.5 * dac2 * dab2 - 0.5 * dab_dac * dab_dac) / det_i
        # Wait, let me redo this.
        # s*dab2 + t*dab_dac = dab2/2
        # s*dab_dac + t*dac2 = dac2/2
        # s = (dab2/2 * dac2 - dac2/2 * dab_dac) / det_i = (dab2*dac2 - dab_dac*dac2) / (2*det_i)
        # t = (dac2/2 * dab2 - dab2/2 * dab_dac) / det_i = (dac2*dab2 - dab2*dab_dac) / (2*det_i)
        s = (dab2 * dac2 - dab_dac * dac2) / (2.0 * det_i)
        t = (dac2 * dab2 - dab2 * dab_dac) / (2.0 * det_i)
        x = a[i] + s * dab + t * dac
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            continue
        x = x / norm
        dots = pts @ x
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        min_ang = np.min(angles)
        if min_ang > max_rad:
            max_rad = min_ang
    return np.degrees(max_rad)


def _objective_and_grad(flat_x):
    """Objective: covering radius in radians. Returns (value, gradient)."""
    pts = flat_x.reshape(N, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    # Compute covering radius
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0, np.zeros_like(flat_x)
    
    simplices = hull.simplices
    max_rad = 0.0
    max_idx = -1
    max_x = None
    
    for i, simplex in enumerate(simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        dab = b - a
        dac = c - a
        dab2 = np.dot(dab, dab)
        dac2 = np.dot(dac, dac)
        dab_dac = np.dot(dab, dac)
        if dab2 < 1e-15 or dac2 < 1e-15:
            continue
        det_i = dab2 * dac2 - dab_dac * dab_dac
        if abs(det_i) < 1e-15:
            continue
        s = (dab2 * dac2 - dab_dac * dac2) / (2.0 * det_i)
        t = (dac2 * dab2 - dab2 * dab_dac) / (2.0 * det_i)
        x = a + s * dab + t * dac
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            continue
        x = x / norm
        dots = pts @ x
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        min_ang = np.min(angles)
        if min_ang > max_rad:
            max_rad = min_ang
            max_idx = i
            max_x = x
    
    if max_idx < 0:
        return 1.0, np.zeros_like(flat_x)
    
    # Gradient computation is complex. For now, use finite differences or skip gradient.
    # Return just the value; let scipy use finite differences if needed.
    return max_rad, None


def _objective_only(flat_x):
    """Objective without gradient for L-BFGS-B."""
    pts = flat_x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.0
    
    simplices = hull.simplices
    max_rad = 0.0
    for simplex in simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        dab = b - a
        dac = c - a
        dab2 = np.dot(dab, dab)
        dac2 = np.dot(dac, dac)
        dab_dac = np.dot(dab, dac)
        if dab2 < 1e-15 or dac2 < 1e-15:
            continue
        det_i = dab2 * dac2 - dab_dac * dab_dac
        if abs(det_i) < 1e-15:
            continue
        s = (dab2 * dac2 - dab_dac * dac2) / (2.0 * det_i)
        t = (dac2 * dab2 - dab2 * dab_dac) / (2.0 * det_i)
        x = a + s * dab + t * dac
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            continue
        x = x / norm
        dots = pts @ x
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        min_ang = np.min(angles)
        if min_ang > max_rad:
            max_rad = min_ang
    return max_rad


def _random_start():
    """Generate a random set of 130 points on the unit sphere."""
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _fibonacci_start():
    """Fibonacci spiral start."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    best_pts = _fibonacci_start()
    best_score = _covering_radius_fast(best_pts)
    
    # Try a few random starts and pick the best initial
    for attempt in range(5):
        if time.time() - start_time > _DEADLINE - 30:
            break
        cand = _random_start()
        score = _covering_radius_fast(cand)
        if score < best_score:
            best_score = score
            best_pts = cand
    
    # Optimize using L-BFGS-B
    flat_x = best_pts.flatten().copy()
    
    # Run optimization with time check
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10  # 10s safety margin
    
    if remaining > 5:
        try:
            result = minimize(
                _objective_only,
                flat_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'disp': False}
            )
            if result.fun < best_score:
                best_pts = result.x.reshape(N, 3)
                best_score = result.fun
        except Exception:
            pass
    
    # Second round of optimization from the improved point
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10
    if remaining > 5 and best_pts is not None:
        flat_x2 = best_pts.flatten().copy()
        try:
            result2 = minimize(
                _objective_only,
                flat_x2,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 300, 'disp': False}
            )
            if result2.fun < best_score:
                best_pts = result2.x.reshape(N, 3)
                best_score = result2.fun
        except Exception:
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts
