N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _facet_circumradius(pts):
    """
    Compute the covering radius in radians using convex hull facets.
    For each triangular facet, the circumradius on the sphere is the
    maximum angular distance from the facet normal (pointing outward)
    to any of the three vertices.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2  # Return a large value if hull fails
    
    max_cr = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        # Ensure normal points outward
        if np.dot(n, p1) < 0:
            n = -n
        # The angular radius is the angle between n and any vertex
        # Due to numerical error, compute max over all three
        dots = np.array([np.dot(n, p1), np.dot(n, p2), np.dot(n, p3)])
        dots = np.clip(dots, -1.0, 1.0)
        cr = np.max(np.arccos(dots))
        if cr > max_cr:
            max_cr = cr
    return max_cr


def _find_largest_facet(pts):
    """
    Find the facet with the largest circumradius.
    Returns (normal_vector, circumradius, facet_indices)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, 0.0, None
    
    max_cr = 0.0
    best_n = None
    best_idx = None
    
    for simplex in hull.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        if np.dot(n, p1) < 0:
            n = -n
        dots = np.array([np.dot(n, p1), np.dot(n, p2), np.dot(n, p3)])
        dots = np.clip(dots, -1.0, 1.0)
        cr = np.max(np.arccos(dots))
        if cr > max_cr:
            max_cr = cr
            best_n = n
            best_idx = simplex
    
    return best_n, max_cr, best_idx


def _repulsion_forces(pts, power=2.0):
    """
    Compute repulsion forces between all point pairs.
    F_i = sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]]^(power+1)
    """
    forces = np.zeros_like(pts)
    n = len(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                forces[i] += np.random.normal(0, 0.01, 3)
                continue
            forces[i] += d / (dist ** (power + 1))
    return forces


def _facet_refine(pts, deadline, max_iter=500):
    """
    Refine using convex hull facet analysis with combined forces.
    At each step, find the largest facet, compute forces (repulsion + 
    attraction to hole), and update.
    """
    best_pts = pts.copy()
    best_cr = _facet_circumradius(pts)
    
    eta = 0.02
    alpha_att = 3.0
    p_rep = 2.0
    no_improve = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n_hole, current_cr, facet_idx = _find_largest_facet(pts)
        if n_hole is None:
            break
        
        # Compute forces
        forces = _repulsion_forces(pts, p_rep)
        
        # Add attraction to the hole for points near it
        for i in range(len(pts)):
            d = pts[i] - n_hole
            dist = np.linalg.norm(d)
            w = np.exp(-alpha_att * dist ** 2)
            forces[i] += w * (-d)
        
        # Update
        new_pts = pts + eta * forces
        new_pts = _normalize(new_pts)
        
        new_cr = _facet_circumradius(new_pts)
        
        if new_cr < best_cr - 1e-10:
            best_cr = new_cr
            best_pts = new_pts.copy()
            pts = new_pts
            no_improve = 0
            eta = min(eta * 1.01, 0.05)
        else:
            no_improve += 1
            eta *= 0.95
            if no_improve >= 50:
                # Perturb to escape local minimum
                noise = np.random.randn(len(pts), 3) * 0.01
                pts = _normalize(best_pts + noise)
                no_improve = 0
                eta = 0.02
    
    return best_pts


def _lbfgs_polish(pts, deadline, max_iter=100):
    """
    Final polishing with L-BFGS-B using a smooth approximation of the
    covering radius. Uses the convex hull facet radius as the objective.
    """
    def objective(x):
        p = _normalize(x.reshape(N, 3))
        return _facet_circumradius(p)
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-4}
        )
        return _normalize(res.x.reshape(N, 3))
    except Exception:
        return pts


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Phase 1: Generate multiple diverse starting configurations
    starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, 1),
        _random_sphere(N, 2),
        _random_sphere(N, 3),
        _random_sphere(N, 4),
    ]
    
    best_pts = starts[0].copy()
    best_cr = _facet_circumradius(best_pts)
    
    # Phase 2: Facet refinement from each start
    for i, s in enumerate(starts):
        if time.time() > deadline - 20:
            break
        sub_deadline = min(deadline - 10, start_time + 20 * (i + 1))
        try:
            refined = _facet_refine(s, sub_deadline, max_iter=300)
            cr = _facet_circumradius(refined)
            if cr < best_cr:
                best_cr = cr
                best_pts = refined.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polishing
    if time.time() < deadline - 5:
        polished = _lbfgs_polish(best_pts, deadline - 2, max_iter=80)
        cr = _facet_circumradius(polished)
        if cr < best_cr:
            best_cr = cr
            best_pts = polished
    
    # Final validation: ensure no coincident points, not in hemisphere
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_dist = min(min_dist, d)
    if min_dist < 1e-6:
        # Add small perturbation
        rng = np.random.default_rng(42)
        best_pts = _normalize(best_pts + rng.randn(N, 3) * 1e-4)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
