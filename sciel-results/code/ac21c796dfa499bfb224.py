"""
Sphere covering with 130 points: minimize the covering radius.
Uses a multi-stage approach:
1. Start from multiple diverse initial configurations (Fibonacci, icosahedral, random).
2. Optimize using a smooth surrogate objective (log-sum-exp of pairwise distances)
   with L-BFGS-B, annealing the temperature.
3. Refine with local search: for each point, try to move it to reduce the
   maximum distance to its Voronoi cell's circumcenter.
4. Use the convex hull to compute the exact covering radius.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    """Normalize rows of pts to unit length."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _covering_radius_deg(pts):
    """
    Compute the covering radius in degrees using the convex hull.
    The covering radius is the maximum circumradius of the spherical triangles
    (facets) of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        facets = hull.simplices
        max_circ = 0.0
        for tri in facets:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            # Compute circumcenter of spherical triangle
            # The circumcenter is along the direction of the cross product of (b-a) and (c-a)
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # Choose the sign so that the circumcenter is on the same side as the facet
            fa = a + b + c
            fa_norm = np.linalg.norm(fa)
            if fa_norm < 1e-12:
                continue
            fa = fa / fa_norm
            if np.dot(n, fa) < 0:
                n = -n
            cos_r = np.clip(np.dot(n, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_circ:
                max_circ = r
        return np.degrees(max_circ)
    except Exception:
        return 180.0


def _lse_objective_and_grad(flat, beta):
    """
    Log-sum-exp surrogate for the covering problem.
    Minimizes a smooth approximation to the maximum pairwise angular distance.
    """
    pts = flat.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute all pairwise dot products
    D = pts @ pts.T
    
    # We want to maximize the minimum off-diagonal dot product
    # Minimize: (1/beta) * log(sum_{i != j} exp(-beta * D[i,j]))
    
    mask = ~np.eye(N, dtype=bool)
    vals = -beta * D[mask]
    
    # Log-sum-exp for numerical stability
    max_val = np.max(vals)
    if max_val > 100:
        logsumexp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    else:
        logsumexp = np.log(np.sum(np.exp(vals)))
    
    f = logsumexp / beta
    
    # Gradient
    grad_pts = np.zeros_like(pts)
    for i in range(N):
        dots_i = -beta * D[i]
        dots_i[i] = -np.inf
        max_v = np.max(dots_i)
        exps = np.exp(dots_i - max_v)
        sum_exps = np.sum(exps)
        if sum_exps < 1e-300:
            continue
        weights = exps / sum_exps
        grad_pts[i] = -np.sum(pts * weights[:, None], axis=0)
    
    grad_flat = grad_pts.reshape(-1)
    
    return f, grad_flat


def _local_search(pts, max_iters=50):
    """
    Local search: for each point, try to move it toward the circumcenter of its
    Voronoi cell (the point on the sphere farthest from all other points in its cell).
    This is a simple refinement step.
    """
    pts = _normalize(pts.copy())
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Find the point on the sphere farthest from p_i among all other points
            # This is an approximation: find the direction that maximizes the minimum
            # distance to all other points
            dots = pts @ pts[i]  # dot products with all points
            dots[i] = -np.inf  # exclude self
            
            # The worst case is the point with the largest angular distance
            # i.e., the smallest dot product
            j = np.argmin(dots)
            
            # Move p_i slightly away from p_j
            # The direction to move is away from p_j, but constrained to the sphere
            # Use a small step in the direction of -p_j (projected to tangent plane)
            tangent = -pts[j] - np.dot(pts[i], pts[j]) * pts[i]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-12:
                continue
            tangent = tangent / tangent_norm
            
            # Try a small step
            step_size = 0.01
            new_p = pts[i] + step_size * tangent
            new_p = _normalize(new_p.reshape(1, 3))[0]
            
            # Check if this improves the covering radius
            # Approximate: check if the minimum distance from new_p to all other points
            # is larger than the current minimum distance from p_i
            old_min_dist = np.arccos(np.clip(np.min(dots), -1.0, 1.0))
            new_dots = pts @ new_p
            new_dots[i] = -np.inf
            new_min_dist = np.arccos(np.clip(np.min(new_dots), -1.0, 1.0))
            
            if new_min_dist > old_min_dist:
                pts[i] = new_p
                improved = True
        
        if not improved:
            break
    
    return pts


def _fibonacci_spiral(n, offset=0.0):
    """Generate points on the sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)


def _icosahedral_seed(n):
    """Start from icosahedron vertices and add more points."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = _normalize(verts)
    rng = np.random.RandomState(42)
    extra = rng.randn(n - 12, 3)
    extra = _normalize(extra)
    return _normalize(np.vstack([verts, extra]))


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Generate several initial configurations
    inits = [
        _fibonacci_spiral(N, 0.0),
        _fibonacci_spiral(N, 1.0),
        _fibonacci_spiral(N, 2.5),
        _random_init(N, 0),
        _random_init(N, 1),
        _random_init(N, 2),
        _icosahedral_seed(N),
    ]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        flat = init_pts.reshape(-1).copy()
        
        # Anneal beta from low to high
        betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            # L-BFGS-B optimization
            try:
                res = minimize(
                    lambda x: _lse_objective_and_grad(x, beta)[0],
                    flat,
                    jac=lambda x: _lse_objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'gtol': 1e-6, 'maxls': 50}
                )
                flat = res.x
            except Exception:
                break
            
            if time.time() > deadline - 10:
                break
        
        # Local search refinement
        final_pts = _normalize(flat.reshape(N, 3))
        if time.time() < deadline - 5:
            final_pts = _local_search(final_pts, max_iters=20)
        
        # Evaluate actual covering radius
        score = _covering_radius_deg(final_pts)
        
        if score < best_score:
            best_score = score
            best_pts = final_pts.copy()
        
        if time.time() > deadline - 15:
            break
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spiral(N, 0.0)
    
    return best_pts

# EVOLVE-BLOCK-END
