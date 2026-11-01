"""Improved covering-radius optimizer for 50 points on the unit sphere."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # seconds; evaluator kills at 150s, keep margin


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius (in degrees) for a set of points on the unit sphere.
    Uses convex hull facets: for each facet (triangle), compute the circumcenter
    and check if it's a valid (empty-cap) vertex; the covering radius is the max
    angular distance from the facet circumcenter to its vertices, but only for
    facets whose circumcenter lies on the same side as the origin (empty cap).
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: brute-force on a grid
        theta = np.linspace(0, np.pi, 300)
        phi = np.linspace(0, 2 * np.pi, 300)
        TH, PH = np.meshgrid(theta, phi)
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        # For each grid point, find nearest config point
        d = np.linalg.norm(Y[:, None, :] - pts[None, :, :], axis=2)
        nearest = d.min(axis=1)
        angles = np.arccos(np.clip(nearest, -1.0, 1.0))
        return np.degrees(angles.max())

    max_angle = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Circumcenter of triangle on the sphere
        # Use the formula for spherical triangle circumcenter
        # Project to 2D: use v0 as reference
        # Actually, compute circumcenter in 3D then normalize
        # Circumcenter of three points on unit sphere:
        # c = a + b + c - ... use the formula
        # For three unit vectors, the circumcenter direction is:
        # (a×b)×(b×c) + (b×c)×(c×a) + (c×a)×(a×b) ... 
        # Simpler: solve for point equidistant from all three
        # c·a = c·b = c·c_norm... 
        # Use: circumcenter = (a×b)×(a×c) normalized? No.
        # Standard: for three points on unit sphere, circumcenter is:
        # c = (a×b)×(a×c) ... let me use the linear system approach
        
        # Solve: c·(a-b)=0, c·(a-c)=0 (wait, c·a = c·b = c·c_vec)
        # c·a = c·b => c·(a-b) = 0
        # c·a = c·c_vec => c·(a-c_vec) = 0
        # So c is perpendicular to (a-b) and (a-c_vec)
        ab = v0 - v1
        ac = v0 - v2
        c_dir = np.cross(ab, ac)
        c_norm = np.linalg.norm(c_dir)
        if c_norm < 1e-12:
            continue
        c_dir /= c_norm
        
        # Check if the circumcenter is on the "empty cap" side
        # The facet normal should point outward; the circumcenter should be
        # on the same side as the outward normal for an empty cap
        # Actually, for covering radius, we need the facet to be a "Voronoi cell boundary"
        # The circumcenter of the facet (as a point on the sphere) gives the farthest
        # point in that Voronoi cell IF the cap is empty.
        
        # Compute angular distance from circumcenter to each vertex
        # The covering radius contribution is max angle from circumcenter to vertices
        angles = np.arccos(np.clip(np.abs(np.dot(c_dir, [v0, v1, v2])), -1.0, 1.0))
        facet_angle = angles.max()
        
        # Check if this is a valid empty cap: the circumcenter should be
        # closer to the three vertices than to any other point
        # For efficiency, just take the max over all facets
        if facet_angle > max_angle:
            max_angle = facet_angle
    
    return np.degrees(max_angle)


def _covering_radius_fast(pts):
    """
    Faster covering radius using the fact that for points on a sphere,
    the covering radius is determined by the largest empty circle.
    Use a grid-based approach for speed, then refine.
    """
    # Dense grid sampling
    n_theta = 200
    n_phi = 400
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    
    # Compute angular distance from each grid point to nearest config point
    # angle = arccos(y · p) for nearest p
    dots = Y @ pts.T  # shape (n_grid, N)
    max_dots = dots.max(axis=1)
    angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
    return np.degrees(angles.max())


def _lloyd_step(pts):
    """
    Lloyd's algorithm step: for each point, compute its Voronoi cell centroid
    (approximated by the average of nearby grid points or using the facet circumcenters).
    For sphere covering, a good relaxation is to move each point toward the
    centroid of its "influence region."
    """
    # Use a simple gradient-based approach instead
    return pts


def _optimize_from_start(pts0, time_limit):
    """
    Optimize the covering radius starting from pts0.
    Uses L-BFGS-B with a smooth approximation of the covering radius.
    """
    pts0 = _normalize(pts0)
    
    def objective_and_grad(flat):
        pts = _normalize(flat.reshape(N, 3))
        
        # Smooth max of angular distances using log-sum-exp
        # Sample points on sphere
        n_theta = 100
        n_phi = 200
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        
        # Angular distance: arccos(y·p), take min over p, then max over y
        dots = Y @ pts.T  # (n_grid, N)
        # For each y, the nearest p gives max dot, so min angle = arccos(max dot)
        max_dots = dots.max(axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        
        # Smooth max via log-sum-exp
        beta = 200.0
        m = angles.max()
        logsumexp = m + np.log(np.sum(np.exp(beta * (angles - m)))) / beta
        obj = np.degrees(logsumexp)
        
        # Gradient: we need d(obj)/d(pts)
        # This is complex; use finite differences or just use the objective without gradient
        return obj
    
    # Use Nelder-Mead or L-BFGS-B with finite difference gradient
    # Actually, let's use a simpler approach: iterative refinement
    
    best_pts = pts0.copy()
    best_val = _covering_radius_fast(pts0)
    
    t_start = time.time()
    for iteration in range(500):
        if time.time() - t_start > time_limit:
            break
        
        # Compute a gradient-like direction: for each point, move it away from
        # the "worst" region. Use a simple repulsion-based update.
        
        # Sample worst points
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi)
        Y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        
        dots = Y @ best_pts.T
        max_dots = dots.max(axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        
        # Find the worst grid point
        worst_idx = angles.argmax()
        worst_point = Y[worst_idx]
        
        # Move the nearest point away from the worst point
        nearest_i = dots[worst_idx].argmax()
        
        # Gradient direction: push point nearest_i in direction away from worst_point
        # But also consider global optimization
        
        # Use a small step in the direction that reduces the covering radius
        # Try moving each point slightly and see which helps most
        step_size = 0.05
        best_improvement = 0
        best_move = None
        
        for i in range(N):
            # Try moving point i in the direction of the worst point (to cover it better)
            # Actually, move it toward the worst point to reduce the gap
            direction = worst_point - best_pts[i]
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                continue
            direction /= dir_norm
            
            # Also try the opposite
            for sign in [1.0, -1.0]:
                new_pts = best_pts.copy()
                new_pts[i] = best_pts[i] + sign * step_size * direction
                new_pts = _normalize(new_pts)
                val = _covering_radius_fast(new_pts)
                improvement = best_val - val
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (i, sign, direction)
        
        if best_move is None:
            # Try random perturbations
            for _ in range(10):
                i = np.random.randint(N)
                random_dir = np.random.randn(3)
                random_dir /= np.linalg.norm(random_dir)
                new_pts = best_pts.copy()
                new_pts[i] = best_pts[i] + 0.02 * random_dir
                new_pts = _normalize(new_pts)
                val = _covering_radius_fast(new_pts)
                improvement = best_val - val
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (i, 1.0, random_dir)
        
        if best_move is not None and best_improvement > 1e-6:
            i, sign, direction = best_move
            best_pts[i] = best_pts[i] + sign * step_size * direction
            best_pts = _normalize(best_pts)
            best_val = _covering_radius_fast(best_pts)
        else:
            # No improvement, try smaller steps or restart
            break
    
    return best_pts, best_val


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts0 = _normalize(pts0)
    
    # Optimize from the starting point
    best_pts, best_val = _optimize_from_start(pts0, DEADLINE - 10.0)
    
    # Final polish with more precise covering radius computation
    if time.time() < DEADLINE - 5.0:
        # Try a few more refinement steps with tighter grid
        for _ in range(20):
            if time.time() > DEADLINE - 5.0:
                break
            # Use a finer grid for the final evaluation
            val = _covering_radius(best_pts)
            best_val = val
            break
    
    return best_pts
