"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius_deg(pts):
    """
    Compute covering radius in degrees.
    Returns: (radius_deg, worst_facet_indices)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0, None
        
    facets = hull.simplices
    max_circum = 0.0
    worst_facet = None
    
    # Vectorized computation of circumradii is tricky due to the normalization and arccos
    # Loop is fast enough for N=130 (hull has ~200-300 facets)
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        
        # Normal to the plane of the triangle
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n = n / nn
        
        # Check orientation: circumcenter should be on the side of the triangle 
        # that is "outward" from the origin? 
        # Actually, for a convex hull of points on a sphere, the circumcenter 
        # of a facet is the point on the sphere furthest from the origin in the 
        # direction of the facet normal? No.
        # The circumcenter C is +/- n.
        # We want the C that is on the sphere.
        # Both n and -n are on the sphere.
        # The correct one is the one that is "outside" the convex hull?
        # The convex hull contains the origin (if points are not in a hemisphere).
        # The facet normal n points outward.
        # The circumcenter of the spherical triangle is the point on the sphere 
        # equidistant from a,b,c.
        # It is given by the normalized vector:
        # C = normalize( a * (b x c) + b * (c x a) + c * (a x b) ) ? 
        # Or simply, C is parallel to (b-a) x (c-a) ?
        # Let's verify.
        # C . (a-b) = 0 and C . (b-c) = 0.
        # So C is parallel to (a-b) x (b-c) which is parallel to (b-a) x (c-a) = n.
        # So C is either n or -n.
        # Which one?
        # The circumcenter must be on the same side of the great circle arc ab as c?
        # Actually, for the covering radius, we want the point on the sphere 
        # that is farthest from a,b,c.
        # If the triangle is small, the circumcenter is "outside" the triangle 
        # on the sphere?
        # Let's test with a small triangle near (1,0,0).
        # a=(1,0,0), b=(1,eps,0), c=(1,0,eps).
        # n = (b-a) x (c-a) = (0,eps,0) x (0,0,eps) = (eps^2, 0, 0).
        # n normalized is (1,0,0) = a.
        # C = (1,0,0).
        # Distance to a is 0.
        # This is the "incenter" or "circumcenter"?
        # For a small triangle, the circumcenter is near the centroid.
        
        # Let's use the property: C is the solution to C.a = C.b = C.c.
        # C = n / |n| or -n / |n|.
        # Let's pick the one with positive dot product with the centroid of the triangle?
        centroid = (a + b + c) / 3
        if np.dot(n, centroid) < 0:
            n = -n
        
        C = n
        cos_R = np.clip(np.dot(C, a), -1, 1)
        R = np.arccos(cos_R)
        
        if R > max_circum:
            max_circum = R
            worst_facet = tri

    return np.degrees(max_circum), worst_facet

def _repulsion_step(pts, dt):
    """
    Apply repulsive force to spread points.
    Force is 1/r^2.
    """
    # Compute pairwise distances
    # To save memory/time, we can do it in chunks or just full for N=130
    # Dots product
    dots = np.clip(pts @ pts.T, -1, 1)
    # Distances squared: 2 - 2*dots
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    # Force magnitude: 1 / dist_sq^1.5 ? 
    # Potential 1/r -> Force 1/r^2.
    # On sphere, distance is angular. 
    # Let's use angular distance.
    dists = np.arccos(dots)
    np.fill_diagonal(dists, 1.0)
    
    # Force vector: sum (p_i - p_j) / |p_i - p_j|^2
    # This is equivalent to gradient of sum 1/|p_i - p_j|
    # Let's use a softer repulsion to avoid instability
    # F = sum (p_i - p_j) / dist^2
    
    forces = np.zeros_like(pts)
    # Vectorized force calculation is hard with the 1/dist^2 weighting on the vector difference
    # But we can do:
    # F_i = sum_j (p_i - p_j) / d_ij^2
    # (p_i - p_j) = p_i - p_j
    # We can compute the sum of p_j / d_ij^2 and then F_i = p_i * sum(1/d^2) - sum(p_j / d^2)
    
    inv_d_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_d_sq, 0.0)
    
    sum_inv_d_sq = np.sum(inv_d_sq, axis=1)
    
    # Sum of p_j * inv_d_sq[i,j]
    # This is a matrix multiplication: inv_d_sq @ pts
    sum_p_j_w = inv_d_sq @ pts
    
    forces = pts * sum_inv_d_sq[:, np.newaxis] - sum_p_j_w
    
    # Update
    new_pts = pts + dt * forces
    return _normalize(new_pts)

def _local_refine_worst(pts, worst_facet, max_iter=50, time_limit=2.0):
    """
    Refine the 3 points of the worst facet using Nelder-Mead.
    """
    if worst_facet is None:
        return pts
        
    tri = worst_facet
    # Initial simplex for Nelder-Mead
    # Variables: 3 points, 3 coords each.
    # But we must keep them on the sphere.
    # We optimize the tangent plane coordinates?
    # Or just optimize the 3D coords and normalize in the objective?
    
    x0 = np.concatenate([pts[tri[0]], pts[tri[1]], pts[tri[2]]])
    
    def objective(x):
        new_pts = pts.copy()
        new_pts[tri[0]] = x[0:3]
        new_pts[tri[1]] = x[3:6]
        new_pts[tri[2]] = x[6:9]
        
        # Normalize
        new_pts = _normalize(new_pts)
        
        # Check if valid
        rad, _ = _covering_radius_deg(new_pts)
        return rad
        
    # Nelder-Mead options
    options = {'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6}
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.success:
            # Update the points
            pts[tri[0]] = res.x[0:3]
            pts[tri[1]] = res.x[3:6]
            pts[tri[2]] = res.x[6:9]
            pts = _normalize(pts)
    except Exception:
        pass
        
    return pts

def construct():
    t0 = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_score, _ = _covering_radius_deg(pts)
    
    # 2. Repulsion Phase
    # Spread points out
    dt = 0.05
    for i in range(200):
        if time.time() > deadline - 100: # Reserve 100s for refinement
            break
        pts = _repulsion_step(pts, dt)
        dt *= 0.99 # Decay step size
        
    # Check score after repulsion
    score, worst = _covering_radius_deg(pts)
    if score < best_score:
        best_score = score
        best_pts = pts.copy()
    
    # 3. Local Refinement Phase
    # Iteratively refine the worst facet
    current_pts = best_pts.copy()
    current_score, worst_facet = _covering_radius_deg(current_pts)
    
    stall_count = 0
    while time.time() < deadline - 5:
        if worst_facet is None:
            break
            
        # Refine the worst facet
        new_pts = _local_refine_worst(current_pts, worst_facet, max_iter=30, time_limit=2.0)
        
        new_score, new_worst = _covering_radius_deg(new_pts)
        
        if new_score < current_score:
            current_pts = new_pts
            current_score = new_score
            worst_facet = new_worst
            stall_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # No improvement
            stall_count += 1
            # Try a small random perturbation to escape local minima
            if stall_count > 5:
                # Perturb a random point
                idx = np.random.randint(N)
                # Small random step
                rand_dir = np.random.randn(3)
                rand_dir -= np.dot(rand_dir, current_pts[idx]) * current_pts[idx]
                rand_dir = _normalize(rand_dir.reshape(1,3)).flatten()
                
                new_pts = current_pts.copy()
                new_pts[idx] = current_pts[idx] + 0.1 * rand_dir
                new_pts = _normalize(new_pts)
                
                p_score, p_worst = _covering_radius_deg(new_pts)
                if p_score < current_score:
                    current_pts = new_pts
                    current_score = p_score
                    worst_facet = p_worst
                    stall_count = 0
                else:
                    # If still stuck, break or try another facet?
                    # For now, break to avoid wasting time
                    break
            else:
                # Try the second worst facet?
                # We only have the worst. Let's just break if stuck too much.
                if stall_count > 20:
                    break

    return best_pts
# EVOLVE-BLOCK-END
