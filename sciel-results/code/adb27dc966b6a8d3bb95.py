"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """Compute the true covering radius in degrees using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_cr_rad = 0.0
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            # Compute circumcenter of spherical triangle a,b,c
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                cr = max(d_ab, d_bc, d_ca) / 2.0
            else:
                u = v / norm_v
                
                # Determine the correct sign for the circumcenter
                # The circumcenter must be on the same side of the chord plane as the origin?
                # No, the circumcenter of the spherical triangle is the point on the sphere equidistant from a,b,c.
                # The vector u is perpendicular to the plane of a,b,c.
                # We need to check which of u or -u is the circumcenter.
                # The circumcenter C satisfies |C-a| = |C-b| = |C-c|.
                # Since C is on the sphere, C = ±u.
                # Let's check the dot products.
                # If u is the circumcenter, dot(u,a) = dot(u,b) = dot(u,c) = cos(cr).
                # Due to symmetry, one of the signs will yield equal dot products (within numerical error).
                # Actually, it's simpler: the circumcenter is the one that is "inside" the spherical triangle.
                # For a convex hull facet, the circumcenter is the one that points "outwards" relative to the polytope?
                # No, the circumcenter of the facet is the center of the empty cap.
                
                # Robust check:
                # Calculate the distance from u to a, b, c.
                d_u_a = np.arccos(np.clip(np.dot(u, a), -1, 1))
                d_u_b = np.arccos(np.clip(np.dot(u, b), -1, 1))
                d_u_c = np.arccos(np.clip(np.dot(u, c), -1, 1))
                
                d_neg_u_a = np.arccos(np.clip(np.dot(-u, a), -1, 1))
                d_neg_u_b = np.arccos(np.clip(np.dot(-u, b), -1, 1))
                d_neg_u_c = np.arccos(np.clip(np.dot(-u, c), -1, 1))
                
                # The correct circumcenter will have minimal variance in distances to vertices
                var_u = np.var([d_u_a, d_u_b, d_u_c])
                var_neg_u = np.var([d_neg_u_a, d_neg_u_b, d_neg_u_c])
                
                if var_u < var_neg_u:
                    cr = np.mean([d_u_a, d_u_b, d_u_c])
                else:
                    cr = np.mean([d_neg_u_a, d_neg_u_b, d_neg_u_c])
            
            max_cr_rad = max(max_cr_rad, cr)
        
        return np.degrees(max_cr_rad)
    except Exception:
        # Fallback to grid sampling if hull fails
        return _covering_radius_grid(pts)

def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.degrees(np.max(min_angles))

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n, seed):
    """Start from icosahedron vertices and subdivide."""
    rng = np.random.RandomState(seed)
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    ico = _normalize(ico)
    all_pts = np.vstack([ico, _random_points(n - len(ico), seed)])
    selected = [0]
    for _ in range(n - 1):
        min_dists = np.min([np.arccos(np.clip(np.dot(all_pts, all_pts[i]), -1, 1)) for i in selected], axis=0)
        selected.append(np.argmax(min_dists))
    return _normalize(all_pts[selected])

def _spherical_lloyd(pts, max_iter=200, tol=1e-8):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point to the centroid of its Voronoi cell.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        incident_facets = [[] for _ in range(n)]
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            normal = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            # Ensure normal points outward
            if np.dot(normal, a) < 0:
                normal = -normal
                
            for idx in simplex:
                incident_facets[idx].append(normal)
                
        new_pts = np.zeros_like(pts)
        max_move = 0.0
        
        for i in range(n):
            if len(incident_facets[i]) == 0:
                new_pts[i] = pts[i]
                continue
                
            centroid = np.sum(incident_facets[i], axis=0)
            norm_c = np.linalg.norm(centroid)
            
            if norm_c < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = centroid / norm_c
                
            move = np.linalg.norm(new_pts[i] - pts[i])
            max_move = max(max_move, move)
            
        pts = _normalize(new_pts)
        
        if max_move < tol:
            break
            
    return pts

def _facet_relax(pts, max_iter=200, initial_step=0.5, decay=0.99, beta=10.0):
    """
    Optimizes the configuration by moving each point towards the weighted
    average of the circumcenters of its incident facets.
    Weights are based on the circumradius of the facet to focus on worst areas.
    """
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step
    
    for it in range(max_iter):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))
        
        max_cr_global = 0.0
        facet_data = []

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                continue
            u = v / norm_v
            
            # Determine correct circumcenter direction
            d_u_a = np.dot(u, a)
            d_u_b = np.dot(u, b)
            d_u_c = np.dot(u, c)
            
            d_neg_u_a = -d_u_a
            d_neg_u_b = -d_u_b
            d_neg_u_c = -d_u_c
            
            var_u = np.var([d_u_a, d_u_b, d_u_c])
            var_neg_u = np.var([d_neg_u_a, d_neg_u_b, d_neg_u_c])
            
            if var_u < var_neg_u:
                cc = u
                cr = np.arccos(np.clip(d_u_a, -1, 1))
            else:
                cc = -u
                cr = np.arccos(np.clip(d_neg_u_a, -1, 1))
                
            max_cr_global = max(max_cr_global, cr)
            facet_data.append((simplex, cc, cr))
            
        if max_cr_global < 1e-6:
            break

        for simplex, cc, cr in facet_data:
            # Weight by how much this facet exceeds the average, or just by cr
            # Using exp(beta * cr) focuses heavily on the largest facets
            w = math.exp(beta * (cr - max_cr_global)) # Normalize by max to keep weights stable
            
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc
                
        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True
                        
        if not moved:
            break
            
        pts = _normalize(pts)
        
        # Decay step size
        step_size *= decay
        
        if step_size < 1e-8:
            break

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    # Multiple diverse initializations
    initializations = []
    initializations.append(('fib', _fibonacci_points(N)))
    for seed in range(5):
        initializations.append((f'rand_{seed}', _random_points(N, seed)))
    for seed in range(3):
        initializations.append((f'ico_{seed}', _icosahedron_based_points(N, seed)))
    
    best_pts = None
    best_cr = np.inf
    
    # Evaluate each initialization with exact metric to pick best start
    for name, pts in initializations:
        if time.time() > deadline - 30:
            break
        cr = _covering_radius_exact(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Refine the best initialization using geometric optimizers
    if best_pts is not None and time.time() < deadline - 10:
        # Stage 1: Spherical Lloyd for general distribution
        lloyd_pts = _spherical_lloyd(best_pts, max_iter=100)
        lloyd_cr = _covering_radius_exact(lloyd_pts)
        if lloyd_cr < best_cr:
            best_cr = lloyd_cr
            best_pts = lloyd_pts
            
        # Stage 2: Facet Relaxation to minimize max covering radius
        if time.time() < deadline - 5:
            relax_pts = _facet_relax(best_pts, max_iter=100)
            relax_cr = _covering_radius_exact(relax_pts)
            if relax_cr < best_cr:
                best_cr = relax_cr
                best_pts = relax_pts
                
            # Stage 3: Another Lloyd pass to settle
            if time.time() < deadline - 2:
                lloyd_pts2 = _spherical_lloyd(best_pts, max_iter=50)
                lloyd_cr2 = _covering_radius_exact(lloyd_pts2)
                if lloyd_cr2 < best_cr:
                    best_cr = lloyd_cr2
                    best_pts = lloyd_pts2

    # If we still have time, try refining other good initializations
    if time.time() < deadline - 20:
        scored_inits = []
        for name, pts in initializations:
            if time.time() > deadline - 15:
                break
            cr = _covering_radius_exact(pts)
            scored_inits.append((cr, pts))
            
        scored_inits.sort(key=lambda x: x[0])
        
        for cr, pts in scored_inits:
            if time.time() > deadline - 15:
                break
            if cr > best_cr + 2.0: # Skip if significantly worse
                continue
            if time.time() < deadline - 10:
                # Quick refinement
                lloyd_pts = _spherical_lloyd(pts, max_iter=50)
                lloyd_cr = _covering_radius_exact(lloyd_pts)
                if lloyd_cr < best_cr:
                    best_cr = lloyd_cr
                    best_pts = lloyd_pts
                if time.time() < deadline - 5:
                    relax_pts = _facet_relax(best_pts, max_iter=50)
                    relax_cr = _covering_radius_exact(relax_pts)
                    if relax_cr < best_cr:
                        best_cr = relax_cr
                        best_pts = relax_pts

    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
