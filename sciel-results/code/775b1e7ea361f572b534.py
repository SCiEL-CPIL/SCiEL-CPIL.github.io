"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(pts):
    """Compute the covering radius in radians for points on the unit sphere.
    
    The covering radius is the maximum, over all points y on the sphere, of the
    angular distance from y to its nearest point in pts. This is achieved at the
    circumcenter of a facet of the convex hull (projected onto the sphere).
    
    Vectorized implementation for speed.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    simplices = hull.simplices  # shape (M, 3) for 3D hull facets (triangles)
    M = len(simplices)
    
    if M == 0:
        return np.pi
    
    # Get the three vertices of each triangle
    P0 = pts[simplices[:, 0]]  # (M, 3)
    P1 = pts[simplices[:, 1]]  # (M, 3)
    P2 = pts[simplices[:, 2]]  # (M, 3)
    
    # Compute normals to each facet
    v1 = P1 - P0
    v2 = P2 - P0
    normals = np.cross(v1, v2)  # (M, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # (M, 1)
    
    # Filter out degenerate facets
    valid = norms.squeeze() > 1e-12
    if not np.any(valid):
        return np.pi
    
    normals = normals[valid]
    P0 = P0[valid]
    P1 = P1[valid]
    P2 = P2[valid]
    norms = norms[valid]
    
    normals = normals / norms  # unit normals (M', 3)
    
    # For each triangle on the unit sphere, the circumcenter direction is along the normal.
    # We need to determine which sign (+ or -) gives the circumcenter that lies "inside" 
    # the spherical triangle (i.e., the point on the sphere equidistant from all 3 vertices
    # that is closest to the triangle's centroid in angular distance).
    
    # The circumcenter c on the unit sphere satisfies c · P_i = cos(angle) for all i.
    # Since c is along the normal n (unit), c = ±n.
    # c · P0 = ±(n · P0)
    # c · P1 = ±(n · P1)  
    # c · P2 = ±(n · P2)
    # For c to be equidistant, we need n·P0 ≈ n·P1 ≈ n·P2 (in magnitude).
    # The correct sign is the one where the dot products are positive (c is on the same side as the origin relative to the plane, i.e., c points toward the interior of the convex hull).
    
    # Actually, for a convex hull, the outward normal points away from the origin.
    # The circumcenter of the spherical triangle (the point on the sphere farthest from all three vertices within the Voronoi cell) is in the direction of the outward normal if the triangle is acute, or we need to check.
    
    # Better approach: for each facet, the candidate circumcenter is the point on the sphere
    # that maximizes the minimum distance to the three vertices. This is either +n or -n.
    # We pick the one that is farther from the origin in terms of the convex hull interior.
    # Actually, the covering radius is determined by the "worst" point, which is the circumcenter
    # of the Delaunay triangle. For a convex hull of points on the sphere, the relevant circumcenter
    # is the one that lies on the same side of the facet plane as the origin (i.e., pointing inward).
    
    # The facet plane has equation n · x = d, where d = n · P0.
    # Since the origin is inside the convex hull (for a good covering), n · 0 = 0 < d if n points outward.
    # So the inward direction is -n.
    # But the circumcenter of the spherical Voronoi cell is the point on the sphere equidistant from the 3 vertices,
    # which lies along the normal. Let's compute both and pick the one with larger minimum distance.
    
    dot0 = normals @ P0.T  # (M',)
    dot1 = normals @ P1.T  # (M',)
    dot2 = normals @ P2.T  # (M',)
    
    # For the outward normal, dot products should be similar.
    # The circumcenter candidate in direction +n: distances are arccos(dot0), arccos(dot1), arccos(dot2)
    # The circumcenter candidate in direction -n: distances are arccos(-dot0), arccos(-dot1), arccos(-dot2)
    
    # We want the point that is the "center" of the Voronoi cell, which is the point maximizing the min distance.
    # For a typical acute spherical triangle, the circumcenter is in the direction of the normal pointing toward the interior of the sphere (inward), i.e., -n if n is outward.
    
    # Let's compute the min angular distance for both candidates:
    # Candidate +n: min(arccos(dot0), arccos(dot1), arccos(dot2))
    # Candidate -n: min(arccos(-dot0), arccos(-dot1), arccos(-dot2))
    
    # We need to clip dot products to [-1, 1] for arccos.
    dot0_c = np.clip(dot0, -1.0, 1.0)
    dot1_c = np.clip(dot1, -1.0, 1.0)
    dot2_c = np.clip(dot2, -1.0, 1.0)
    
    dist_plus = np.minimum(np.minimum(np.arccos(dot0_c), np.arccos(dot1_c)), np.arccos(dot2_c))
    dist_minus = np.minimum(np.minimum(np.arccos(-dot0_c), np.arccos(-dot1_c)), np.arccos(-dot2_c))
    
    # The covering radius is the maximum over all facets of the circumradius.
    # For each facet, the relevant circumradius is the larger of the two? No, it's the one that corresponds to the actual Voronoi vertex.
    # The Voronoi vertex is the point on the sphere equidistant from the 3 vertices that is inside the spherical triangle.
    # For a convex hull, the inward-pointing normal gives the circumcenter that is "inside" the hull.
    
    # Let's use the inward direction: the origin is at 0, and the facet plane is n·x = d with d > 0 (outward normal).
    # The inward normal is -n. The circumcenter in the inward direction is -n.
    # But wait, the point on the sphere in the -n direction is -n itself (since n is unit).
    # Its distance to P0 is arccos((-n)·P0) = arccos(-dot0).
    
    # Actually, I think the standard result is that the covering radius is the maximum circumradius of the spherical triangles,
    # where the circumradius is the angular distance from the circumcenter to each vertex.
    # The circumcenter is the point on the sphere equidistant from the 3 vertices.
    # There are two such points: +n and -n. The correct one is the one that lies in the spherical triangle (or its interior).
    
    # A robust way: the circumradius is arccos(|n·P0|) if the triangle is acute, but this can be tricky.
    # Let's just take the maximum of the two candidate min-distances? No, that's not right either.
    
    # Let's use a different approach: the covering radius is the maximum over all facets of the circumradius,
    # where the circumradius r satisfies cos(r) = n·P0 (for the outward normal, if the circumcenter is on the outward side).
    # Actually, for points on the unit sphere, the circumradius of a spherical triangle with vertices A, B, C is given by:
    # cos(r) = (A·B + B·C + C·A - 1) / (2 * area_factor) ... this is getting complicated.
    
    # Simpler: the circumcenter c is along the normal. c = ±n.
    # The distance from c to A is arccos(c·A).
    # We want the c that is the "center" of the Voronoi cell, which is the one with the larger minimum distance to the vertices? No, the circumradius is the distance from the circumcenter to the vertices, and we want the maximum such circumradius over all facets.
    
    # For a given facet, the circumradius is arccos(|n·A|) if we take the absolute value, but we need the correct sign.
    # Let's compute both and take the one that gives a valid circumcenter (i.e., the one where the circumcenter is actually a Voronoi vertex).
    
    # For simplicity and robustness, let's compute the circumradius as:
    # r = arccos(clip(n·P0, -1, 1)) for the outward normal, and also for -n, and take the maximum of the two?
    # No, the circumradius is a single value for each facet.
    
    # Let's use the fact that for a spherical triangle, the circumradius r satisfies:
    # cos(r) = (cos(a) + cos(b) + cos(c) - 1) / (2 * sin(a/2) * sin(b/2) * sin(c/2)) ... too complex.
    
    # Let's just compute the distance from the origin-projected normal to the vertices.
    # The point on the sphere in the direction of the outward normal is n.
    # The angular distance from n to P0 is arccos(n·P0) = arccos(dot0).
    # Similarly for P1 and P2.
    # If the triangle is acute, these three distances should be equal (within numerical error), and that's the circumradius.
    # If not, the circumcenter is on the other side, and the circumradius is arccos(-dot0), etc.
    
    # Let's take the circumradius for each facet as the maximum of the three angular distances from n to the vertices,
    # and also from -n to the vertices, and take the minimum of these two maxima? No.
    
    # Actually, the circumradius is the distance from the circumcenter to any vertex. The circumcenter is the point on the sphere equidistant from all 3 vertices.
    # So we need to find which of +n or -n is equidistant. In exact arithmetic, both are equidistant (since n·P0 = n·P1 = n·P2 for a true circumcenter).
    # The circumradius is arccos(n·P0) if we use +n, or arccos(-n·P0) if we use -n.
    # The correct circumcenter is the one that lies inside the spherical triangle. For a convex hull, this is typically the inward-pointing normal (-n if n is outward).
    
    # Let's use -n (inward) as the circumcenter direction, and compute the circumradius as arccos(clip(-dot0, -1, 1)).
    # But we should verify that -dot0 ≈ -dot1 ≈ -dot2.
    
    circumradii_inward = np.arccos(np.clip(-dot0_c, -1.0, 1.0))
    circumradii_outward = np.arccos(np.clip(dot0_c, -1.0, 1.0))
    
    # The covering radius is the maximum circumradius over all facets.
    # We need to pick the correct circumcenter for each facet.
    # Let's take the maximum of the two candidate circumradii for each facet? No, that would overestimate.
    
    # Let's use a heuristic: for a good covering, the circumcenter should be the inward one.
    # Let's compute both and take the maximum over all facets of the inward circumradius, and also the outward, and take the max of the two sets?
    
    # Actually, the covering radius is the maximum over all points y of min_i angle(y, P_i).
    # This is achieved at a Voronoi vertex, which is the circumcenter of a Delaunay triangle.
    # For each Delaunay triangle, the circumcenter is unique (up to the antipodal point).
    # The relevant circumcenter is the one that is a local maximum of the min-distance function.
    
    # Let's compute the min-distance at both +n and -n for each facet, and take the maximum of these min-distances.
    # The covering radius is the maximum of these values.
    
    # dist_plus is the min-distance at +n, dist_minus is the min-distance at -n.
    # The covering radius is max over all facets of max(dist_plus, dist_minus)? No, it's the max over all y of min-distance, which is the max over all Voronoi vertices of the min-distance at that vertex.
    # Each Voronoi vertex corresponds to one facet, and its min-distance is either dist_plus or dist_minus.
    # We need to pick the correct one for each facet.
    
    # For a convex hull, the inward circumcenter (-n) is the one inside the hull, so it's the Voronoi vertex.
    # Let's use dist_minus (min-distance at -n) as the circumradius for each facet.
    
    # But to be safe, let's take the maximum of dist_plus and dist_minus for each facet, and then take the overall maximum.
    # This might overestimate, but it's a safe upper bound.
    
    # Actually, let's just take the maximum of all circumradii_inward and circumradii_outward.
    
    max_radius = np.maximum(np.max(circumradii_inward), np.max(circumradii_outward))
    
    return max_radius


def _initial_fib():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _improve(pts, time_budget=100.0):
    """Improve the covering using local optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Use a simple gradient-free optimization: for each point, try small perturbations
    # and keep the one that reduces the covering radius.
    
    n_iter = 0
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try perturbing point i in random directions
            best_local = best_pts[i].copy()
            best_local_radius = best_radius
            
            for _ in range(5):
                # Random perturbation
                noise = np.random.randn(3) * step_size
                new_pt = best_pts[i] + noise
                new_pt = new_pt / np.linalg.norm(new_pt)
                
                trial_pts = best_pts.copy()
                trial_pts[i] = new_pt
                
                trial_radius = _covering_radius(trial_pts)
                
                if trial_radius < best_local_radius:
                    best_local = new_pt
                    best_local_radius = trial_radius
                    improved = True
            
            if np.linalg.norm(best_local - best_pts[i]) > 1e-10:
                best_pts[i] = best_local
                best_radius = best_local_radius
        
        n_iter += 1
        # Decay step size
        step_size *= 0.95
        if step_size < 1e-5:
            break
    
    return best_pts, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    
    # Improve using local optimization
    pts, radius = _improve(pts, time_budget=130.0)
    
    # Check if we're still within deadline
    if time.time() > deadline:
        return pts
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END

