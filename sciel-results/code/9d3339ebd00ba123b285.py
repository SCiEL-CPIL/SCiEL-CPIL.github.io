<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START


def _covering_radius_deg(pts):
    """Compute covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        # For each facet, find the farthest point from the facet plane
        # The covering radius is the max over all facets of the angular distance
        # from the facet centroid (projected to sphere) to the facet vertices
        # Actually, we need to find the max empty cap radius
        # For each facet, the circumradius of the triangle (in angular terms)
        # The covering radius is the max over all facets of the largest empty cap
        
        # Get facet normals and compute the angular circumradius for each facet
        # A facet with vertices i,j,k has a circumcenter on the sphere
        # The angular distance from that center to any vertex is the circumradius
        
        # For a spherical triangle, the circumradius can be computed
        # But a simpler approach: for each facet, the farthest point on the sphere
        # from all three vertices is at the circumcenter of the triangle
        
        # Alternative: use the fact that for each facet, the covering radius
        # is the max angular distance from the facet's circumcenter to its vertices
        
        # Let's compute using the formula for spherical triangle circumradius
        # For triangle with side lengths a,b,c (angular), circumradius R satisfies:
        # tan(R) = (tan(a/2)*tan(b/2)*tan(c/2)) / (tan(a/2)+tan(b/2)+tan(c/2) - 2*tan(a/2)*tan(b/2)*tan(c/2))
        # But this can be numerically unstable. Let's use a different approach.
        
        # For each facet, compute the circumcenter and the angular distance to vertices
        # The circumcenter of a spherical triangle can be found as the intersection
        # of the great circle perpendicular bisectors
        
        # Simpler: for each facet, the covering radius contribution is the max
        # angular distance from the circumcenter to the three vertices.
        # We can find the circumcenter by solving for the point equidistant from all three.
        
        # Actually, let's just use the sample-based approach for speed
        # and use hull only for validation
        return None
    except:
        return None


def _covering_radius_samples(pts, n_samples=10000):
    """Compute approximate covering radius using sample points."""
    # Generate sample points on sphere
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, find distance to nearest point
    # Use dot products for angular distance
    dots = samples @ pts.T  # shape (n_samples, N)
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance = arccos(dot)
    # We want the max over samples of min over pts of arccos(dot)
    # = max over samples of arccos(max over pts of dot)
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.degrees(np.max(angles))


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s eval
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    best_pts = pts.copy()
    best_radius = _covering_radius_samples(best_pts, 5000)
    
    # Local optimization: iteratively move points to reduce covering radius
    # Strategy: for each point, find the direction that most reduces the local covering radius
    # Use a simple gradient-free approach: try small perturbations
    
    step_size = 0.01
    n_samples = 5000
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        
        # Try to improve by moving each point slightly
        improved = False
        for idx in range(N):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 5.0:
                break
            
            # Compute local covering radius contribution for this point
            # The point is responsible for the cap around it
            # Move it in the direction that reduces the max angular distance in its region
            
            # Find the farthest sample point from this point (within its Voronoi region)
            # Approximate: find the direction of the farthest point in the hemisphere around this point
            
            # Get samples that are closer to this point than to any other
            dots_all = samples_cache @ pts.T if 'samples_cache' in dir() else None
            
            # Simpler: just try random perturbations and keep if better
            best_local = pts[idx].copy()
            best_local_radius = _covering_radius_samples(pts, n_samples // 2)
            
            # Try a few random perturbations
            for _ in range(5):
                elapsed = time.time() - start_time
                if elapsed > time_limit - 5.0:
                    break
                
                # Random unit vector perturbation
                dv = np.random.randn(3)
                dv /= np.linalg.norm(dv)
                new_pt = _normalize(pts[idx] + step_size * dv)
                pts_new = pts.copy()
                pts_new[idx] = new_pt
                
                new_radius = _covering_radius_samples(pts_new, n_samples // 2)
                if new_radius < best_local_radius - 1e-6:
                    best_local = new_pt
                    best_local_radius = new_radius
                    improved = True
            
            pts[idx] = best_local
        
        # Check overall improvement
        current_radius = _covering_radius_samples(pts, n_samples)
        if current_radius < best_radius - 1e-6:
            best_radius = current_radius
            best_pts = pts.copy()
        
        if not improved:
            # Try smaller step size
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to reconsider. The sample-based approach is slow. Let me use a more efficient method with precomputed samples and vectorized operations.

