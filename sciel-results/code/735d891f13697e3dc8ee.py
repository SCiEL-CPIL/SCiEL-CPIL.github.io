"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, ConvexHull

def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    # Wall clearance
    wall_clearance = np.min(np.minimum(np.abs(centers), 0.5 - np.abs(centers)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(wall_clearance, min_pair_dist / 2.0)


def _lloyd_step(centers, iterations=200, tol=1e-10):
    """
    Lloyd's algorithm variant for sphere packing in a cube.
    Move each center toward the centroid of its Voronoi cell clipped by the cube.
    """
    centers = centers.copy()
    cube_min = -0.5
    cube_max = 0.5
    
    for it in range(iterations):
        try:
            vor = Voronoi(centers)
        except Exception:
            break
        
        new_centers = np.zeros_like(centers)
        max_move = 0.0
        
        for i in range(len(centers)):
            # Get the Voronoi region vertices for point i
            region_idx = vor.point_region[i]
            region_vertices = vor.regions[region_idx]
            
            if len(region_vertices) == 0 or -1 in region_vertices:
                # Unbounded region (shouldn't happen for interior points in a bounded domain,
                # but handle gracefully by keeping the point)
                new_centers[i] = centers[i]
                continue
            
            verts = vor.vertices[region_vertices]
            
            # Clip to the cube [-0.5, 0.5]^3
            # A simple approach: for each vertex, clamp to the cube
            # But this isn't a proper clipping. Let's use a more robust method.
            
            # Actually, for Lloyd's in a bounded domain, we should compute the centroid
            # of the intersection of the Voronoi cell with the cube.
            # A practical approximation: clamp vertices to the cube and take centroid.
            # This is an approximation but works reasonably well.
            
            clipped_verts = np.clip(verts, cube_min, cube_max)
            
            # Also add the center itself as a fallback if region is degenerate
            if len(clipped_verts) > 0:
                centroid = np.mean(clipped_verts, axis=0)
            else:
                centroid = centers[i]
            
            # Ensure the new position stays within the cube
            centroid = np.clip(centroid, cube_min, cube_max)
            new_centers[i] = centroid
        
        move = np.max(np.linalg.norm(new_centers - centers, axis=1))
        max_move = move
        centers = new_centers
        
        if move < tol:
            break
    
    return centers


def _gradient_ascent_step(centers, step_size=0.01, iterations=500):
    """
    Gradient ascent on the soft-min objective using a log-sum-exp approximation.
    r = min(min_pairwise_dist/2, wall_clearance)
    We use a smooth approximation and do gradient ascent.
    """
    centers = centers.copy()
    beta = 100.0  # Sharpness of the soft-min
    
    for it in range(iterations):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Wall clearance for each sphere
        wall_clear = np.minimum(np.abs(centers), 0.5 - np.abs(centers))
        min_wall = np.min(wall_clear, axis=1)
        
        # Pairwise min distance for each sphere (distance to nearest neighbor)
        pair_min = np.min(dists, axis=1)
        
        # The radius is min of (pair_min/2 for each i) and (min_wall for each i)
        # Actually, the global radius is min over all i of min(pair_min[i]/2, min_wall[i])
        
        # Soft-min over all constraints
        # Constraints: for each i, pair_min[i]/2 and min_wall[i]
        constraints = np.concatenate([pair_min / 2.0, min_wall])
        
        # Soft-min via log-sum-exp
        m = np.min(constraints)
        log_sum = m + np.log(np.sum(np.exp(-beta * (constraints - m))))
        soft_min = -log_sum / beta
        
        # Gradient of soft-min w.r.t. centers
        weights = np.exp(-beta * (constraints - m))
        weights /= np.sum(weights)
        
        grad = np.zeros_like(centers)
        
        # Gradient w.r.t. pair_min[i]/2
        for i in range(N):
            # d(pair_min[i])/d(centers) - this is tricky because pair_min[i] is min over j
            # Use the weight for constraint i (pair part)
            w_pair = weights[i]
            # The nearest neighbor j for sphere i
            j = np.argmin(dists[i])
            # d(pair_min[i])/d(centers[i]) = 2*(centers[i] - centers[j]) / dist[i,j]
            # d(pair_min[i])/d(centers[j]) = 2*(centers[j] - centers[i]) / dist[i,j]
            d_ij = dists[i, j]
            if d_ij > 1e-12:
                dir_ij = (centers[i] - centers[j]) / d_ij
                grad[i] += w_pair * dir_ij
                grad[j] -= w_pair * dir_ij
        
        # Gradient w.r.t. min_wall[i]
        for i in range(N):
            w_wall = weights[N + i]
            # min_wall[i] = min over k of (0.5 - |centers[i,k]|)
            # The active coordinate is the one with min |centers[i,k]|
            abs_c = np.abs(centers[i])
            k = np.argmin(abs_c)
            sign = np.sign(centers[i, k])
            if abs_c[k] < 1e-12:
                sign = 1.0
            # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
            # So d(min_wall)/d(c_i) has component -sign at index k
            g = np.zeros(3)
            g[k] = -sign
            grad[i] += w_wall * g
        
        # Gradient ascent
        centers += step_size * grad
        
        # Clip to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Adaptive step size: reduce if radius decreases
        if it % 50 == 0:
            step_size *= 0.95
    
    return centers


def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0  # 140s deadline for safety
    
    # Start with a reasonable initial configuration
    # Use a jittered grid as starting point
    m = 4  # 4x4x4 = 64 > 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Lloyd's algorithm iterations
    while time.time() < deadline - 30:
        centers = _lloyd_step(centers, iterations=50, tol=1e-10)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Gradient ascent polish
    while time.time() < deadline - 10:
        centers = best_centers.copy()
        centers = _gradient_ascent_step(centers, step_size=0.005, iterations=200)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        else:
            break
    
    # Final Lloyd's polish
    while time.time() < deadline - 2:
        centers = _lloyd_step(best_centers, iterations=50, tol=1e-12)
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        else:
            break
    
    return best_centers
# EVOLVE-BLOCK-END
