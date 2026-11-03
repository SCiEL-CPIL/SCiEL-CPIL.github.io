"""
Optimized sphere covering with 130 points — Spectral Force-Directed Optimization.
Strategy:
1. Initialize with diverse Fibonacci spirals.
2. Refine using force-directed layout minimizing variance of pairwise distances (spectral uniformity).
   This avoids ConvexHull in the main loop, allowing thousands of fast iterations.
3. Final polish with a few Lloyd steps using ConvexHull to fix local covering radius defects.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    return _covering_radius(pts)

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spectral_force_step(pts, lr=0.01, beta=10.0):
    """
    Force-directed step to minimize variance of pairwise angular distances.
    Uses a smooth approximation of the repulsive force to avoid singularities.
    """
    pts = _normalize(pts.copy())
    
    # Compute pairwise dot products (Gram matrix)
    G = pts @ pts.T
    np.fill_diagonal(G, 0.0)
    
    # Pairwise angular distances
    # Use arccos for exactness, but clip for safety
    dists = np.arccos(np.clip(G, -1.0, 1.0))
    
    # Target distance: for N points on sphere, average distance is ~sqrt(2/pi) * sqrt(N)? 
    # Actually, for uniform distribution, the mean squared distance is 2.
    # We want to minimize the variance of the distances.
    # Gradient of variance w.r.t. pts[i]:
    # Var = (1/M) * sum (d_ij - d_mean)^2
    # This is complex to compute directly in O(N^2) per point for gradient.
    # Instead, we use a heuristic force: 
    # If d_ij < target, repel. If d_ij > target, attract (weakly).
    # Target distance for 130 points: 
    # Area per point = 4*pi/130. Radius of cap ~ sqrt(4*pi/130 / pi) = sqrt(4/130) ~ 0.175 rad.
    # Chord distance ~ 2*sin(theta/2).
    # Let's use a target angular distance theta_target.
    # A good estimate for optimal covering is related to the first zero of spherical harmonics?
    # Simpler: use the mean distance as a dynamic target, or a fixed one.
    # Mean distance for uniform on sphere is pi/2? No, that's for random pairs.
    # For 130 points, the nearest neighbor distance is ~ 11.3 deg = 0.197 rad.
    # Let's use a target of 0.2 rad.
    
    target_d = 0.20
    
    # Compute forces
    # Force on i from j: proportional to (d_ij - target_d) * direction
    # Direction is (pts[j] - pts[i]) projected tangent to sphere at i.
    
    forces = np.zeros_like(pts)
    
    # Vectorized computation of forces
    # diff = pts[None, :, :] - pts[:, None, :] # (N, N, 3)
    # This is 130*130*3 = 50700 floats. Very fast.
    
    diff = pts[None, :, :] - pts[:, None, :]
    
    # We want to push points apart if too close, pull together if too far.
    # Force magnitude w_ij = (d_ij - target_d)
    # But we need to be careful with the derivative of arccos.
    # Let's use a simpler potential: U = (d_ij - target_d)^2
    # grad U / grad pts_i = 2 * (d_ij - target_d) * grad d_ij / grad pts_i
    # grad d_ij / grad pts_i = - (pts_j - pts_i) / |pts_j - pts_i| (chord direction)
    # Wait, d_ij is angular distance. 
    # Let's stick to the heuristic: 
    # Force direction: pts_j - pts_i (chord vector)
    # Magnitude: (d_ij - target_d)
    
    # Compute chord distances
    chord_dists = np.linalg.norm(diff, axis=2)
    chord_dists[np.diag_indices_from(chord_dists)] = 1.0 # Avoid div by zero
    
    # Angular distances
    # d_ij = 2 * arcsin(chord_d / 2)
    # Let's use the angular distances computed earlier for consistency in target
    
    # Actually, let's just use the chord distance for the force calculation, it's smoother.
    # Target chord distance for 0.2 rad is 2*sin(0.1) ~ 0.199.
    target_chord = 2.0 * np.sin(target_d / 2.0)
    
    # Weight matrix W_ij = (chord_d_ij - target_chord)
    # If too close (chord < target), W is negative -> Force should be repulsive (push apart).
    # Force on i from j should be in direction (pts_i - pts_j) if repulsive.
    # So Force_i = sum_j W_ij * (pts_i - pts_j) ? 
    # If W is negative (too close), we want force in direction (pts_i - pts_j) (away).
    # So Force_i = sum_j -W_ij * (pts_i - pts_j) = sum_j W_ij * (pts_j - pts_i)
    
    W = chord_dists - target_chord
    np.fill_diagonal(W, 0.0)
    
    # Force_i = sum_j W_ij * (pts_j - pts_i)
    # diff is pts_j - pts_i (row i, col j is pts_j - pts_i? No.
    # diff[i, j] = pts[j] - pts[i]. Correct.
    
    forces = np.sum(W[:, :, None] * diff, axis=1)
    
    # Project forces to tangent space
    # Tangent space at pts[i] is orthogonal to pts[i].
    # F_tan = F - (F . pts_i) pts_i
    dot_prod = np.sum(forces * pts, axis=1, keepdims=True)
    forces_tan = forces - dot_prod * pts
    
    # Update
    new_pts = pts + lr * forces_tan
    new_pts = _normalize(new_pts)
    
    return new_pts

def _local_refine(pts, max_iters=50):
    """
    Final polish using Lloyd's algorithm with ConvexHull.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    
    for _ in range(max_iters):
        if not _time_ok(5.0):
            break
            
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        CC_list = []
        facet_indices = []
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            v1, v2 = b - a, c - a
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            x = cross / cnorm
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            CC_list.append(x)
            facet_indices.append(simplex)

        if not CC_list:
            break

        CC = np.array(CC_list)
        F = len(CC)
        
        new_pts = pts.copy()
        for i in range(N):
            mask = np.any(np.array(facet_indices) == i, axis=1)
            if not np.any(mask):
                continue
            cc_i = CC[mask]
            # Move towards the circumcenter of the worst facet? 
            # Or average of adjacent circumcenters.
            # Let's use the circumcenter of the facet with the largest circumradius adjacent to i.
            dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
            dists = np.arccos(dots)
            worst_idx = np.argmax(dists)
            target = cc_i[worst_idx]
            
            # Blend with current position
            new_pts[i] = 0.9 * pts[i] + 0.1 * target
            
        new_pts = _normalize(new_pts)
        cr = _covering_radius_fast(new_pts)
        
        if cr < best_r:
            best_r = cr
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Backtrack or stop if no improvement
            break
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    best_pts = None
    best_r = np.pi

    # Generate diverse initial configurations
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271, 888]:
        inits.append(_fibonacci_spiral(N, jitter=0.02, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    
    # Sort by initial covering radius
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])

    for r0, p0 in init_scores:
        if not _time_ok(15.0):
            break

        # Phase 1: Spectral Force-Directed Optimization
        # Run many fast iterations
        pts = p0.copy()
        lr = 0.05
        for it in range(2000):
            if not _time_ok(10.0):
                break
            pts = _spectral_force_step(pts, lr=lr)
            # Decay learning rate
            lr *= 0.999
            
        cr_spec = _covering_radius_fast(pts)
        
        # Phase 2: Local Refinement
        pts_ref, cr_ref = _local_refine(pts, max_iters=100)
        
        if cr_ref < best_r:
            best_r = cr_ref
            best_pts = pts_ref.copy()
            
        # If spectral phase was good, maybe refine that too?
        if cr_spec < cr_ref:
             pts_spec_ref, cr_spec_ref = _local_refine(pts, max_iters=50)
             if cr_spec_ref < best_r:
                 best_r = cr_spec_ref
                 best_pts = pts_spec_ref.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
