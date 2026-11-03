import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 5.0

def compute_radius(points):
    """
    Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    Returns 0 if invalid.
    """
    if points is None or points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check inside container
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms >= 1.0):
        return 0.0
    
    # Check non-overlapping
    # Efficient pairwise distance calculation
    # For N=50, 50x50 matrix is small
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, np.min(1.0 - norms))
    
    if r <= 0:
        return 0.0
        
    return r

def smooth_objective_and_grad(x_flat, beta=50.0):
    """
    Smooth surrogate for the max-min problem.
    We want to maximize r = min(d_ij/2, 1-|c_i|).
    This is equivalent to minimizing -r.
    Using log-sum-exp to create a smooth approximation of the min function.
    
    f(x) = - (1/beta) * log( sum_{constraints} exp(-beta * constraint_val) )
    
    Constraints:
    1. d_ij/2 - r  (for all i<j)
    2. 1 - |c_i| - r (for all i)
    
    Actually, a simpler smooth approximation for maximizing the minimum distance:
    Let D be the vector of all pairwise distances / 2 and boundary distances.
    We want to maximize min(D).
    Approximate min(D) by -1/beta * log(sum(exp(-beta * D))).
    So we minimize 1/beta * log(sum(exp(-beta * D))).
    
    Gradient calculation:
    Let S = sum(exp(-beta * D_k))
    f = (1/beta) * log(S)
    df/dx = (1/beta) * (1/S) * sum( exp(-beta * D_k) * (-beta * dD_k/dx) )
    df/dx = - sum( (exp(-beta * D_k) / S) * dD_k/dx )
    Let w_k = exp(-beta * D_k) / S. These are weights summing to 1.
    df/dx = - sum( w_k * dD_k/dx )
    """
    pts = x_flat.reshape(-1, 2)
    
    # Calculate all distances
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Avoid sqrt for gradient stability? No, we need d(dist)/dx.
    # dist_ij = sqrt(sum(diff^2))
    # d(dist_ij)/dx_i = diff_ij / dist_ij
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - norms
    # d(boundary_i)/dx_i = -pts_i / norm_i
    
    # We need to construct the weighted sum of gradients.
    # To avoid storing all 1225+50 distances explicitly in memory if N was huge, 
    # but N=50 is small.
    
    # Pairwise part
    # dists matrix has 0 on diagonal. We ignore diagonal.
    # We need d(dist_ij)/dx_k.
    # Only non-zero for k=i or k=j.
    
    # Initialize gradient accumulator
    grad = np.zeros_like(pts)
    
    # Calculate weights for pairwise
    # dists is 50x50.
    # exp(-beta * dists)
    # We must handle dists=0 (i=j). Set to 0 weight.
    exp_terms = np.exp(-beta * dists_sq) # Wait, dists_sq is squared distance.
    # The objective is based on distance, not squared distance.
    # Let's use actual distances.
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, 1e10) # Large value so exp(-beta*large) ~ 0
    
    exp_pair = np.exp(-beta * dists)
    # Zero out diagonal
    np.fill_diagonal(exp_pair, 0.0)
    
    sum_pair = np.sum(exp_pair)
    
    # Boundary part
    exp_bound = np.exp(-beta * boundary_dists)
    sum_bound = np.sum(exp_bound)
    
    total_sum = sum_pair + sum_bound
    
    if total_sum == 0:
        return 0.0, grad
        
    # Weights
    w_pair = exp_pair / total_sum
    w_bound = exp_bound / total_sum
    
    # Objective value
    obj_val = (1.0 / beta) * np.log(total_sum)
    
    # Gradient of pairwise part
    # sum_{i<j} w_ij * d(dist_ij)/dx
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
    
    # Vectorized gradient for pairwise
    # grad_pair = sum_j w_ij * (x_i - x_j)/d_ij  - sum_i w_ij * (x_i - x_j)/d_ij
    # Let's compute term by term or vectorized.
    
    # d(dist_ij)/dx_k is sparse.
    # grad[i] += sum_j w_ij * (pts[i] - pts[j]) / dists[i,j]
    # grad[j] += sum_i w_ij * (pts[j] - pts[i]) / dists[i,j]
    
    # Vectorized:
    # G = w_pair * (1/dists)  # 50x50 matrix
    # G_ii = 0
    # grad_pair[i] = sum_j G[i,j] * (pts[i] - pts[j])
    #             = pts[i] * sum_j G[i,j] - sum_j G[i,j] * pts[j]
    
    G = w_pair / dists
    np.fill_diagonal(G, 0.0)
    
    # sum_j G[i,j] is row sum
    row_sums = np.sum(G, axis=1)
    # sum_j G[i,j] * pts[j] is G @ pts
    G_pts = G @ pts
    
    grad_pair = pts * row_sums[:, np.newaxis] - G_pts
    
    # Gradient of boundary part
    # sum_i w_i * d(1 - |x_i|)/dx_i = sum_i w_i * (-x_i / |x_i|)
    # d(|x_i|)/dx_i = x_i / |x_i|
    # So term is - w_i * x_i / |x_i|
    
    # Handle |x_i| = 0
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    grad_bound = -w_bound[:, np.newaxis] * (pts / safe_norms[:, np.newaxis])
    
    total_grad = grad_pair + grad_bound
    
    return obj_val, total_grad

def initial_rings():
    """
    Generate initial positions based on the known optimal packing structure:
    1 center, 6, 12, 18, 13 rings.
    """
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    
    # Ring 1: 6 points
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    # Ring 2: 12 points
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi/12 # staggered
        pts[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
    # Ring 3: 18 points
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
        
    # Ring 4: 13 points
    r4 = 0.95
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi/13
        pts[37 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
        
    return pts

def initial_hexagonal():
    """
    Generate a hexagonal lattice clipped to the unit circle.
    """
    pts = []
    # Estimate spacing based on r ~ 0.125
    # distance between centers = 2r ~ 0.25
    # Hex grid spacing a = 0.25
    a = 0.25
    # Cover range -1 to 1
    range_val = 1.2
    nx = int(range_val / a) + 1
    ny = int(range_val / (a * np.sqrt(3)/2)) + 1
    
    for i in range(-nx, nx+1):
        for j in range(-ny, ny+1):
            x = i * a + (j % 2) * a / 2
            y = j * a * np.sqrt(3) / 2
            if np.sqrt(x*x + y*y) < 1.0:
                pts.append([x, y])
                
    # If we have more than N, take the closest N to origin? Or just first N?
    # Better to take N closest to center to ensure high density in middle?
    # Or just take N random ones?
    # Let's sort by distance to center and take N closest to ensure they fit.
    if len(pts) > N:
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
    else:
        # Pad with random points if too few (unlikely)
        pts = np.array(pts)
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.9, 0.9, 2)])
            
    return pts

def push_apart(points, iterations=50):
    """
    Simple local repair to ensure no overlaps and inside boundary.
    """
    pts = points.copy()
    for _ in range(iterations):
        if _check_deadline():
            break
        # Check validity
        r = compute_radius(pts)
        if r > 0:
            # Still, let's try to maximize r by pushing apart
            pass
            
        # Calculate forces
        forces = np.zeros_like(pts)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, 1e10)
        
        # Target distance 2 * current_r + epsilon?
        # Let's just push apart if dist < 2*r_est
        # Estimate r from current min dist
        min_dist = np.min(dists)
        target_r = min_dist / 2.0 + 0.01
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 2 * target_r:
                    # Push apart
                    diff = pts[i] - pts[j]
                    if d > 1e-10:
                        force = diff / d * (2 * target_r - d)
                        forces[i] += force
                        forces[j] -= force
        
        # Boundary force
        norms = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if norms[i] > 1.0 - target_r:
                # Push in
                forces[i] -= pts[i] / norms[i] * (norms[i] - (1.0 - target_r))
        
        # Update
        pts += forces * 0.5
        
        # Project inside
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask] * (1.0 - 1e-6)
            
    return pts

def construct():
    best_pts = None
    best_r = 0.0
    
    # Initial candidates
    candidates = []
    
    # 1. Rings
    pts_rings = initial_rings()
    candidates.append(pts_rings)
    
    # 2. Hexagonal
    pts_hex = initial_hexagonal()
    candidates.append(pts_hex)
    
    # 3. Random (a few)
    for _ in range(3):
        pts_rand = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(pts_rand, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts_rand[mask] = pts_rand[mask] / norms[mask] * (1.0 - 1e-6)
        candidates.append(pts_rand)
    
    # Optimize each candidate
    for init_pts in candidates:
        if _check_deadline():
            break
            
        x0 = init_pts.flatten()
        
        try:
            # Use L-BFGS-B
            # The objective is smooth (log-sum-exp)
            # We want to minimize the surrogate which approximates -r (scaled)
            # Actually, our smooth_objective returns positive value (1/beta * log(sum)).
            # Minimizing this maximizes the min distance.
            
            res = minimize(
                smooth_objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, # We return (f, g)
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxfun': 2000
                }
            )
            
            opt_pts = res.x.reshape(-1, 2)
            
            # Repair to ensure validity
            opt_pts = push_apart(opt_pts, iterations=20)
            
            r = compute_radius(opt_pts)
            
            if r > best_r:
                best_r = r
                best_pts = opt_pts
                
        except Exception as e:
            pass
            
    # If we didn't find a valid one, fallback to a known valid configuration
    if best_pts is None or best_r == 0.0:
        # Use the ring initialization as a safe fallback, even if not optimized
        best_pts = initial_rings()
        # Ensure it's valid by scaling if necessary
        norms = np.linalg.norm(best_pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0.9:
            best_pts = best_pts * (0.9 / max_norm)
            
    # Final check and return
    # Ensure strictly inside
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / norms[mask] * (1.0 - 1e-6)
        
    return best_pts

if __name__ == "__main__":
    # Simple self-test
    t0 = time.time()
    pts = construct()
    t1 = time.time()
    r = compute_radius(pts)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.6f}")
