"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Vectorized pairwise distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)
    wmin = np.min(wall_dists)
    
    return min(dmin / 2.0, wmin)

def _relax(pts, time_limit):
    """
    Perform soft-sphere molecular dynamics relaxation.
    Spheres repel each other and the walls. We integrate motion until the system
    is 'jammed' (forces are small) or time runs out.
    """
    pts = pts.copy()
    n = len(pts)
    
    # Parameters
    dt = 0.005          # Time step
    damping = 0.95      # Velocity damping
    repulsion_k = 50.0  # Repulsion strength
    max_iter = 5000     # Max iterations
    
    start_time = time.time()
    
    velocities = np.zeros_like(pts)
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        
        forces = np.zeros_like(pts)
        
        # 1. Pairwise repulsion
        # Only consider pairs that are close to improve speed? 
        # For N=50, O(N^2) is 2500 ops, which is fast. Vectorize it.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero on diagonal
        
        dists = np.sqrt(dists_sq)
        
        # Repulsive force is proportional to 1/d^2 or similar.
        # We want them to separate.
        # F_ij = k * (diff_ij) / d_ij^3  (points from j to i)
        # If d < threshold, apply force.
        # To make it stable, we can use a soft potential like 1/d^2.
        
        # Force on i from j: k * (pts[i] - pts[j]) / d^3
        # This is equivalent to gradient of -k/d^2? No, grad(1/d) = -1/d^2 * r_hat.
        # Let's use F = k * diff / dists_sq. This is 1/r^2 force magnitude.
        
        # Safe division
        safe_dists_sq = np.where(dists_sq > 1e-6, dists_sq, 1e-6)
        
        # Force matrix: F[i,j] is force on i due to j
        # F[i,j] = k * (pts[i] - pts[j]) / dists_sq[i,j]
        # We sum over j for each i.
        
        # Vectorized:
        # diff is (N, N, 3)
        # safe_dists_sq is (N, N)
        # forces = np.sum(k * diff / safe_dists_sq[:, :, np.newaxis], axis=1)
        
        # However, we only want repulsion if they are "close". 
        # In soft sphere MD, usually all pairs repel if overlapping or close.
        # For packing, we want them to settle into a local minimum of min distance.
        # Using 1/r^2 repulsion everywhere works well for spreading out.
        
        forces += repulsion_k * np.sum(diff / safe_dists_sq[:, :, np.newaxis], axis=1)
        
        # 2. Wall repulsion
        # Distance to wall: w = 0.5 - |x|
        # Force pushes away from wall if w < threshold.
        # Let's say if w < 0.1, apply force proportional to (0.1 - w)^2 or similar.
        # Or simply: if close to wall, push inwards.
        
        wall_dists = 0.5 - np.abs(pts)
        # If wall_dist is small, we want a large force inward.
        # F_wall = k_wall * (1/wall_dist^2) * sign(x)  (if inside)
        # But if outside, wall_dist is negative. 
        # Let's clamp wall_dist to a small positive value for force calculation, 
        # but the force direction should always be towards center (0,0,0) relative to wall.
        
        # Actually, simpler: 
        # For each coord x:
        # If x > 0, wall is at 0.5. dist = 0.5 - x.
        # If dist < 0.05, force_x -= k_wall * (0.05 - dist)^2 (push left)
        # If x < 0, wall is at -0.5. dist = 0.5 + x.
        # If dist < 0.05, force_x += k_wall * (0.05 - dist)^2 (push right)
        
        # Let's use a smooth potential for walls too.
        # V_wall = k_wall / (0.5 - |x|)^2 ? No, singular.
        # Let's use: if d_wall < d_cut, F = k_wall * (d_cut - d_wall)
        
        d_cut = 0.1
        # Vectorized wall force
        # For each point p:
        # dx = p[0], dy = p[1], dz = p[2]
        # dist_x_pos = 0.5 - dx. If dx > 0 and dist_x_pos < d_cut, Fx -= k*(d_cut - dist_x_pos)
        # dist_x_neg = 0.5 + dx. If dx < 0 and dist_x_neg < d_cut, Fx += k*(d_cut - dist_x_neg)
        
        # This is easier to do with loops for clarity and safety, or careful vectorization.
        # N=50 is small, loop is fine.
        
        for i in range(n):
            for dim in range(3):
                val = pts[i, dim]
                if val >= 0:
                    dist = 0.5 - val
                    if dist < d_cut:
                        forces[i, dim] -= repulsion_k * (d_cut - dist)
                else:
                    dist = 0.5 + val
                    if dist < d_cut:
                        forces[i, dim] += repulsion_k * (d_cut - dist)
                        
        # 3. Update velocities and positions
        velocities = velocities * damping + forces * dt
        pts += velocities * dt
        
        # 4. Clamp to inside cube (strictly) to prevent escape
        pts = np.clip(pts, -0.499, 0.499)
        
        # Optional: Stop if forces are very small (converged)
        # if np.max(np.abs(forces)) < 1e-4:
        #     break
        
    return pts

def _local_polish(pts, time_limit):
    """
    Gradient-based polishing of the minimum distance.
    Identify the bottleneck (closest pair or wall) and move points to increase it.
    """
    pts = pts.copy()
    n = len(pts)
    start_time = time.time()
    
    # We will try to increase the minimum distance.
    # Use a simple hill climbing on the score.
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    step_size = 0.005
    
    while time.time() < time_limit:
        # Compute score components
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        dmin = np.min(dists)
        
        wall_dists = 0.5 - np.abs(pts)
        wmin = np.min(wall_dists)
        
        current_r = min(dmin / 2.0, wmin)
        
        if current_r > best_score:
            best_score = current_r
            best_pts = pts.copy()
            
        # Determine bottleneck
        if dmin / 2.0 < wmin:
            # Closest pair
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            if i > j: i, j = j, i
            
            d_vec = pts[i] - pts[j]
            norm = np.linalg.norm(d_vec)
            if norm > 1e-10:
                d_hat = d_vec / norm
                # Move i out, j out
                pts[i] += d_hat * step_size
                pts[j] -= d_hat * step_size
        else:
            # Closest wall
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            if pts[i, k] > 0:
                pts[i, k] -= step_size
            else:
                pts[i, k] += step_size
                
        pts = np.clip(pts, -0.499, 0.499)
        
        # Reduce step size slightly to converge
        step_size *= 0.99
        
        if step_size < 1e-6:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Strategy:
    # 1. Generate a good initial configuration (Halton/Stratified)
    # 2. Run Soft-Sphere MD to spread them out
    # 3. Run Local Polish to tighten the minimum distance
    
    # 1. Initialization: Halton sequence for low discrepancy
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
        
    bases = [2, 3, 5]
    init_pts = np.array([[halton(i, b) for b in bases] for i in range(1, N + 1)])
    # Map to [-0.49, 0.49]
    init_pts = init_pts * 0.98 - 0.49
    
    # 2. MD Relaxation
    # Give it most of the time
    md_time_budget = 100
    md_deadline = start_time + md_time_budget
    
    relaxed_pts = _relax(init_pts, md_deadline)
    
    # 3. Polish
    polish_deadline = start_time + 130
    polished_pts, score = _local_polish(relaxed_pts, polish_deadline)
    
    if score > best_score:
        best_score = score
        best_pts = polished_pts.copy()
        
    # 4. Try a second restart with random initialization if time permits
    if time.time() < start_time + 110:
        random_pts = np.random.uniform(-0.4, 0.4, (N, 3))
        md_deadline2 = start_time + 120
        relaxed_pts2 = _relax(random_pts, md_deadline2)
        
        polish_deadline2 = start_time + 135
        polished_pts2, score2 = _local_polish(relaxed_pts2, polish_deadline2)
        
        if score2 > best_score:
            best_score = score2
            best_pts = polished_pts2.copy()
            
    # Fallback
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
