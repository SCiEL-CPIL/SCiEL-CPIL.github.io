import numpy as np

N = 50

def construct():
    """
    Uses Force-Directed Relaxation to maximize the minimum radius of 50 circles
    in a unit square. This approach treats the circles as particles repelling
    each other and the walls, using a cooling schedule to settle into a
    high-quality packing configuration.
    """
    rng = np.random.RandomState(42)
    
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        d_left = pts[:, 0] + 0.5
        d_right = 0.5 - pts[:, 0]
        d_bottom = pts[:, 1] + 0.5
        d_top = 0.5 - pts[:, 1]
        
        min_wall_dist = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    def run_relaxation(initial_points, steps=10000, dt_start=0.01, dt_end=0.0001, damping=0.9):
        points = initial_points.copy()
        velocities = np.zeros_like(points)
        best_r = compute_radius(points)
        best_points = points.copy()
        
        # Cooling schedule
        cooling_rate = (dt_end / dt_start) ** (1.0 / steps)
        
        # Stiffness parameter for repulsion
        # We want to push apart any pair closer than 2 * current_best_r
        # But initially we don't know the best r.
        # We can use a dynamic target radius.
        
        for i in range(steps):
            dt = dt_start * (cooling_rate ** i)
            
            # Calculate forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            # We only care about pairs that are close.
            # To be efficient, we can check all pairs but only apply force if close.
            # For N=50, N^2/2 = 1225 pairs. This is fast.
            
            for i_idx in range(N):
                for j_idx in range(i_idx + 1, N):
                    diff = points[i_idx] - points[j_idx]
                    dist = np.linalg.norm(diff)
                    
                    # Target distance: we want them to be at least 2 * best_r apart
                    # If they are closer, push apart.
                    # Using a spring force: F = k * (target - dist) * direction
                    # target = 2 * best_r
                    target = 2.0 * best_r
                    if dist < target and dist > 1e-8:
                        # Force magnitude
                        # Stiffness k can be scaled with dt to ensure stability
                        k = 10.0
                        f_mag = k * (target - dist)
                        # Direction
                        f_dir = diff / dist
                        # Apply equal and opposite forces
                        forces[i_idx] += f_mag * f_dir
                        forces[j_idx] -= f_mag * f_dir
                    elif dist <= 1e-8:
                        # Coincident, push randomly
                        forces[i_idx] += rng.normal(0, 1, 2)
                        forces[j_idx] -= rng.normal(0, 1, 2)

            # Wall repulsion
            # Push away from walls if closer than best_r
            # Left wall: x < -0.5 + best_r
            # Right wall: x > 0.5 - best_r
            # Bottom wall: y < -0.5 + best_r
            # Top wall: y > 0.5 - best_r
            
            k_wall = 10.0
            
            # Left
            dist_left = points[:, 0] + 0.5
            overlap_left = np.maximum(0, best_r - dist_left)
            forces[:, 0] += k_wall * overlap_left
            
            # Right
            dist_right = 0.5 - points[:, 0]
            overlap_right = np.maximum(0, best_r - dist_right)
            forces[:, 0] -= k_wall * overlap_right
            
            # Bottom
            dist_bottom = points[:, 1] + 0.5
            overlap_bottom = np.maximum(0, best_r - dist_bottom)
            forces[:, 1] += k_wall * overlap_bottom
            
            # Top
            dist_top = 0.5 - points[:, 1]
            overlap_top = np.maximum(0, best_r - dist_top)
            forces[:, 1] -= k_wall * overlap_top
            
            # Update velocities
            velocities = velocities * damping + forces * dt
            
            # Update positions
            points += velocities * dt
            
            # Clamp to boundaries
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
            
            # Calculate current radius
            curr_r = compute_radius(points)
            
            if curr_r > best_r:
                best_r = curr_r
                best_points = points.copy()
                
        return best_points, best_r

    # Generate multiple initial configurations
    # 1. Grid 7x8
    # 2. Grid 8x7
    # 3. Grid with jitter
    
    best_points_overall = None
    best_r_overall = 0.0
    
    for m in [7, 8]:
        for n in [7, 8]:
            if m * n < N:
                continue
            ticks_x = (np.arange(m) + 0.5) / m - 0.5
            ticks_y = (np.arange(n) + 0.5) / n - 0.5
            grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
            points = grid[:N].astype(float)
            # Add small jitter
            points += rng.uniform(-0.001, 0.001, size=points.shape)
            points = np.clip(points, -0.49, 0.49)
            
            pts, r = run_relaxation(points, steps=5000)
            if r > best_r_overall:
                best_r_overall = r
                best_points_overall = pts

    return best_points_overall
