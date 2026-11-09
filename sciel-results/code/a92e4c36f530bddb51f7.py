import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Helper to calculate the score (radius) of a configuration
    def calc_radius(pts):
        # Min distance between centers / 2
        if len(pts) < 2:
            return np.inf
        # Compute pairwise distances
        # Using broadcasting for speed
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Mask out diagonal
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Min distance to boundary
        dist_to_center = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - dist_to_center)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def get_hex_init(r_target):
        # Generate a hexagonal grid with spacing 2*r_target
        # We want points inside radius 1-r_target
        # Spacing s = 2 * r_target
        s = 2 * r_target
        if s <= 0:
            s = 0.01
        # Row height
        h = s * np.sqrt(3) / 2
        max_r = 1.0 - r_target
        num_rows = int(max_r / h) + 2
        pts = []
        for j in range(-num_rows, num_rows + 1):
            y = j * h
            if abs(y) > max_r:
                continue
            offset = s / 2 if j % 2 != 0 else 0
            num_cols = int((max_r - abs(y)) / s) + 2
            for i in range(-num_cols, num_cols + 1):
                x = i * s + offset
                if x*x + y*y <= max_r*max_r:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) < N:
            # If hex grid doesn't have enough points, pad with random
            while len(pts) < N:
                angle = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, max_r)
                pts = np.vstack([pts, [rad*np.cos(angle), rad*np.sin(angle)]])
        # Select N closest to center
        d = np.linalg.norm(pts, axis=1)
        idx = np.argsort(d)
        return pts[idx[:N]]

    def get_ring_init(r_target):
        # Center + rings
        pts = [[0, 0]]
        # 50 points: 1 + 6 + 12 + 18 + 13 = 50?
        # 1 + 6 = 7
        # 7 + 12 = 19
        # 19 + 18 = 37
        # 37 + 13 = 50
        ring_counts = [6, 12, 18, 13]
        # Estimate ring radii based on packing density
        # Area of circle N=50 with radius r is approx 50 * pi * r^2
        # Area of container is pi * 1^2
        # 50 * r^2 ~ 1 => r ~ 0.14 (theoretical max)
        # Best known r ~ 0.126
        # Let's place rings at radii proportional to sqrt(k)
        max_r = 1.0 - r_target
        total_rings = len(ring_counts)
        for k, count in enumerate(ring_counts):
            # Radius for ring k+1
            # Approximate radius such that area covered matches
            # r_k ~ sqrt((k+1)/N) * max_r
            ring_r = np.sqrt((k + 1) / N) * max_r
            # Adjust for hexagonal packing efficiency if needed, but simple radial is a good init
            for m in range(count):
                angle = 2 * np.pi * m / count + (np.pi / count if k % 2 else 0)
                pts.append([ring_r * np.cos(angle), ring_r * np.sin(angle)])
        pts = np.array(pts[:N])
        return pts

    def get_random_init(r_target):
        max_r = 1.0 - r_target
        pts = []
        for _ in range(N):
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.sqrt(np.random.uniform(0, max_r**2))
            pts.append([rad * np.cos(angle), rad * np.sin(angle)])
        return np.array(pts)

    def relax(pts, r_target, max_iters=10000, time_limit=None):
        """
        Relax points to satisfy constraints for radius r_target.
        Returns (pts, success, achieved_r)
        """
        pts = pts.copy()
        max_r = 1.0 - r_target
        min_dist = 2.0 * r_target
        
        for it in range(max_iters):
            if time_limit and (time.time() - start_time > time_limit):
                break
            
            moved = False
            # Pairwise repulsion
            # Vectorized computation of distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Identify violations
            # We want dist < min_dist
            mask = dist < min_dist
            # Exclude diagonal
            np.fill_diagonal(mask, False)
            
            if np.any(mask):
                # Calculate forces
                # For each violating pair (i,j), push i and j apart
                # force = (min_dist - dist) / 2 * direction
                # We can sum up forces for each point
                
                # Create a force matrix
                forces = np.zeros_like(pts)
                
                # Get indices of violations
                i_idx, j_idx = np.where(mask)
                
                for i, j in zip(i_idx, j_idx):
                    d = pts[i] - pts[j]
                    d_norm = np.linalg.norm(d)
                    if d_norm < 1e-10:
                        # Random kick
                        d = np.random.randn(2)
                        d_norm = np.linalg.norm(d)
                    # Push magnitude
                    push = (min_dist - d_norm) / (2.0 * d_norm)
                    forces[i] += push * d
                    forces[j] -= push * d
                
                # Apply forces with a damping factor to avoid oscillation
                # Simple damping: 0.5
                pts += 0.5 * forces
                moved = True
            
            # Boundary constraint
            dist_to_center = np.linalg.norm(pts, axis=1)
            # Points outside max_r
            out_mask = dist_to_center > max_r
            if np.any(out_mask):
                for i in np.where(out_mask)[0]:
                    if dist_to_center[i] < 1e-10:
                        continue
                    # Pull back to boundary
                    pts[i] = pts[i] * (max_r / dist_to_center[i])
                    moved = True
            
            # If no movement, we converged
            if not moved:
                break
        
        # Calculate achieved radius
        achieved_r = calc_radius(pts)
        # Check if it's valid for the target
        success = achieved_r >= r_target - 1e-9
        
        return pts, success, achieved_r

    best_r = 0.0
    best_pts = None
    
    # List of initializers
    inits = []
    # Start with a slightly smaller radius for initialization to ensure they fit
    r_init = 0.12
    
    inits.append(get_hex_init(r_init))
    inits.append(get_ring_init(r_init))
    inits.append(get_random_init(r_init))
    
    # Add a few more random inits
    for _ in range(2):
        inits.append(get_random_init(r_init))
        
    # Shuffle inits to diversify search
    np.random.shuffle(inits)
    
    # We will try to increase the radius incrementally
    # Start from a safe lower bound
    current_r = 0.10
    
    # We iterate through inits, and for each, we try to push the radius up
    # We keep a global best
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Use the current best points as a potential init for subsequent runs if they are good?
        # Or just use the generated init.
        # Let's use the generated init.
        
        pts = init_pts.copy()
        r_target = current_r
        
        # Incremental increase
        # We can also use a binary search on r for each init, but incremental is more stable for relaxation
        # Let's do a loop: try to satisfy r_target. If success, r_target += step.
        
        # First, ensure we can satisfy the initial r_target
        pts, success, achieved_r = relax(pts, r_target, max_iters=5000, time_limit=time_limit-10)
        
        if success:
            if achieved_r > best_r:
                best_r = achieved_r
                best_pts = pts.copy()
            
            # Now try to increase
            step = 0.001
            while time.time() - start_time < time_limit - 5:
                r_target += step
                # If r_target is too high, we might fail. 
                # We can also adapt step size.
                
                pts, success, achieved_r = relax(pts, r_target, max_iters=5000, time_limit=time_limit-5)
                
                if success:
                    if achieved_r > best_r:
                        best_r = achieved_r
                        best_pts = pts.copy()
                    # Keep stepping
                    # Maybe increase step slightly? No, keep it fine for precision.
                else:
                    # Failed to achieve r_target
                    # We can try to backtrack or just stop this init's progression
                    # Since we have the best_pts from the last successful step, we are safe.
                    # We can break here to save time for other inits.
                    break
        else:
            # Couldn't even achieve the initial r_target
            # This init is bad, skip
            continue

    # Final check: if we didn't find anything better than the seed, return the seed-like hex
    if best_pts is None:
        # Fallback
        pts = get_hex_init(0.12)
        # Relax it a bit to ensure validity
        pts, _, _ = relax(pts, 0.12, max_iters=1000, time_limit=time_limit-1)
        best_pts = pts
        best_r = calc_radius(best_pts)
        
    # Ensure the returned points are strictly valid
    # The relax function should guarantee this, but let's double check
    # If best_r is very small, it might be invalid.
    # The score is calculated by the evaluator.
    
    return best_pts
