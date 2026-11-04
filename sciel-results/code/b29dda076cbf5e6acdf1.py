import numpy as np

N = 50

def construct():
    """
    Uses Simulated Annealing to maximize the minimum radius of 50 circles
    in a unit square.
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

    # Initial configuration: Grid
    # 7 columns, 8 rows = 56 points, take 50
    # Or 8x7
    # Let's try a few initial grids and pick the best start
    best_start_r = -1
    best_start_pts = None
    
    for m in [7, 8]:
        for n in [7, 8]:
            if m * n < N: continue
            ticks_x = (np.arange(m) + 0.5) / m - 0.5
            ticks_y = (np.arange(n) + 0.5) / n - 0.5
            grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
            pts = grid[:N].astype(float)
            # Jitter
            pts += rng.uniform(-0.005, 0.005, size=pts.shape)
            pts = np.clip(pts, -0.49, 0.49)
            r = compute_radius(pts)
            if r > best_start_r:
                best_start_r = r
                best_start_pts = pts.copy()
                
    pts = best_start_pts
    r_curr = best_start_r
    r_best = r_curr
    best_pts = pts.copy()
    
    # SA Parameters
    T_start = 0.01
    T_end = 0.0001
    alpha = (T_end / T_start) ** (1.0 / 20000) # 20000 steps
    steps = 20000
    sigma_start = 0.05
    sigma_end = 0.001
    sigma_alpha = (sigma_end / sigma_start) ** (1.0 / steps)
    
    import time
    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    for i in range(steps):
        if time.time() - start_time > timeout:
            break
            
        T = T_start * (alpha ** i)
        sigma = sigma_start * (sigma_alpha ** i)
        
        # Generate moves
        moves = rng.normal(0, sigma, size=pts.shape)
        new_pts = pts + moves
        # Keep inside square (hard constraint for validity, but we can allow soft violation in SA? 
        # No, the score is invalid if outside. So we must project back or penalize heavily.
        # Projecting back changes the move. Let's project to [-0.5, 0.5].
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        r_new = compute_radius(new_pts)
        
        if r_new > r_curr:
            # Accept
            pts = new_pts
            r_curr = r_new
            if r_curr > r_best:
                r_best = r_curr
                best_pts = pts.copy()
        else:
            # Metropolis acceptance
            delta_r = r_curr - r_new
            if delta_r < 1e-12:
                pts = new_pts
                r_curr = r_new
            else:
                # Prob = exp(-delta_r / T)
                # If delta_r/T is large, prob is 0.
                if delta_r / T < 50: # Avoid underflow
                    prob = np.exp(-delta_r / T)
                    if rng.random() < prob:
                        pts = new_pts
                        r_curr = r_new
                        if r_curr > r_best:
                            r_best = r_curr
                            best_pts = pts.copy()
                            
    return best_pts
