import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s kill

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        if len(path) == 0:
            return np.array([P_start, P_end])
        
        full_path = np.vstack([P_start, path, P_end])
        
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        
        return np.array(dense_points)

    def score_path_vectorized(path):
        """Vectorized scoring: process all dense placements at once."""
        dense = densify_path(path)
        M = dense.shape[0]
        if M == 0:
            return 0.0
        
        # All grid points: (n_points, 2)
        pts = np.column_stack([X_flat, Y_flat])  # (N, 2)
        
        # For each dense placement, compute u, v for all points
        # x0, y0, a for each placement
        x0 = dense[:, 0]  # (M,)
        y0 = dense[:, 1]  # (M,)
        a_rad = np.deg2rad(dense[:, 2])  # (M,)
        c = np.cos(a_rad)  # (M,)
        s = np.sin(a_rad)  # (M,)
        
        # For each placement i and each point j:
        # dx = X_flat[j] - x0[i], dy = Y_flat[j] - y0[i]
        # u = dx*c[i] + dy*s[i]
        # v = -dx*s[i] + dy*c[i]
        
        # Process in batches of placements to manage memory
        batch_size = min(M, 50)
        all_inside = np.ones(n_points, dtype=bool)
        
        for batch_start in range(0, M, batch_size):
            if not np.any(all_inside):
                break
            batch_end = min(batch_start + batch_size, M)
            
            xb = x0[batch_start:batch_end]  # (B,)
            yb = y0[batch_start:batch_end]  # (B,)
            cb = c[batch_start:batch_end]   # (B,)
            sb = s[batch_start:batch_end]   # (B,)
            
            # dx: (N, B), dy: (N, B)
            dx = X_flat[:, None] - xb[None, :]  # (N, B)
            dy = Y_flat[:, None] - yb[None, :]  # (N, B)
            
            # u = dx*c + dy*s : (N, B)
            u = dx * cb[None, :] + dy * sb[None, :]
            v = -dx * sb[None, :] + dy * cb[None, :]
            
            # in_r1: 0 <= u <= 1, -3 <= v <= 1
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            # in_r2: -3 <= u <= 1, 0 <= v <= 1
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            inside_batch = in_r1 | in_r2  # (N, B)
            
            # A point must be inside ALL placements in this batch
            inside_all_in_batch = inside_batch.all(axis=1)  # (N,)
            
            all_inside &= inside_all_in_batch
        
        count = np.count_nonzero(all_inside)
        return count * 0.0001

    def make_initial_path(n_wp, seed_val=42):
        """Create an initial path based on the piano movers curve shape."""
        rng = np.random.RandomState(seed_val)
        
        # The optimal path for moving a sofa around a corner typically
        # follows a curve. Use a parametric form.
        t = np.linspace(0.1, 0.9, n_wp)
        
        # x goes from 0 to -2, with a bulge
        xs_init = -2.0 * t + 0.3 * np.sin(np.pi * t)
        # y has a bump in the middle
        ys_init = 0.4 * np.sin(np.pi * t)
        # angles from 0 to 90
        angles = 90.0 * t
        
        # Add small random perturbation
        xs_init += rng.normal(0, 0.05, n_wp)
        ys_init += rng.normal(0, 0.05, n_wp)
        angles += rng.normal(0, 1.0, n_wp)
        
        # Enforce monotonic angles
        angles = np.maximum.accumulate(angles)
        angles = np.clip(angles, 2, 88)
        
        return np.column_stack([xs_init, ys_init, angles])

    def hill_climb(path, max_time, seed_val=0):
        """Coordinate descent hill climbing with adaptive step size."""
        rng = np.random.RandomState(seed_val)
        current = path.copy()
        current_score = score_path_vectorized(current)
        
        sigma = 0.12
        no_improve = 0
        max_no_improve = 40
        n_wp = current.shape[0]
        
        while time.time() < max_time and no_improve < max_no_improve:
            improved = False
            
            # Try each coordinate a few times
            for trial in range(4):
                if time.time() > max_time:
                    break
                
                idx = rng.randint(0, n_wp * 3)
                row = idx // 3
                col = idx % 3
                
                best_delta = 0
                best_s = current_score
                
                for d_sign in [1, -1]:
                    for d_scale in [1.0, 0.5, 0.25]:
                        delta = d_sign * sigma * d_scale
                        new_path = current.copy()
                        new_path[row, col] += delta
                        
                        if col == 2:
                            # Enforce monotonic angles
                            if row > 0:
                                new_path[row, 2] = max(new_path[row, 2], new_path[row-1, 2])
                            if row < n_wp - 1:
                                new_path[row, 2] = min(new_path[row, 2], new_path[row+1, 2])
                        
                        s = score_path_vectorized(new_path)
                        if s > best_s:
                            best_s = s
                            best_delta = delta
                            best_path = new_path.copy()
                
                if best_delta != 0:
                    current = best_path
                    current_score = best_s
                    improved = True
                    break
            
            if improved:
                no_improve = 0
                sigma *= 0.92
            else:
                no_improve += 1
                sigma *= 0.85
                if sigma < 0.005:
                    break
        
        return current, current_score

    # Main optimization loop
    best_path = None
    best_score = 0.0
    
    # Try different numbers of waypoints and seeds
    configs = [
        (5, 42), (6, 43), (7, 44), (8, 45), (6, 100), (7, 101), (5, 200)
    ]
    
    for n_wp, seed in configs:
        if time.time() > deadline - 60:
            break
        
        path = make_initial_path(n_wp, seed)
        s = score_path_vectorized(path)
        
        if s > best_score:
            best_score = s
            best_path = path.copy()
    
    # Hill climb the best path
    if best_path is not None:
        if time.time() < deadline - 30:
            best_path, best_score = hill_climb(best_path, deadline - 30, seed_val=7)
        
        # Try hill climbing from a few different initializations
        for n_wp, seed in [(6, 50), (7, 60), (8, 70)]:
            if time.time() > deadline - 20:
                break
            
            path = make_initial_path(n_wp, seed)
            s = score_path_vectorized(path)
            if s > best_score * 0.95:  # Only hill climb if competitive
                path, s = hill_climb(path, deadline - 20, seed_val=seed + 1)
                if s > best_score:
                    best_score = s
                    best_path = path.copy()
    
    if best_path is None:
        best_path = np.array([[-0.5, 0.3, 22.5], [-1.0, 0.5, 45.0], [-1.5, 0.3, 67.5]])
    
    return best_path
