import numpy as np
import time

N = 200

def construct():
    t_start = time.time()
    deadline = t_start + 115.0 # Safety margin
    
    # 1. Initialization: Hexagonal Grid
    # Approximate hex packing
    # Area per point ~ 1/N. Side of triangle ~ sqrt(1/N).
    # Spacing s. Area per hex cell ~ s^2 * sqrt(3)/2.
    # s ~ sqrt(2 / (N * sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    pts = []
    y = -0.5 + s/2
    row = 0
    while y < 0.5 and len(pts) < N:
        x_offset = (row % 2) * s / 2
        x = -0.5 + s/2 + x_offset
        while x < 0.5 and len(pts) < N:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
        
    if len(pts) < N:
        # Fill remaining with random
        remaining = N - len(pts)
        rand_pts = np.random.rand(remaining, 2) - 0.5
        pts.extend(rand_pts.tolist())
        
    pts = np.array(pts[:N])
    # Ensure strictly inside
    pts = np.clip(pts, -0.49, 0.49)

    def calc_radius(p):
        # Vectorized distance calc
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall) / 2

    current_r = calc_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # 2. Optimization: Gradient Descent on Energy
    # Energy E = sum_{i<j} 1/d_ij^2 + sum_i 1/w_i^2
    # We want to minimize E.
    
    eta = 0.001 # Initial step size
    eta_decay = 0.9995
    min_eta = 1e-6
    
    steps = 0
    while time.time() < deadline - 2.0:
        # Calculate Forces
        # dist_matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dists_sq[dists_sq < 1e-12] = 1e-12
        
        # Force from pairs: F_i = sum_j (p_i - p_j) / d_ij^4 ? 
        # Gradient of 1/d^2 is -2 * (p_i - p_j) / d^4.
        # We want to move in direction of -grad E, which is +2 * (p_i - p_j) / d^4.
        # So force is repulsive.
        
        # W[i,j] = 1/d_ij^4
        W = 1.0 / (dists_sq * dists_sq)
        np.fill_diagonal(W, 0)
        
        # Force from pairs: F_pairs = W @ diff ? No.
        # F_i = sum_j W[i,j] * (p_i - p_j)
        #      = p_i * sum_j W[i,j] - sum_j W[i,j] * p_j
        row_sums = W.sum(axis=1)
        F_pairs = row_sums[:, np.newaxis] * pts - W.dot(pts)
        
        # Force from walls
        # E_wall = sum_i 1/w_i^2.
        # w_i = min(0.5-x, x+0.5, 0.5-y, y+0.5)
        # Gradient w.r.t x:
        # if x > 0, min is 0.5-x (right wall) or 0.5-y? No, depends on y.
        # Let's compute wall distances and gradients explicitly.
        
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_right = 0.5 - x
        d_left = x + 0.5
        d_top = 0.5 - y
        d_bottom = y + 0.5
        
        # Wall distance is min of these 4.
        # We need gradient of 1/min_d^2.
        # If d_right is min, d(1/d_right^2)/dx = -2 * (-1) / d_right^3 = 2/d_right^3.
        # Wait, d_right = 0.5 - x. d(d_right)/dx = -1.
        # d(1/d_right^2)/dx = -2 * d_right^-3 * (-1) = 2 * d_right^-3.
        # So force is +x direction? No, we want to move AWAY from wall.
        # If close to right wall (x large), d_right small. Force should be -x (left).
        # Let's check sign.
        # E = 1/d^2. dE/dx = 2/d^3 * dd/dx.
        # If d = 0.5 - x, dd/dx = -1. dE/dx = -2/d^3.
        # Gradient descent: p_new = p - eta * grad.
        # p_new_x = x - eta * (-2/d^3) = x + 2*eta/d^3.
        # This moves x to the RIGHT? That's wrong.
        # Ah, E is high when close. We want to MINIMIZE E.
        # If x is close to 0.5, d is small, E is large.
        # We want to decrease x.
        # Let's re-evaluate:
        # E = 1/d^2.
        # If x increases, d decreases, E increases.
        # So dE/dx is positive.
        # Gradient descent: x_new = x - eta * (positive) -> x decreases. Correct.
        
        # So we need dE/dx.
        # dE/dx = sum over active walls.
        # Only the closest wall contributes to the min.
        
        F_walls = np.zeros_like(pts)
        
        # For each point, find min wall dist and apply force
        # Vectorized approach for wall forces:
        # This is tricky to vectorize perfectly without loops due to 'min' selection.
        # But N=200, loop is fine.
        
        for i in range(N):
            dx_r = 0.5 - pts[i, 0]
            dx_l = pts[i, 0] + 0.5
            dy_t = 0.5 - pts[i, 1]
            dy_b = pts[i, 1] + 0.5
            
            min_d = min(dx_r, dx_l, dy_t, dy_b)
            
            # Gradient of 1/min_d^2
            # d(1/d^2)/dd = -2/d^3
            # Chain rule:
            # If dx_r is min: d/dx_r (1/dx_r^2) * d(dx_r)/dx = (-2/dx_r^3) * (-1) = 2/dx_r^3
            # Wait, earlier I said dE/dx is positive.
            # Let's check: x=0.49, d=0.01. E=10000.
            # x=0.48, d=0.02. E=2500.
            # E decreased as x decreased.
            # So dE/dx should be positive (slope is positive).
            # My derivation: 2/d^3 is positive. Correct.
            
            grad_x = 0
            grad_y = 0
            
            if dx_r == min_d:
                grad_x += 2.0 / (dx_r**3)
            if dx_l == min_d:
                # d_left = x + 0.5. d/dx = 1.
                # d(1/d^2)/dx = (-2/d^3) * 1 = -2/d^3.
                grad_x -= 2.0 / (dx_l**3)
            if dy_t == min_d:
                # d_top = 0.5 - y. d/dy = -1.
                # d(1/d^2)/dy = (-2/d^3) * (-1) = 2/d^3.
                grad_y += 2.0 / (dy_t**3)
            if dy_b == min_d:
                # d_bottom = y + 0.5. d/dy = 1.
                # d(1/d^2)/dy = -2/d^3.
                grad_y -= 2.0 / (dy_b**3)
                
            F_walls[i, 0] = grad_x
            F_walls[i, 1] = grad_y
            
        Total_Force = F_pairs + F_walls
        
        # Update
        pts += eta * Total_Force
        pts = np.clip(pts, -0.49, 0.49)
        
        eta *= eta_decay
        
        # Periodic check of actual radius
        if steps % 10 == 0:
            current_r = calc_radius(pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
                
        steps += 1
        
    return best_pts
