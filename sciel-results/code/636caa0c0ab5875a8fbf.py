import numpy as np
import scipy.optimize as opt
import time

N = 21

def smooth_max(x, beta=10.0):
    """Smooth approximation of max(0, x)"""
    return (x + np.sqrt(x**2 + 1.0/beta**2)) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_sum = -np.inf
    best_config = None
    
    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4 => W+H = 2
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 1.3, 0.75]
    
    # Pre-compute initial grid positions for a unit square, scaled later
    # Grid of 5x5 for 21 circles
    side = 5
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            if len(grid_x) < N:
                grid_x.append((i + 0.5) / side)
                grid_y.append((j + 0.5) / side)
    
    init_x = np.array(grid_x)
    init_y = np.array(grid_y)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0:
            continue
            
        # Initial radii estimate
        # Area of rect = W*H. Area of N circles = N * pi * r^2.
        # Max r approx sqrt(W*H / (N*pi))
        r_init = np.sqrt(W * H / (N * np.pi)) * 0.9
        
        # Multi-start for this aspect ratio
        num_restarts = 5
        
        for r_idx in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Perturb initial positions
            if r_idx == 0:
                x0 = init_x * W
                y0 = init_y * H
            else:
                # Random perturbation
                x0 = init_x * W + np.random.uniform(-0.1*W, 0.1*W, N)
                y0 = init_y * H + np.random.uniform(-0.1*H, 0.1*H, N)
            
            # Initial radii
            r0 = np.full(N, r_init)
            
            # Variables: x (N), y (N), r (N)
            z0 = np.concatenate([x0, y0, r0])
            
            def cost(z):
                x = z[:N]
                y = z[N:2*N]
                r = z[2*N:]
                
                # Objective: Maximize sum(r) -> Minimize -sum(r)
                obj = -np.sum(r)
                
                # Penalty: Overlaps
                # Vectorized distance calculation
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[:, None] + r[None, :]
                
                # Only consider i < j
                iu = np.triu_indices(N, k=1)
                overlaps = min_dist[iu] - dist[iu]
                # Smooth penalty
                pen_overlap = np.sum(smooth_max(overlaps, beta=20.0)**2)
                
                # Penalty: Boundaries
                # x - r >= 0, x + r <= W, etc.
                bound_violations = []
                bound_violations.append(smooth_max(r - x, beta=20.0))
                bound_violations.append(smooth_max(r - (W - x), beta=20.0))
                bound_violations.append(smooth_max(r - y, beta=20.0))
                bound_violations.append(smooth_max(r - (H - y), beta=20.0))
                
                pen_bound = np.sum(np.vstack(bound_violations)**2)
                
                # Total cost
                # Weight for penalties
                w_pen = 100.0
                return obj + w_pen * (pen_overlap + pen_bound)
            
            try:
                res = opt.minimize(cost, z0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
                if res.success or res.status == 1:
                    z_opt = res.x
                    r_opt = z_opt[2*N:]
                    # Check validity
                    x_opt = z_opt[:N]
                    y_opt = z_opt[N:2*N]
                    
                    # Verify constraints strictly
                    valid = True
                    # Boundaries
                    if np.any(x_opt - r_opt < -1e-5) or np.any(x_opt + r_opt > W + 1e-5):
                        valid = False
                    if np.any(y_opt - r_opt < -1e-5) or np.any(y_opt + r_opt > H + 1e-5):
                        valid = False
                    
                    # Overlaps
                    if valid:
                        dx = x_opt[:, None] - x_opt[None, :]
                        dy = y_opt[:, None] - y_opt[None, :]
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = r_opt[:, None] + r_opt[None, :]
                        iu = np.triu_indices(N, k=1)
                        if np.any(min_dist[iu] - dist[iu] > 1e-5):
                            valid = False
                    
                    if valid:
                        sum_r = np.sum(r_opt)
                        if sum_r > best_sum:
                            best_sum = sum_r
                            # Store config: (x, y, r)
                            best_config = np.column_stack((x_opt, y_opt, r_opt))
            except Exception:
                pass

    if best_config is None:
        # Fallback to simple grid in 1x1 square
        side = 5
        r = 0.5 / side - 1e-3
        cells = [((i + 0.5) / side, (j + 0.5) / side, r) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
