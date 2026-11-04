import numpy as np

N = 21

def construct():
    best_score = -1
    best_result = None
    
    # Try different aspect ratios
    # W + H = 2, so H = 2 - W
    # W is width.
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    
    for W in aspect_ratios:
        H = 2.0 - W
        
        # Initialize
        side = int(np.ceil(np.sqrt(N)))
        grid_x = []
        grid_y = []
        for i in range(side):
            for j in range(side):
                grid_x.append((i + 0.5) / side * W)
                grid_y.append((j + 0.5) / side * H)
                if len(grid_x) == N:
                    break
            if len(grid_x) == N:
                break
        xs = np.array(grid_x)
        ys = np.array(grid_y)
        rs = np.full(N, 0.05)
        
        # Optimization
        lr_pos = 0.05
        lr_rad = 0.01
        steps = 1000
        penalty_weight = 50.0
        reward_weight = 1.0
        
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        pairs = np.array(pairs)
        
        for step in range(steps):
            decay = 1.0 / (1.0 + step * 0.01)
            
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)
            grad_r = np.zeros(N)
            
            # Pairwise
            for i, j in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-8:
                    d = 1e-8
                    # random perturbation
                    dx = 1e-8
                    dy = 0
                
                sum_r = rs[i] + rs[j]
                if sum_r > d:
                    diff = sum_r - d
                    # Gradient of (diff)^2
                    # d/dx_i (r_i+r_j - d)^2 = 2*diff * (-(x_i-x_j)/d)
                    gx = 2 * diff * (-dx / d)
                    gy = 2 * diff * (-dy / d)
                    
                    grad_x[i] += gx
                    grad_y[i] += gy
                    grad_x[j] -= gx
                    grad_y[j] -= gy
                    
                    grad_r[i] += 2 * diff
                    grad_r[j] += 2 * diff
            
            # Boundaries
            for i in range(N):
                # Left
                if rs[i] > xs[i]:
                    diff = rs[i] - xs[i]
                    grad_x[i] -= 2 * diff
                    grad_r[i] += 2 * diff
                # Right
                if rs[i] > W - xs[i]:
                    diff = rs[i] - (W - xs[i])
                    grad_x[i] += 2 * diff
                    grad_r[i] += 2 * diff
                # Bottom
                if rs[i] > ys[i]:
                    diff = rs[i] - ys[i]
                    grad_y[i] -= 2 * diff
                    grad_r[i] += 2 * diff
                # Top
                if rs[i] > H - ys[i]:
                    diff = rs[i] - (H - ys[i])
                    grad_y[i] += 2 * diff
                    grad_r[i] += 2 * diff
            
            # Objective
            grad_r -= reward_weight * np.ones(N)
            
            # Update
            xs -= lr_pos * decay * grad_x
            ys -= lr_pos * decay * grad_y
            rs -= lr_rad * decay * grad_r
            rs = np.maximum(rs, 1e-5)
            
            # Clip
            xs = np.clip(xs, 0, W)
            ys = np.clip(ys, 0, H)
            
        # Final cleanup
        # Ensure strict non-overlap and boundary
        for _ in range(10): # Relaxation steps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-8:
                        continue
                    if d < rs[i] + rs[j]:
                        overlap = rs[i] + rs[j] - d
                        rs[i] -= overlap / 2
                        rs[j] -= overlap / 2
                # Boundaries
                rs[i] = min(rs[i], xs[i], W - xs[i], ys[i], H - ys[i])
                rs[i] = max(rs[i], 1e-6)
        
        score = np.sum(rs)
        if score > best_score:
            best_score = score
            best_result = np.column_stack((xs, ys, rs))
            
    return best_result
