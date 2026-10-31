import numpy as np
import time
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
DEADLINE_FRACTION = 0.90

def construct():
    t0 = time.time()
    deadline = 120.0 * DEADLINE_FRACTION
    
    best_val = -1.0
    best_circles = None
    
    # We will try a few different random seeds / initializations
    # and different aspect ratios implicitly by letting W vary.
    
    # Number of restarts to try
    max_restarts = 50
    
    # Precompute pair indices for overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(z):
        # z: [x1..xN, y1..yN, r1..rN, W]
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        W = z[3*N]
        H = 2.0 - W
        
        # If W or H is invalid, return high penalty
        if W <= 0 or H <= 0:
            return 1e6
        
        # Boundary penalties
        # x_i - r_i >= 0  => r_i - x_i <= 0
        # x_i + r_i <= W  => x_i + r_i - W <= 0
        # y_i - r_i >= 0  => r_i - y_i <= 0
        # y_i + r_i <= H  => y_i + r_i - H <= 0
        
        pen_x_min = np.maximum(0, r - x)
        pen_x_max = np.maximum(0, x + r - W)
        pen_y_min = np.maximum(0, r - y)
        pen_y_max = np.maximum(0, y + r - H)
        
        boundary_pen = np.sum(pen_x_min**2) + np.sum(pen_x_max**2) + \
                       np.sum(pen_y_min**2) + np.sum(pen_y_max**2)
        
        # Overlap penalties
        # d_ij >= r_i + r_j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx**2 + dy**2)
        overlap = np.maximum(0, r[i_idx] + r[j_idx] - d)
        overlap_pen = np.sum(overlap**2)
        
        total_pen = boundary_pen + overlap_pen
        
        # Objective: maximize sum(r) -> minimize -sum(r) + penalty
        # Penalty weight
        lam = 100.0
        return -np.sum(r) + lam * total_pen
    
    # Bounds for L-BFGS-B
    # x, y can be roughly in [0, 2]
    # r in [0, 1]
    # W in [0.5, 1.5]
    bounds_x = [(0.0, 2.0)] * N
    bounds_y = [(0.0, 2.0)] * N
    bounds_r = [(0.01, 1.0)] * N
    bounds_W = [(0.5, 1.5)]
    bounds = bounds_x + bounds_y + bounds_r + bounds_W
    
    rng = np.random.default_rng(42)
    
    for restart in range(max_restarts):
        if time.time() - t0 > deadline:
            break
        
        # Initialize
        W0 = rng.uniform(0.8, 1.2)
        H0 = 2.0 - W0
        
        # Jittered grid initialization for better spread
        # 5x5 grid points
        grid_x = np.linspace(0.1*W0, 0.9*W0, 5)
        grid_y = np.linspace(0.1*H0, 0.9*H0, 5)
        
        # Create 21 points from grid with jitter
        # 5x5 = 25 points, take 21
        gx, gy = np.meshgrid(grid_x, grid_y)
        gx = gx.flatten()
        gy = gy.flatten()
        
        # Shuffle and take first 21
        idx = rng.permutation(25)[:21]
        x0 = gx[idx] + rng.uniform(-0.05, 0.05, 21)
        y0 = gy[idx] + rng.uniform(-0.05, 0.05, 21)
        
        # Clip to bounds
        x0 = np.clip(x0, 0.05, W0 - 0.05)
        y0 = np.clip(y0, 0.05, H0 - 0.05)
        
        r0 = np.full(21, 0.05)
        
        z0 = np.concatenate([x0, y0, r0, [W0]])
        
        # Optimize
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10})
        except Exception:
            continue
            
        if not res.success:
            # Still check if it's valid, sometimes it converges enough
            pass
            
        z_opt = res.x
        x_opt = z_opt[:N]
        y_opt = z_opt[N:2*N]
        r_opt = z_opt[2*N:3*N]
        W_opt = z_opt[3*N]
        H_opt = 2.0 - W_opt
        
        # Check validity strictly
        # 1. Bounds
        if np.any(x_opt - r_opt < -1e-6) or np.any(x_opt + r_opt > W_opt + 1e-6) or \
           np.any(y_opt - r_opt < -1e-6) or np.any(y_opt + r_opt > H_opt + 1e-6):
            continue
            
        # 2. Overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x_opt[i]-x_opt[j], y_opt[i]-y_opt[j])
                if d < r_opt[i] + r_opt[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Calculate value
        val = np.sum(r_opt)
        
        # Scale to perimeter 4 if necessary (should be close)
        # Bounding box of circles
        minx = np.min(x_opt - r_opt)
        miny = np.min(y_opt - r_opt)
        maxx = np.max(x_opt + r_opt)
        maxy = np.max(y_opt + r_opt)
        
        bw = maxx - minx
        bh = maxy - miny
        per = 2.0 * (bw + bh)
        
        if per <= 0:
            continue
            
        s = 4.0 / per
        
        # Scale coordinates
        x_scaled = (x_opt - minx) * s
        y_scaled = (y_opt - miny) * s
        r_scaled = r_opt * s
        
        # Re-check validity after scaling (should be same, but floating point)
        # The scaling preserves relative distances and ratios, so validity is preserved.
        # However, the rectangle defined by min/max might be smaller than W_opt x H_opt
        # if circles don't touch all 4 sides.
        # The problem asks for "bounding rectangle has perimeter at most 4".
        # My optimization assumed the rectangle is W_opt x H_opt.
        # If the circles fit in a smaller rectangle, we can shrink the rectangle to the bounding box
        # and scale up to perimeter 4.
        
        # The value to report is sum of radii.
        # If we scale up by s, the radii become r_scaled.
        # The new value is sum(r_scaled) = s * sum(r_opt).
        
        new_val = np.sum(r_scaled)
        
        if new_val > best_val:
            best_val = new_val
            best_circles = np.column_stack([x_scaled, y_scaled, r_scaled])
            
    if best_circles is None:
        # Fallback: simple grid
        # 5x5 grid
        W = 1.0
        H = 1.0
        side = 5
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    pts.append((x, y, r))
        best_circles = np.array(pts)
        # Scale
        minx = min(c[0] - c[2] for c in best_circles)
        miny = min(c[1] - c[2] for c in best_circles)
        maxx = max(c[0] + c[2] for c in best_circles)
        maxy = max(c[1] + c[2] for c in best_circles)
        bw = maxx - minx
        bh = maxy - miny
        per = 2.0 * (bw + bh)
        s = 4.0 / per
        best_circles = np.column_stack([
            (best_circles[:, 0] - minx) * s,
            (best_circles[:, 1] - miny) * s,
            best_circles[:, 2] * s
        ])

    return best_circles

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
