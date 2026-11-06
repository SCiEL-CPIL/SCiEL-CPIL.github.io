import numpy as np
import time
import scipy.optimize as opt

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # 1. Generate a good initial hexagonal lattice configuration
    d = 0.076
    points_list = []
    dy = d * np.sqrt(3) / 2
    n_rows = int(np.ceil(1.0 / dy)) + 2
    for j in range(-n_rows, n_rows + 1):
        y = j * dy
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (j % 2) * d / 2
        n_cols = int(np.ceil(1.0 / d)) + 2
        for i in range(-n_cols, n_cols + 1):
            x = i * d + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    if len(points) > N:
        np.random.shuffle(points)
        points = points[:N]
    elif len(points) < N:
        remaining = N - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    def get_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_c = np.min(dist) / 2.0
        
        dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
        dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        
        return min(min_c, min_b)
    
    current_r = get_radius(points)
    
    # Vectorized Objective and Gradient
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        # 1. Boundary Violations (Vectorized)
        d_left = p[:, 0] + 0.5
        d_right = 0.5 - p[:, 0]
        d_bottom = p[:, 1] + 0.5
        d_top = 0.5 - p[:, 1]
        
        d_bounds = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
        min_b = np.min(d_bounds, axis=1)
        
        viol_b = np.maximum(0.0, r_target - min_b)
        E += np.sum(viol_b**2)
        
        min_idx = np.argmin(d_bounds, axis=1)
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        
        gx[min_idx == 0] = 1.0
        gx[min_idx == 1] = -1.0
        gy[min_idx == 2] = 1.0
        gy[min_idx == 3] = -1.0
        
        grad[:, 0] += -2.0 * viol_b * gx
        grad[:, 1] += -2.0 * viol_b * gy
        
        # 2. Pairwise Violations (Vectorized)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_points = points.copy()
    best_r = current_r
    
    r = current_r
    step_r = 0.0005
    
    while time.time() < deadline:
        new_r = r + step_r
        
        p_flat = best_points.flatten()
        
        try:
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(new_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
                r = new_actual_r
            else:
                break
                
        except Exception:
            break
            
    # Final Polish
    if time.time() < deadline - 5.0:
        p_flat = best_points.flatten()
        try:
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(best_r,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            new_points = res.x.reshape(N, 2)
            new_actual_r = get_radius(new_points)
            if new_actual_r > best_r:
                best_r = new_actual_r
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points

# Test
if __name__ == "__main__":
    pts = construct()
    r = get_radius(pts) if 'get_radius' in dir() else None
    print(f"Shape: {pts.shape}")
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_c = np.min(dist) / 2.0
    dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
    dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    min_b = np.min(np.minimum(dx, dy))
    print(f"Radius: {min(min_c, min_b)}")
