import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate initial hexagonal layout
    # We want to pack 26 circles. 
    # Let's try a 5x5 grid with some adjustments or a hex grid.
    # A 5x5 square grid has 25 circles. We need 26.
    # Let's use a hexagonal grid.
    
    def get_initial_layout():
        # Generate a hex grid of points
        # r is the "radius" in the grid sense, distance between centers is 2r
        # Let's estimate r. Area of square is 1. 
        # Approx area per circle in hex pack is ~ 2.17 r^2.
        # 26 * 2.17 r^2 <= 1 => r^2 <= 1/56.4 => r <= 0.133
        # Let's start with r = 0.12
        r0 = 0.12
        centers = []
        # Generate enough points
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        # Sort by distance from center (0.5, 0.5) to get a compact cluster
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        # Calculate bounding box to scale and center
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        # Scale to fit in [0.1, 0.9] to allow room for optimization
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
        # Initial radii
        # Set all radii to the same value, slightly less than half the min distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
                if d < min_dist:
                    min_dist = d
        # Also check boundary
        min_boundary = np.inf
        for x, y in scaled_centers:
            min_boundary = min(min_boundary, x, 1-x, y, 1-y)
            
        r_init = min(min_dist / 2, min_boundary) * 0.95
        
        v = []
        for x, y in scaled_centers:
            v.extend([x, y, r_init])
        return np.array(v)

    best_score = -1
    best_v = None
    
    # Objective function
    def objective(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Sum of radii
        score = np.sum(r)
        
        # Penalties
        penalty = 0.0
        
        # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Violation = max(0, r - x), etc.
        b_viol = np.maximum(0, r - x) + np.maximum(0, r - (1-x)) + np.maximum(0, r - y) + np.maximum(0, r - (1-y))
        penalty += np.sum(b_viol**2)
        
        # Overlap constraints: d_ij >= r_i + r_j
        # Violation = max(0, r_i + r_j - d_ij)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12) # regularize
                sum_r = r[i] + r[j]
                if sum_r > d:
                    penalty += (sum_r - d)**2
                    
        return -score + 1000 * penalty

    def gradient(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Gradient of boundary penalties
        # P_b = sum (max(0, r-x)^2 + ...)
        # d/dx_i: 2 * max(0, r_i - x_i) * (-1) if r_i > x_i
        # d/d(1-x_i): 2 * max(0, r_i - (1-x_i)) * (1) if r_i > 1-x_i -> -2 * ... w.r.t x_i
        # d/dr_i: 2 * (max(0, r_i-x_i) + max(0, r_i-(1-x_i)) + ...)
        
        for i in range(N):
            # x boundary
            v_x = r[i] - x[i]
            if v_x > 0:
                grad[i*3] += -2 * 1000 * v_x
                grad[i*3+2] += 2 * 1000 * v_x
                
            v_1mx = r[i] - (1-x[i])
            if v_1mx > 0:
                grad[i*3] += 2 * 1000 * v_1mx
                grad[i*3+2] += 2 * 1000 * v_1mx
                
            v_y = r[i] - y[i]
            if v_y > 0:
                grad[i*3+1] += -2 * 1000 * v_y
                grad[i*3+2] += 2 * 1000 * v_y
                
            v_1my = r[i] - (1-y[i])
            if v_1my > 0:
                grad[i*3+1] += 2 * 1000 * v_1my
                grad[i*3+2] += 2 * 1000 * v_1my
                
        # Gradient of overlap penalties
        # P_o = sum (max(0, r_i+r_j - d_ij)^2)
        # d/dx_i: 2 * max(0, ...) * (-dx/dx_i) = 2 * max(0, ...) * (-(x_i-x_j)/d)
        # d/dr_i: 2 * max(0, ...) * 1
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                sum_r = r[i] + r[j]
                viol = sum_r - d
                if viol > 0:
                    factor = 2 * 1000 * viol / d
                    # d/dx_i
                    grad[i*3] += factor * (-dx)
                    # d/dx_j
                    grad[j*3] += factor * dx
                    # d/dy_i
                    grad[i*3+1] += factor * (-dy)
                    # d/dy_j
                    grad[j*3+1] += factor * dy
                    # d/dr_i
                    grad[i*3+2] += 2 * 1000 * viol
                    # d/dr_j
                    grad[j*3+2] += 2 * 1000 * viol
                    
        return grad

    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Main loop
    v_init = get_initial_layout()
    
    while time.time() < deadline:
        # Optimize
        res = minimize(objective, v_init, jac=gradient, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        
        v_res = res.x
        # Check validity
        x = v_res[0::3]
        y = v_res[1::3]
        r = v_res[2::3]
        
        valid = True
        for i in range(N):
            if r[i] > x[i] + 1e-6 or r[i] > 1-x[i] + 1e-6 or r[i] > y[i] + 1e-6 or r[i] > 1-y[i] + 1e-6:
                valid = False
                break
            if r[i] < -1e-6: # Radius should be positive
                valid = False
                break
            if x[i] < -1e-6 or x[i] > 1+1e-6 or y[i] < -1e-6 or y[i] > 1+1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_v = v_res.copy()
        
        # Perturb for next restart
        # If we found a good solution, perturb it. Otherwise, perturb initial or random.
        if best_v is not None and np.random.rand() < 0.7:
            v_init = best_v + np.random.normal(0, 0.05, size=3*N)
            # Clip to bounds to keep it sane
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 0, 0.5)
        else:
            # New random hex layout or similar
            v_init = get_initial_layout()
            v_init = v_init + np.random.normal(0, 0.1, size=3*N)
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 0, 0.5)

    if best_v is None:
        # Fallback to initial layout if no valid solution found (should not happen)
        return np.array(get_initial_layout()).reshape(N, 3)
    
    return best_v.reshape(N, 3)
