import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin for 120s limit
    
    # 1. Initialization
    # Grid 5x5x4
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 4)
    
    centers = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    # Add jitter
    centers += np.random.uniform(-0.01, 0.01, size=(N, 3))
    
    # Simple force-directed relaxation to break symmetry
    # (Optional, but helps)
    for _ in range(50):
        if time.time() - start_time > timeout: break
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 0.15: # Repel if close
                    forces[i] += d / (dist**3 + 1e-6)
            # Wall repulsion
            for k in range(3):
                if centers[i, k] > 0.4: forces[i, k] -= 1.0
                elif centers[i, k] < -0.4: forces[i, k] += 1.0
        centers += 0.001 * forces
        # Project back to box
        centers = np.clip(centers, -0.5, 0.5)

    best_radius = 0.0
    best_centers = centers.copy()
    
    # 2. Binary Search on Radius
    # Theoretical max ~ 0.133. Grid ~ 0.107. Best known 0.111.
    low_r = 0.10
    high_r = 0.115
    
    def compute_radius(c):
        # Min distance between pairs
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                dists.append(np.linalg.norm(c[i] - c[j]))
        min_pair_dist = min(dists) if dists else 0.0
        # Min distance to walls
        min_wall_dist = np.inf
        for i in range(N):
            for k in range(3):
                min_wall_dist = min(min_wall_dist, 0.5 - abs(c[i, k]))
        return min(min_pair_dist / 2.0, min_wall_dist)

    def penalty_and_grad(c_flat, r):
        c = c_flat.reshape(N, 3)
        penalty = 0.0
        grad = np.zeros_like(c_flat)
        
        # Sphere-sphere
        for i in range(N):
            for j in range(i+1, N):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 2*r:
                    violation = 2*r - dist
                    penalty += violation**2
                    # d/dc_i (2r - dist)^2 = 2(2r-dist) * (-d/dist)
                    # d/dc_j (2r - dist)^2 = 2(2r-dist) * (d/dist)
                    factor = 2 * violation / dist
                    grad[i*3:(i+1)*3] -= factor * d
                    grad[j*3:(j+1)*3] += factor * d
                    # Wait, derivative of (2r - dist) wrt c_i is -d/dist.
                    # So grad penalty wrt c_i is 2*viol * (-d/dist).
                    # My code: grad[i] -= factor * d. factor = 2*viol/dist. Correct.
                    
        # Sphere-wall
        for i in range(N):
            for k in range(3):
                coord = c[i, k]
                limit = 0.5 - r
                if coord > limit:
                    viol = coord - limit
                    penalty += viol**2
                    grad[i*3+k] += 2 * viol
                elif coord < -limit:
                    viol = -limit - coord
                    penalty += viol**2
                    grad[i*3+k] -= 2 * viol
        return penalty, grad

    def check_feasibility(r, current_centers):
        # Optimize to see if we can fit radius r
        c0 = current_centers.flatten()
        
        # Use L-BFGS-B
        # maxiter needs to be enough but not too slow
        res = minimize(
            lambda x: penalty_and_grad(x, r)[0],
            c0,
            jac=lambda x: penalty_and_grad(x, r)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < 1e-6:
            c_opt = res.x.reshape(N, 3)
            # Verify actual radius
            r_actual = compute_radius(c_opt)
            if r_actual >= r - 1e-5:
                return True, c_opt, r_actual
        return False, None, 0.0

    # Binary Search
    for _ in range(25): # ~25 iterations should be enough for 1e-3 precision
        if time.time() - start_time > timeout:
            break
            
        mid_r = (low_r + high_r) / 2.0
        feasible, c_opt, r_actual = check_feasibility(mid_r, best_centers)
        
        if feasible:
            low_r = mid_r
            if r_actual > best_radius:
                best_radius = r_actual
                best_centers = c_opt
        else:
            high_r = mid_r

    # Final Polish: Try to improve best_radius slightly
    # Run one more optimization at best_radius
    if time.time() - start_time < timeout - 5:
        feasible, c_opt, r_actual = check_feasibility(best_radius, best_centers)
        if feasible and r_actual > best_radius:
            best_radius = r_actual
            best_centers = c_opt

    return best_centers
