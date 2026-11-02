"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        # x is [x1,y1,r1, x2,y2,r2, ...]
        valid = True
        # Check bounds
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                valid = False
                break
        if not valid:
            return
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()

    def physics_simulate(x0, w, h, t_start, t_end, steps, temp_start, temp_end, seed):
        """
        Simulates annealing of circles using a penalty-based energy function.
        x0: initial state [x1, y1, r1, ...]
        w, h: rectangle dimensions
        """
        random.seed(seed)
        x = x0.copy()
        v = np.zeros_like(x)
        
        # Precompute indices for efficiency
        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                pairs.append((i, j))
        
        # Energy function: Penalty for overlap + Penalty for boundary violation - Reward for radius
        def get_energy_and_grad(x):
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            
            energy = 0.0
            grad = np.zeros_like(x)
            
            # 1. Boundary penalties
            for i in range(N):
                xi, yi, ri = xs[i], ys[i], rs[i]
                # X bounds
                if xi - ri < 0:
                    pen = (xi - ri)**2
                    energy += pen
                    grad[3*i] += -2 * pen
                if xi + ri > w:
                    pen = (xi + ri - w)**2
                    energy += pen
                    grad[3*i] += 2 * pen
                # Y bounds
                if yi - ri < 0:
                    pen = (yi - ri)**2
                    energy += pen
                    grad[3*i+1] += -2 * pen
                if yi + ri > h:
                    pen = (yi + ri - h)**2
                    energy += pen
                    grad[3*i+1] += 2 * pen
                # Radius positivity
                if ri < 0:
                    pen = ri**2
                    energy += pen
                    grad[3*i+2] += 2 * pen
                else:
                    # Reward for radius (we want to maximize sum of radii, so minimize -sum(r))
                    energy -= ri
                    grad[3*i+2] -= 1.0

            # 2. Circle-Circle overlap penalties
            for i, j in pairs:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    pen = overlap**2
                    energy += pen
                    
                    # Gradient w.r.t positions
                    # d(pen)/d(dist) = -2*overlap
                    # d(dist)/dx_i = -dx/dist (since dist = sqrt((xi-xj)^2...)) -> wait, dist = |xi-xj|
                    # Let's use vector form: grad_dist_i = (xi-xj)/dist, grad_dist_j = (xj-xi)/dist
                    # Actually, if dist < min_dist, we want to push them apart.
                    # Force on i is in direction (xi-xj)/dist
                    # Force on j is in direction (xj-xi)/dist
                    norm_dist = 1.0 / dist
                    fx = -2 * overlap * dx * norm_dist
                    fy = -2 * overlap * dy * norm_dist
                    
                    # grad_x_i = fx, grad_x_j = -fx
                    # grad_y_i = fy, grad_y_j = -fy
                    grad[3*i] += fx
                    grad[3*i+1] += fy
                    grad[3*j] -= fx
                    grad[3*j+1] -= fy
                    
                    # Gradient w.r.t radii
                    # d(pen)/d(ri) = d(pen)/d(min_dist) * d(min_dist)/d(ri) = -2*overlap * 1
                    grad[3*i+2] += -2 * overlap
                    grad[3*j+2] += -2 * overlap
            
            return energy, grad

        # Velocity Verlet with annealing
        dt = 0.01
        # Initial temperature based on typical energy scale
        # We'll just use a simple cooling schedule
        
        current_energy, _ = get_energy_and_grad(x)
        best_local_energy = current_energy
        best_local_x = x.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Cooling
            temp = temp_start * (temp_end / temp_start) ** (step / steps)
            
            _, grad = get_energy_and_grad(x)
            
            # Update velocity
            v += dt * grad
            # Damping to help settle
            v *= 0.9
            
            # Update position
            x += dt * v
            
            # Calculate new energy
            new_energy, _ = get_energy_and_grad(x)
            
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0:
                # Accept
                current_energy = new_energy
                if new_energy < best_local_energy:
                    best_local_energy = new_energy
                    best_local_x = x.copy()
            else:
                # Accept with probability exp(-delta_e / T)
                if random.random() < np.exp(-delta_e / max(temp, 1e-9)):
                    current_energy = new_energy
                    if new_energy < best_local_energy:
                        best_local_energy = new_energy
                        best_local_x = x.copy()
                else:
                    # Reject: revert x and v
                    x -= dt * v
                    v /= 0.9 # Reverse damping approximately
                    # Note: proper reverse Verlet would be more complex, but this simple rollback is okay for heuristic search
            
        return best_local_x

    # Loop over aspect ratios (widths)
    # w + h = 2, so h = 2 - w
    # Try a range of widths
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize x0 with a grid
        # Estimate grid size
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        x0 = np.zeros(3*N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3*i] = (col + 0.5) * w / a
            x0[3*i+1] = (row + 0.5) * h / b
            x0[3*i+2] = min(w/a, h/b) * 0.45 # Initial radius guess
        
        # Run physics simulation
        # Parameters: steps, temp_start, temp_end
        # We want to explore the space, so high initial temp, low final temp
        # Number of steps depends on time budget
        # Let's allocate ~10s per width for physics + polish
        
        steps = 2000
        temp_start = 0.1
        temp_end = 0.001
        
        x_physics = physics_simulate(x0, w, h, 0, 1, steps, temp_start, temp_end, seed=42)
        
        # Polish with L-BFGS-B
        def objective(params):
            # params: [x1, y1, r1, ..., x21, y21, r21]
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            if np.any(rs <= 0):
                return 1e10 + np.sum(np.minimum(rs, 0)**2)
                
            # Circle-Circle Overlap Penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            # Boundary Penalty for polish to ensure strict validity
            for i in range(N):
                xi, yi, ri = xs[i], ys[i], rs[i]
                if xi - ri < 0: penalty += (xi - ri)**2
                if xi + ri > w: penalty += (xi + ri - w)**2
                if yi - ri < 0: penalty += (yi - ri)**2
                if yi + ri > h: penalty += (yi + ri - h)**2

            return penalty - np.sum(rs)
        
        # Bounds for L-BFGS-B
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
            
        try:
            res = minimize(objective, x_physics, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline:
                break
            check_and_update(res.x, w, h)
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
