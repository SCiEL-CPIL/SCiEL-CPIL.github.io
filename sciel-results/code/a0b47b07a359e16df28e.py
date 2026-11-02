import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s
    
    # 1. Initialization: Jittered Grid
    # 100 points. 5x5x4 grid.
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    
    # Create grid points
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    points = np.array(points[:N])
    
    # Add small random jitter to break symmetry
    np.random.seed(42)
    jitter_scale = 0.01
    points = points + np.random.uniform(-jitter_scale, jitter_scale, points.shape)
    
    # Ensure points are strictly inside
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_radius = -1.0
    
    def compute_radius(pts):
        # Compute min pair distance / 2
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                dists.append(np.linalg.norm(pts[i] - pts[j]) / 2.0)
        min_pair_dist = min(dists) if dists else 0.0
        
        # Compute min wall distance
        wall_dists = []
        for i in range(N):
            for k in range(3):
                wall_dists.append(0.5 - np.abs(pts[i][k]))
        min_wall_dist = min(wall_dists) if wall_dists else 0.0
        
        return min(min_pair_dist, min_wall_dist)
    
    current_radius = compute_radius(points)
    if current_radius > best_radius:
        best_radius = current_radius
        best_points = points.copy()
        
    # 2. L-BFGS with Soft-Min Objective
    # Objective: minimize -softmin(distances)
    # softmin(d) = -1/beta * ln(sum(exp(-beta * d)))
    # f(x) = 1/beta * ln(sum(exp(-beta * d_k)))
    
    def objective_and_grad(x, beta):
        pts = x.reshape(N, 3)
        
        # Calculate distances and gradients
        # We need sum(exp(-beta * d) * grad_d) / sum(exp(-beta * d))
        
        sum_exp = 0.0
        grad_sum = np.zeros_like(pts)
        
        # Pair distances
        for i in range(N):
            for j in range(i+1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    continue
                d = dist / 2.0
                exp_term = np.exp(-beta * d)
                sum_exp += exp_term
                
                # grad of d wrt pts[i] is 0.5 * diff/dist
                # grad of d wrt pts[j] is 0.5 * -diff/dist
                g = 0.5 * diff / dist
                grad_sum[i] -= exp_term * g  # Note: objective is +1/beta ln(sum), so grad f = -1/sum * sum(exp * grad_d)
                # Wait. f = (1/beta) ln S. df/dx = (1/beta) (1/S) dS/dx.
                # S = sum exp(-beta d). dS/dx = sum (-beta exp(-beta d)) dd/dx.
                # df/dx = (1/beta) (1/S) sum (-beta exp(-beta d)) dd/dx = - (1/S) sum exp(-beta d) dd/dx.
                # So we accumulate sum(exp * dd/dx) and then multiply by -1/S.
                # Let's re-verify signs.
                # We want to maximize min distance. So minimize -min distance.
                # f(x) approximates -min(d).
                # If d increases, f should decrease.
                # dd/di = 0.5 * (ci-cj)/dist.
                # If we move ci in direction (ci-cj), dist increases, d increases.
                # df/dci should be negative.
                # Formula: df/dx = - (1/S) sum exp(-beta d) dd/dx.
                # dd/dci is positive in direction of separation.
                # So df/dci is negative. Correct.
                
                grad_sum[i] += exp_term * g
                grad_sum[j] -= exp_term * g
        
        # Wall distances
        for i in range(N):
            for k in range(3):
                d = 0.5 - np.abs(pts[i][k])
                exp_term = np.exp(-beta * d)
                sum_exp += exp_term
                
                # grad of d wrt pts[i][k] is -sign(pts[i][k])
                g_k = -np.sign(pts[i][k])
                # df/dx = - (1/S) sum exp * dd/dx
                # So add exp * g_k to grad_sum, then negate at end.
                grad_sum[i][k] += exp_term * g_k
        
        if sum_exp == 0:
            return 0.0, grad_sum
            
        f_val = (1.0 / beta) * np.log(sum_exp)
        grad_val = -grad_sum / sum_exp
        
        return f_val, grad_val

    betas = [10, 50, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        # Flatten points
        x0 = points.flatten()
        
        # Bounds for L-BFGS-B
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                args=(beta,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            points = res.x.reshape(N, 3)
            
            # Clip to be safe
            points = np.clip(points, -0.5, 0.5)
            
            current_radius = compute_radius(points)
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = points.copy()
                
        except Exception:
            pass

    # 3. Coordinate Descent Refinement
    # Try to improve by pushing apart binding pairs and inward wall-violating points
    for _ in range(200):
        if time.time() > deadline - 5:
            break
            
        # Find min radius and critical constraint
        min_dist = 1e9
        critical_type = None # 'pair' or 'wall'
        critical_idx = None
        
        # Check pairs
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j]) / 2.0
                if d < min_dist:
                    min_dist = d
                    critical_type = 'pair'
                    critical_idx = (i, j)
                    
        # Check walls
        for i in range(N):
            for k in range(3):
                d = 0.5 - np.abs(points[i][k])
                if d < min_dist:
                    min_dist = d
                    critical_type = 'wall'
                    critical_idx = (i, k)
                    
        if min_dist > best_radius:
            best_radius = min_dist
            best_points = points.copy()
            
        # Attempt to improve
        step = 0.001
        improved = False
        
        if critical_type == 'pair':
            i, j = critical_idx
            diff = points[i] - points[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-8:
                continue
            dir_vec = diff / norm
            
            # Try moving i out, j in
            new_i = points[i] + step * dir_vec
            new_j = points[j] - step * dir_vec
            
            # Check bounds
            if np.all(np.abs(new_i) <= 0.5) and np.all(np.abs(new_j) <= 0.5):
                # Check if this move decreases any other distance significantly
                # Simple check: does it increase the min distance?
                old_min = min_dist
                # Temporarily update
                old_i, old_j = points[i].copy(), points[j].copy()
                points[i] = new_i
                points[j] = new_j
                
                new_min = compute_radius(points)
                
                if new_min > old_min:
                    improved = True
                else:
                    points[i] = old_i
                    points[j] = old_j
                    
        elif critical_type == 'wall':
            i, k = critical_idx
            sign = np.sign(points[i][k])
            # Move inward
            new_i = points[i].copy()
            new_i[k] -= step * sign
            
            if np.all(np.abs(new_i) <= 0.5):
                old_min = min_dist
                old_i = points[i].copy()
                points[i] = new_i
                
                new_min = compute_radius(points)
                
                if new_min > old_min:
                    improved = True
                else:
                    points[i] = old_i

        if not improved:
            # Try smaller step or break
            step *= 0.5
            if step < 1e-5:
                break

    return best_points

# EVOLVE-BLOCK-END
