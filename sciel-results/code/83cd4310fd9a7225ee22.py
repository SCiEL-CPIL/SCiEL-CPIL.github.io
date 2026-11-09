import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    
    best_points = None
    best_score = -np.inf

    # Helper to compute exact score
    def compute_score(points):
        # points: (N, 2)
        # Check validity
        if not np.all(np.isfinite(points)):
            return -np.inf
        if points.shape != (N, 2):
            return -np.inf
        
        # Distances between centers
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                dists.append(d / 2.0)
        
        # Distances to boundary
        for i in range(N):
            d = 1.0 - np.linalg.norm(points[i])
            dists.append(d)
            
        if not dists:
            return -np.inf
        return min(dists)

    # Soft min objective and gradient
    def soft_min_obj_and_grad(x_flat, beta):
        points = x_flat.reshape(N, 2)
        
        # Calculate all constraints v_k
        # We will accumulate sum_exp and grads
        sum_exp = 0.0
        grad = np.zeros_like(x_flat)
        
        # Pairwise distances
        for i in range(N):
            for j in range(i+1, N):
                diff = points[i] - points[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                v = d / 2.0
                exp_v = np.exp(-beta * v)
                sum_exp += exp_v
                
                # Gradient of v w.r.t points[i] and points[j]
                # d(v)/d(points[i]) = (points[i] - points[j]) / (2 * d)
                grad_vec = diff / (2.0 * d)
                grad[i*2:i*2+2] += exp_v * grad_vec
                grad[j*2:j*2+2] += exp_v * (-grad_vec)
                
        # Boundary distances
        for i in range(N):
            p = points[i]
            norm_p = np.linalg.norm(p)
            if norm_p < 1e-12:
                norm_p = 1e-12
            v = 1.0 - norm_p
            exp_v = np.exp(-beta * v)
            sum_exp += exp_v
            
            # Gradient of v w.r.t points[i] = -p / norm_p
            grad_vec = -p / norm_p
            grad[i*2:i*2+2] += exp_v * grad_vec
            
        # Objective f = -1/beta * ln(sum_exp)
        # We want to maximize f, so minimize -f
        # -f = 1/beta * ln(sum_exp)
        obj = (1.0 / beta) * np.log(sum_exp)
        
        # Gradient of -f w.r.t x
        # d(-f)/dx = (1/beta) * (1/sum_exp) * sum(d(exp_v)/dx)
        # d(exp_v)/dx = exp_v * (-beta) * dv/dx
        # So d(-f)/dx = (1/beta) * (1/sum_exp) * sum(exp_v * (-beta) * dv/dx)
        #             = -(1/sum_exp) * sum(exp_v * dv/dx)
        # Note: our 'grad' variable accumulated sum(exp_v * dv/dx)
        grad = -grad / sum_exp
        
        return obj, grad

    def optimize_initial(points_init, time_slice):
        # points_init: (N, 2)
        # Returns best points found during optimization
        x0 = points_init.flatten()
        
        # Start with a moderate beta, increase it
        beta = 10.0
        
        best_local_points = points_init.copy()
        best_local_score = compute_score(points_init)
        
        # Simple iterative refinement with increasing beta
        # We use a fixed number of steps or time-based
        current_time = time.time()
        
        # Phase 1: Lower beta for global exploration
        for _ in range(5):
            if time.time() - current_time > time_slice:
                break
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, beta)[0],
                x0,
                jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            x_opt = res.x
            points_opt = x_opt.reshape(N, 2)
            score = compute_score(points_opt)
            if score > best_local_score:
                best_local_score = score
                best_local_points = points_opt.copy()
                x0 = x_opt
            beta *= 1.5
            
        # Phase 2: High beta for local polish
        beta = 50.0
        for _ in range(5):
            if time.time() - current_time > time_slice:
                break
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, beta)[0],
                x0,
                jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            x_opt = res.x
            points_opt = x_opt.reshape(N, 2)
            score = compute_score(points_opt)
            if score > best_local_score:
                best_local_score = score
                best_local_points = points_opt.copy()
                x0 = x_opt
            beta *= 1.5
            
        return best_local_points, best_local_score

    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    spacing = 0.25
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing/2:
                points.append([x, y])
    if len(points) >= N:
        # Take N closest to center or random? Closest to center is usually better for packing density near center
        points = np.array(points)
        radii = np.hypot(points[:,0], points[:,1])
        idx = np.argsort(radii)[:N]
        inits.append(points[idx])
        
    # 2. Random uniform inside disk
    np.random.seed(42)
    for _ in range(5):
        pts = []
        for _ in range(N):
            r = np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2*np.pi)
            pts.append([r*np.cos(theta), r*np.sin(theta)])
        inits.append(np.array(pts))
        
    # 3. Ring based
    for _ in range(5):
        pts = []
        # Center
        pts.append([0,0])
        # Rings
        radii = [0.2, 0.4, 0.6, 0.8]
        counts = [6, 12, 18, 13] # Sum = 50
        for r, c in zip(radii, counts):
            for i in range(c):
                theta = 2*np.pi*i/c
                pts.append([r*np.cos(theta), r*np.sin(theta)])
        inits.append(np.array(pts[:N]))
        
    # Optimize each init
    total_time = time_limit - safety_margin
    time_per_init = total_time / len(inits)
    
    for init in inits:
        if time.time() - start_time > total_time:
            break
        res_points, res_score = optimize_initial(init, time_per_init)
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
            
    if best_points is None:
        # Fallback to simple hex if something went wrong
        return np.array([[0,0]]) # Invalid shape, but prevents crash. Actually need valid shape.
        # Let's return the first init if optimization failed completely
        if inits:
            best_points = inits[0]
        else:
            best_points = np.zeros((N, 2))
            
    return best_points
