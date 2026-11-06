import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def fibonacci_disk(n, radius=1.0):
        points = np.zeros((n, 2))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = radius * np.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        return points

    def calc_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.sqrt(np.sum(points**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def get_gradient_and_obj(points, beta):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        radii = np.sqrt(np.sum(points**2, axis=1))
        boundary_dists = 1.0 - radii
        
        pair_terms = np.exp(-beta * dists)
        np.fill_diagonal(pair_terms, 0.0)
        
        boundary_terms = np.exp(-beta * boundary_dists)
        
        sum_exp = np.sum(pair_terms) + np.sum(boundary_terms)
        obj = -1.0 / beta * np.log(sum_exp)
        
        inv_dists = 1.0 / dists
        np.fill_diagonal(inv_dists, 0.0)
        
        weights = pair_terms * inv_dists
        sum_w = np.sum(weights, axis=1)
        sum_wx = weights @ points
        
        sum_wx_diff = sum_w[:, np.newaxis] * points - sum_wx
        
        boundary_grad = boundary_terms * points / np.where(radii[:, np.newaxis] > 1e-10, radii[:, np.newaxis], 1.0)
        
        grad = (sum_wx_diff - boundary_grad) / sum_exp
        
        return obj, grad

    best_points = fibonacci_disk(N)
    best_score = calc_score(best_points)
    
    # Run optimization
    # We can do a few restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial points slightly for restarts > 0
        if restart == 0:
            points = best_points.copy()
        else:
            points = best_points + np.random.normal(0, 0.01, (N, 2))
            # Project back to disk
            radii = np.sqrt(np.sum(points**2, axis=1))
            mask = radii > 1.0
            points[mask] = points[mask] / radii[mask, np.newaxis] * 0.999
            
        beta = 50.0
        learning_rate = 0.05
        num_iterations = 2000
        
        for iter in range(num_iterations):
            if time.time() > deadline:
                break
                
            obj, grad = get_gradient_and_obj(points, beta)
            
            # Simple adaptive step size? 
            # For now, fixed decay
            lr = learning_rate * (0.99 ** iter)
            
            points += lr * grad
            
            # Project to disk
            radii = np.sqrt(np.sum(points**2, axis=1))
            mask = radii > 1.0
            if np.any(mask):
                points[mask] = points[mask] / radii[mask, np.newaxis] * 0.999
            
            if iter % 10 == 0:
                current_score = calc_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            
            beta *= 1.001

    return best_points
