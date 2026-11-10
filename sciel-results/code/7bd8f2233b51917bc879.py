"""Improved packing of 300 equal circles in a unit circle using smooth optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        for trial in range(20):
            spacing = s_est * (1.0 + trial * 0.005)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 5:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def compute_score(points):
        """Compute the minimum radius for the entire configuration using KDTree."""
        tree = cKDTree(points)
        min_dist = np.inf
        
        # Check pairwise distances using KDTree
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], 0.2)
            for j in neighbors:
                if j > i:
                    d = np.linalg.norm(points[i] - points[j])
                    if d < min_dist:
                        min_dist = d
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def soft_min_score(points, beta=50.0):
        """Compute a smooth surrogate using log-sum-exp for the min constraint."""
        tree = cKDTree(points)
        
        # Compute all pairwise distances efficiently
        pair_dists = []
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], 0.3)
            for j in neighbors:
                if j > i:
                    d = np.linalg.norm(points[i] - points[j])
                    pair_dists.append(d / 2.0)
        
        if not pair_dists:
            pair_dists = [1.0]
        
        pair_dists = np.array(pair_dists)
        
        # Boundary constraints
        radii = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - radii
        
        # Combine all constraints
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min via log-sum-exp
        max_d = np.max(all_dists)
        log_sum = np.log(np.sum(np.exp(-(all_dists - max_d) * beta))) / beta
        soft_min = max_d - log_sum
        
        return soft_min

    def soft_min_grad(points, beta=50.0):
        """Compute gradient of the soft-min surrogate."""
        tree = cKDTree(points)
        grads = np.zeros_like(points)
        
        # Compute weights for soft-min
        pair_dists = []
        pair_indices = []
        for i in range(N):
            neighbors = tree.query_ball_point(points[i], 0.3)
            for j in neighbors:
                if j > i:
                    d = np.linalg.norm(points[i] - points[j])
                    pair_dists.append(d / 2.0)
                    pair_indices.append((i, j))
        
        if not pair_dists:
            pair_dists = [1.0]
            pair_indices = [(0, 1)]
        
        pair_dists = np.array(pair_dists)
        
        # Boundary constraints
        radii = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - radii
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min weights
        max_d = np.max(all_dists)
        weights = np.exp(-(all_dists - max_d) * beta)
        weights /= np.sum(weights)
        
        # Gradient from pairwise constraints
        for k, ((i, j), w) in enumerate(zip(pair_indices, weights[:len(pair_indices)])):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d > 1e-10:
                grad_dir = d_vec / d
                grads[i] += w * grad_dir
                grads[j] -= w * grad_dir
        
        # Gradient from boundary constraints
        for i, (r_i, w) in enumerate(zip(radii, weights[len(pair_indices):])):
            if r_i > 1e-10:
                grads[i] -= w * points[i] / r_i
        
        return grads

    def gradient_optimize(points, max_iter=200, beta=50.0):
        """Gradient-based optimization using soft-min surrogate."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Compute gradient of soft-min
            grads = soft_min_grad(points, beta)
            
            # Add boundary constraint gradient
            radii = np.linalg.norm(points, axis=1)
            boundary_mask = radii > 1.0 - current_score * 1.2
            if np.any(boundary_mask):
                for i in np.where(boundary_mask)[0]:
                    if radii[i] > 1e-10:
                        grads[i] += 0.5 * (-points[i] / radii[i])
            
            # Normalize gradients
            grad_norms = np.linalg.norm(grads, axis=1)
            max_grad = np.max(grad_norms)
            if max_grad > 1e-10:
                grads /= max_grad
            
            # Update points
            points += lr * grads
            
            # Project back to unit circle
            radii = np.linalg.norm(points, axis=1)
            mask = radii > 0.999
            if np.any(mask):
                points[mask] *= 0.999 / radii[mask]
            
            # Compute score
            new_score = compute_score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                lr *= 0.95
            
            if no_improve > 20 or lr < 1e-6:
                break
        
        return best_points, best_score

    def local_optimize(points, max_iter=500):
        """Fast local optimization using KDTree for neighbor queries."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                for _ in range(3):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_x, old_y = points[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.999:
                        scale = 0.999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Use KDTree for fast neighbor distance query
                    temp_points = points.copy()
                    temp_points[i] = [new_x, new_y]
                    tree = cKDTree(temp_points)
                    
                    # Find closest neighbor
                    dists, _ = tree.query([new_x, new_y], k=2)
                    closest_dist = dists[1] if len(dists) > 1 else np.inf
                    
                    bound_dist = 1.0 - math.hypot(new_x, new_y)
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        improved = True
                        break
            
            if it % 30 == 0 or not improved:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    lr *= 0.95
                    if no_improve_count > 10 or lr < 1e-5:
                        break
            else:
                no_improve_count = 0
        
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    def high_precision_polish(points, max_iter=100):
        """Final high-precision polish with tighter tolerances."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.001
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            
            # Fine-tune using small random perturbations
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 1.0:
                    break
                
                for _ in range(5):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    new_x, new_y = points[i][0] + dx, points[i][1] + dy
                    
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.9999:
                        scale = 0.9999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Check if improvement
                    temp_points = points.copy()
                    temp_points[i] = [new_x, new_y]
                    
                    tree = cKDTree(temp_points)
                    dists, _ = tree.query([new_x, new_y], k=2)
                    closest_dist = dists[1] if len(dists) > 1 else np.inf
                    
                    bound_dist = 1.0 - math.hypot(new_x, new_y)
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score + 1e-10:
                        points[i] = [new_x, new_y]
                        break
            
            if it % 10 == 0:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    lr *= 0.9
                    if no_improve > 5 or lr < 1e-7:
                        break
        
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    # Generate initial configuration
    points = initial_hex(N)
    
    # Phase 1: Gradient-based optimization with soft-min surrogate
    points, score = gradient_optimize(points, max_iter=150, beta=50.0)
    
    # Phase 2: Local random search with KDTree
    points, score = local_optimize(points, max_iter=200)
    
    # Phase 3: Fine gradient polish
    if time.time() < deadline - 25:
        points, score2 = gradient_optimize(points, max_iter=80, beta=100.0)
        if score2 > score:
            score = score2
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        points, score3 = high_precision_polish(points, max_iter=80)
        if score3 > score:
            score = score3
    
    return points
# EVOLVE-BLOCK-END
