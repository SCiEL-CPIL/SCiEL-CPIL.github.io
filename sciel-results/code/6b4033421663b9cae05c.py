"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

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
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def get_bottlenecks(points, r_target):
        """Find points that are limiting the score (within r_target of others or boundary)."""
        tree = cKDTree(points)
        bottleneck_idx = []
        
        for i in range(N):
            # Check boundary
            r_i = np.linalg.norm(points[i])
            if 1.0 - r_i <= r_target * 1.01:
                bottleneck_idx.append(i)
                continue
            
            # Check neighbors within 2*r_target
            neighbors = tree.query_ball_point(points[i], 2.0 * r_target)
            for j in neighbors:
                if j != i:
                    d = np.linalg.norm(points[i] - points[j])
                    if d <= 2.0 * r_target * 1.01:
                        bottleneck_idx.append(i)
                        break
        
        return list(set(bottleneck_idx))

    def gradient_optimize(points, max_iter=200):
        """Gradient-based optimization using subgradients of the min distance."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.005
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            # Build KDTree for fast neighbor queries
            tree = cKDTree(points)
            
            # Compute subgradient for each point
            grads = np.zeros_like(points)
            
            for i in range(N):
                r_i = np.linalg.norm(points[i])
                
                # Boundary force: push inward if close to boundary
                if r_i > 1.0 - current_score * 1.5:
                    if r_i > 1e-10:
                        grads[i] += -0.5 * points[i] / r_i
                    else:
                        # At center, push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        grads[i] += 0.5 * np.array([math.cos(angle), math.sin(angle)])
                
                # Find close neighbors
                neighbors = tree.query_ball_point(points[i], 2.0 * current_score * 1.5)
                
                for j in neighbors:
                    if j == i:
                        continue
                    d_vec = points[i] - points[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-10:
                        # If too close, push apart
                        if d < 2.0 * current_score:
                            strength = (2.0 * current_score - d) / (2.0 * current_score)
                            grads[i] += strength * d_vec / d
                        elif d < 2.0 * current_score * 1.5:
                            # Mild repulsion for near-miss
                            strength = 0.1 * (2.0 * current_score * 1.5 - d) / (2.0 * current_score * 1.5)
                            grads[i] += strength * d_vec / d
            
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
        """Fast local optimization using O(N) score for single point updates."""
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
                
                for _ in range(2):
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
                    
                    # Compute distance to closest neighbor for new pos
                    diffs = points - np.array([new_x, new_y])
                    dists = np.linalg.norm(diffs, axis=1)
                    dists[i] = np.inf
                    closest_dist = np.min(dists)
                    
                    r_new = math.hypot(new_x, new_y)
                    bound_dist = 1.0 - r_new
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        improved = True
                        break
            
            if it % 50 == 0 or not improved:
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

    # Generate initial configuration
    points = initial_hex(N)
    
    # Phase 1: Gradient-based optimization
    points, score = gradient_optimize(points, max_iter=100)
    
    # Phase 2: Local random search
    points, score = local_optimize(points, max_iter=150)
    
    # Phase 3: Fine gradient polish
    if time.time() < deadline - 20:
        points, score2 = gradient_optimize(points, max_iter=50)
        if score2 > score:
            score = score2
    
    # Phase 4: Final local polish
    if time.time() < deadline - 5:
        points, score3 = local_optimize(points, max_iter=30)
        if score3 > score:
            score = score3
    
    return points
# EVOLVE-BLOCK-END
