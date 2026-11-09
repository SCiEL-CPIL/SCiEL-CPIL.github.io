import numpy as np
import math
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def get_fibonacci_points(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def get_random_points(n, seed):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize(pts, max_steps, time_limit, initial_step=0.01, min_step=1e-7):
        """Optimize with backtracking line search for guaranteed descent."""
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        step = initial_step
        
        for step_idx in range(max_steps):
            if time.time() > time_limit:
                break
            
            grad = gradient(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Direction: negative gradient
            direction = -grad / grad_norm
            
            # Backtracking line search
            alpha = step
            e_current = best_e
            improved = False
            
            for backtrack in range(30):
                pts_new = normalize(pts + alpha * direction * grad_norm)
                e_new = log_energy(pts_new)
                
                if e_new < e_current - 1e-10:
                    improved = True
                    break
                alpha *= 0.5
            
            if improved:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                # Increase step slightly if we improved
                step = min(step * 1.1, 0.5)
            else:
                # Shrink step
                step *= 0.5
                if step < min_step:
                    break
        
        return best_pts, best_e

    def optimize_momentum(pts, max_steps, time_limit, initial_step=0.01, momentum=0.9, min_step=1e-7):
        """Optimize with momentum and adaptive step size."""
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        step = initial_step
        v = np.zeros_like(pts)
        
        for step_idx in range(max_steps):
            if time.time() > time_limit:
                break
                
            grad = gradient(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            v = momentum * v - step * grad
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                v = v / v_norm * min(step, 0.1)
            
            pts_new = normalize(pts + v)
            e_new = log_energy(pts_new)
            
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                pts = pts_new
                # Slightly increase step
                step = min(step * 1.05, 0.1)
            else:
                # Revert and shrink step
                step *= 0.7
                v *= 0.5
                if step < min_step:
                    break
        
        return best_pts, best_e

    # Phase 1: Multiple random restarts with coarse optimization
    best_overall_pts = None
    best_overall_e = float('inf')
    
    # Start with Fibonacci
    initial_configs = [get_fibonacci_points(N)]
    # Add a few random starts
    for seed in [42, 123, 456]:
        initial_configs.append(get_random_points(N, seed))
    
    time_per_config = (deadline - start_time) / (len(initial_configs) * 2)
    
    for config_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Coarse optimization
        pts, e = optimize_momentum(init_pts, 2000, start_time + time_per_config, 
                                    initial_step=0.05, momentum=0.9, min_step=1e-4)
        
        # Fine optimization
        pts, e = optimize(pts, 3000, start_time + 2 * time_per_config,
                          initial_step=0.01, min_step=1e-6)
        
        if e < best_overall_e:
            best_overall_e = e
            best_overall_pts = pts.copy()
    
    # Phase 2: Polishing the best configuration
    if best_overall_pts is not None:
        pts, e2 = optimize(best_overall_pts, 5000, start_time + deadline - 10,
                          initial_step=1e-3, min_step=1e-8)
        if e2 < best_overall_e:
            best_overall_e = e2
            best_overall_pts = pts
    
    # Final cleanup
    best_pts = normalize(best_overall_pts)
    
    # Check for invalid states
    if not np.all(np.isfinite(best_pts)):
        return get_fibonacci_points(N)

    # Verify minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        rng = np.random.RandomState(42)
        for i in range(N):
            for j in range(i+1, N):
                if dist_sq[i,j] < 1e-18:
                    direction = rng.randn(3)
                    direction -= np.dot(direction, best_pts[i]) * best_pts[i]
                    best_pts[i] += 1e-8 * direction
                    best_pts[i] /= np.linalg.norm(best_pts[i])

    return best_pts

# EVOLVE-BLOCK-END
