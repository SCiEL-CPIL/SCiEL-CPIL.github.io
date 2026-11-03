import numpy as np
import time

N = 150

def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists = np.maximum(dists, 1e-12)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return -np.sum(np.log(dists[mask]))

def _log_energy_grad(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    dists_sq = np.maximum(dists_sq, 1e-12)
    mask = ~np.eye(N, dtype=bool)
    inv_dists_sq = np.zeros_like(dists_sq)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    term = diff * inv_dists_sq[:, :, np.newaxis]
    grad_sum = np.sum(term, axis=1)
    return -grad_sum

def _optimize(points, time_limit):
    start_time = time.time()
    alpha = 0.1
    min_alpha = 1e-7
    max_alpha = 0.5
    best_energy = _log_energy(points)
    best_points = points.copy()
    no_improve = 0
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        grad = _log_energy_grad(points)
        dots = np.sum(grad * points, axis=1)
        grad_tan = grad - dots[:, np.newaxis] * points
        
        new_points = points - alpha * grad_tan
        norms = np.linalg.norm(new_points, axis=1)
        norms = np.maximum(norms, 1e-12)
        new_points = new_points / norms[:, np.newaxis]
        
        new_energy = _log_energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()
            points = new_points
            no_improve = 0
            if alpha < max_alpha:
                alpha *= 1.02
        else:
            no_improve += 1
            alpha *= 0.95
            if alpha < min_alpha:
                alpha = min_alpha
                if no_improve > 100:
                    break
                    
    return best_points, best_energy

def construct():
    total_time_limit = 145.0
    start_time = time.time()
    
    best_points = None
    best_energy = float('inf')
    
    starts = [
        _fibonacci_sphere(N),
        _fibonacci_sphere(N, offset=0.1),
        _fibonacci_sphere(N, offset=0.3),
        _fibonacci_sphere(N, offset=0.5),
        _fibonacci_sphere(N, offset=0.7),
    ]
    
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        starts.append(pts)
        
    num_starts = len(starts)
    
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        if remaining < 5.0:
            break
            
        remaining_starts = num_starts - i
        alloc_time = min(20.0, remaining / remaining_starts)
        
        if alloc_time < 1.0:
            continue
            
        pts, energy = _optimize(start_pts, time_limit=alloc_time)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts
            
    if best_points is None:
        best_points = _fibonacci_sphere(N)
        
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points
