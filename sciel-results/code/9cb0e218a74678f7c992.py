import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 110.0  # Safety margin from 120s

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Helper to compute the actual score (radius) for a set of points
    def compute_score(points):
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        # Boundary
        radii = np.sqrt(np.sum(points**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        return min(min_dist / 2.0, min_boundary)
    
    # Helper to check validity
    def is_valid(points):
        if len(points) != N:
            return False
        if not np.all(np.isfinite(points)):
            return False
        radii = np.sqrt(np.sum(points**2, axis=1))
        if np.any(radii >= 1.0):
            return False
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-10:
            return False
        return True
    
    # Initialization 1: Multi-ring structure (from inspiration 2)
    def init_multi_ring():
        points = []
        points.append([0.0, 0.0])
        
        r1 = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
        r2 = 0.5
        for i in range(12):
            angle = 2 * np.pi * i / 12 + np.pi / 12
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        
        r3 = 0.75
        for i in range(18):
            angle = 2 * np.pi * i / 18
            points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
        
        r4 = 0.92
        for i in range(13):
            angle = 2 * np.pi * i / 13 + np.pi / 13
            points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
        
        return np.array(points[:N])
    
    # Initialization 2: Hexagonal lattice (from current program, refined)
    def init_hexagonal():
        spacing = 0.25
        points = []
        k = 10
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) > N:
            radii = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(radii)[:N]
            points = points[idx]
        elif len(points) < N:
            while len(points) < N:
                angle = 2 * np.pi * len(points) / N
                points = np.vstack([points, [[0.95 * np.cos(angle), 0.95 * np.sin(angle)]]])
        
        return points[:N]
    
    # Initialization 3: Random with rejection
    def init_random(seed=42):
        np.random.seed(seed)
        points = []
        attempts = 0
        while len(points) < N and attempts < 10000:
            x = np.random.uniform(-0.9, 0.9)
            y = np.random.uniform(-0.9, 0.9)
            if x*x + y*y < 0.9**2:
                points.append([x, y])
            attempts += 1
        
        while len(points) < N:
            angle = 2 * np.pi * len(points) / N
            points.append([0.95 * np.cos(angle), 0.95 * np.sin(angle)])
        
        return np.array(points[:N])
    
    # Objective function: maximize the minimum radius (minimize negative)
    def objective(x):
        pts = x.reshape(N, 2)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        # Boundary
        radii = np.sqrt(np.sum(pts**2, axis=1))
        min_boundary = np.min(1.0 - radii)
        score = min(min_dist / 2.0, min_boundary)
        return -score
    
    # Local optimization using Nelder-Mead
    def optimize(points, max_iters=3000):
        points = points.copy()
        x0 = points.flatten()
        
        try:
            result = opt.minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': max_iters, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < objective(x0):
                optimized = result.x.reshape(N, 2)
                if is_valid(optimized):
                    return optimized
        except Exception:
            pass
        
        return points
    
    # Try different initializations and pick the best
    inits = []
    inits.append(init_multi_ring())
    inits.append(init_hexagonal())
    
    # Add random initializations with different seeds
    for seed in [0, 1, 2, 3]:
        inits.append(init_random(seed))
    
    best_score = 0.0
    best_points = None
    
    for points in inits:
        if time_left() < 15:
            break
        
        if not is_valid(points):
            # Try to fix
            radii = np.sqrt(np.sum(points**2, axis=1))
            mask = radii < 1.0
            points = points[mask]
            if len(points) < N:
                continue
        
        initial_score = compute_score(points)
        if initial_score > best_score:
            best_score = initial_score
            best_points = points.copy()
        
        # Optimize this initialization
        if time_left() < 20:
            break
        
        optimized = optimize(points, max_iters=2000)
        opt_score = compute_score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
    
    # If we still have time, do a few more random restarts from the best solution
    if best_points is not None and time_left() > 30:
        for i in range(5):
            if time_left() < 15:
                break
            
            # Perturb the best solution
            perturbed = best_points + np.random.normal(0, 0.02, best_points.shape)
            # Clip to inside
            norms = np.linalg.norm(perturbed, axis=1)
            scale = np.where(norms > 0.95, 0.95 / norms, 1.0)
            perturbed *= scale[:, np.newaxis]
            
            if is_valid(perturbed):
                optimized = optimize(perturbed, max_iters=1500)
                opt_score = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
    
    # Final verification
    if best_points is None or not is_valid(best_points):
        # Fallback to hexagonal
        return init_hexagonal()
    
    return best_points

if __name__ == "__main__":
    c = construct()
    if len(c) == N:
        print(f"Radius: {compute_score(c)}")
