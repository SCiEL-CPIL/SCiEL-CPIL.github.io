import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing to minimize a high-order inverse distance energy, 
    followed by gradient descent refinement.
    """
    n = 14
    
    # 1. Initialize points using Fibonacci Sphere for uniform distribution
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden_ratio
        points[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
        
    # Normalize to unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Simulated Annealing
    # Energy function: sum(1/d^10) to strongly penalize close pairs
    def energy(pts):
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-10)
        return np.sum(1.0 / (dists ** 10))
    
    current_energy = energy(points)
    best_points = points.copy()
    best_energy = current_energy
    
    T = 5000.0
    cooling_rate = 0.9995
    min_T = 1.0
    steps_per_T = 20
    step_size = 0.1
    
    np.random.seed(42) # For reproducibility
    
    while T > min_T:
        for _ in range(steps_per_T):
            idx = np.random.randint(n)
            delta = np.random.randn(3)
            radial = np.dot(delta, points[idx]) * points[idx]
            delta -= radial
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-10:
                continue
            delta /= delta_norm
            
            new_points = points.copy()
            new_points[idx] = points[idx] + step_size * delta
            new_points[idx] /= np.linalg.norm(new_points[idx])
            
            new_energy = energy(new_points)
            
            if new_energy < current_energy:
                points = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = points.copy()
            else:
                delta_E = new_energy - current_energy
                prob = np.exp(-delta_E / T)
                if np.random.random() < prob:
                    points = new_points
                    current_energy = new_energy
                    
        T *= cooling_rate
        # Adaptive step size
        step_size *= 0.999
        
    points = best_points
    
    # 3. Gradient Descent Refinement
    # Minimize Thomson Energy (k=2) for smooth convergence
    lr = 0.01
    for step in range(2000):
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    continue
                f_mag = 2.0 / (dist_sq * dist_sq)
                f_vec = diff * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec
        
        for i in range(n):
            radial = np.dot(forces[i], points[i]) * points[i]
            forces[i] -= radial
            
        points += lr * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        if step % 500 == 0:
            lr *= 0.5
            
    # Final centering and normalization
    points -= np.mean(points, axis=0)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
