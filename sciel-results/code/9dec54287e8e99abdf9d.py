# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def calculate_metric(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def energy(pts):
        dists = pdist(pts)
        # Use a softer repulsion to avoid numerical issues with very small distances
        # Sum of inverse squared distances is a good proxy for maximizing min distance
        return np.sum(1.0 / (dists ** 2))

    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return pts / norms

    def simulated_annealing(seed):
        np.random.seed(seed)
        # Initialize on sphere
        pts = np.random.randn(n, d)
        pts = project_to_sphere(pts)
        
        current_energy = energy(pts)
        best_energy = current_energy
        best_pts = pts.copy()
        
        # SA Parameters
        T_start = 10.0
        T_end = 1e-4
        n_iter = 5000
        cooling = (T_end / T_start) ** (1.0 / n_iter)
        
        for i in range(n_iter):
            T = T_start * (cooling ** i)
            
            # Propose a move: rotate a random point by a small angle
            idx = np.random.randint(n)
            # Generate a random rotation axis
            axis = np.random.randn(d)
            axis /= np.linalg.norm(axis)
            # Small angle proportional to temperature
            angle = np.random.uniform(-1.0, 1.0) * T
            
            # Apply rotation to pts[idx]
            # Rodrigues' rotation formula
            v = pts[idx]
            v_rot = v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))
            
            new_pts = pts.copy()
            new_pts[idx] = v_rot
            # Project back to sphere to maintain constraint
            new_pts = project_to_sphere(new_pts)
            
            new_energy = energy(new_pts)
            delta_e = new_energy - current_energy
            
            # Accept if better or with probability exp(-delta_e / T)
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                pts = new_pts
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
                    
        return best_pts

    best_score = -np.inf
    best_points = None
    
    # Run multiple SA restarts with different seeds
    for seed in range(20):
        pts = simulated_annealing(seed)
        score = calculate_metric(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points = project_to_sphere(best_points)

    return best_points

# EVOLVE-BLOCK-END
