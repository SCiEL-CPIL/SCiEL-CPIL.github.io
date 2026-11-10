# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    deadline = time.time() + 580.0  # Safety margin before 630s limit
    
    def score(pts):
        """Compute (d_min / d_max)^2"""
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        dmin = np.min(d)
        dmax = np.max(d)
        if dmax < 1e-12:
            return 0.0
        return (dmin / dmax) ** 2

    def coulomb_energy(pts):
        """
        Compute the Coulomb energy sum 1/d_ij^2.
        This is a smooth proxy for maximizing min distance.
        """
        d = pdist(pts)
        # Avoid division by zero
        d = np.maximum(d, 1e-10)
        return np.sum(1.0 / (d ** 2))

    def penalty(pts):
        """
        Penalty for points not being on the unit sphere.
        """
        norms_sq = np.sum(pts * pts, axis=1)
        return np.sum((norms_sq - 1.0) ** 2)

    def objective(pts):
        """
        Combined objective: Coulomb energy + sphere constraint penalty.
        """
        pts = pts.reshape(14, 3)
        return coulomb_energy(pts) + 100.0 * penalty(pts)

    def fibonacci_sphere(n_points):
        """Generate points on a Fibonacci sphere"""
        phi = np.arccos(1 - 2 * (np.arange(n_points) + 0.5) / n_points)
        theta = np.pi * (1 + 5**0.5) * np.arange(n_points)
        return np.column_stack((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ))

    def random_sphere(n_points, seed=None):
        """Generate random points on unit sphere"""
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(n_points, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        return pts / norms

    best_points = None
    best_score = 0.0

    # Generate initial configurations
    initial_configs = []
    initial_configs.append(fibonacci_sphere(n))
    for seed in range(10):
        initial_configs.append(random_sphere(n, seed=seed))
    
    # Add a specific known good structure: Icosahedron + 4 extra points perturbed
    # Icosahedron vertices
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 4 points near the "gaps" or just random perturbation of ico
    # We will just use the 12 ico points and 4 random points as a start
    extra = random_sphere(4, seed=42)
    ico_14 = np.vstack([ico, extra])
    initial_configs.append(ico_14)

    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        init_pts = init_pts.reshape(-1)
        
        try:
            # Use L-BFGS-B for fast convergence on the smooth penalty objective
            res = minimize(
                objective,
                init_pts,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            final_pts = res.x.reshape(14, 3)
            
            # Project to sphere to ensure constraint is satisfied exactly for scoring
            norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            final_pts = final_pts / norms
            
            current_score = score(final_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = final_pts.copy()
                
        except Exception:
            continue

    # Final refinement: Local search on the actual metric
    # The Coulomb minimizer finds a very good local minimum for Tammes, 
    # but we can polish the specific (dmin/dmax)^2 metric.
    if best_points is not None and time.time() < deadline - 10:
        current = best_points.copy()
        current_score = best_score
        lr = 0.005
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient of the metric? 
            # Instead, do a simple random perturbation / hill climbing
            # Or compute forces from the metric's "pressure"
            
            # Let's use a simple gradient ascent on the metric using numerical diff for a few steps
            # But numerical diff on min/max is noisy.
            
            # Better: Use the Coulomb gradient direction as a proxy for "spreading out"
            # which generally increases dmin.
            
            # Compute Coulomb forces
            forces = np.zeros_like(current)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = current[i] - current[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < 1e-10:
                        continue
                    dist = np.sqrt(dist_sq)
                    # Force is repulsive: 3 * diff / dist^4
                    forces[i] += 3.0 * diff / (dist_sq * dist)
            
            # Update
            new_pts = current + lr * forces
            # Project to sphere
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            new_pts = new_pts / norms
            
            new_score = score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current.copy()
                # Optionally increase lr slightly if improving
                # lr *= 1.01
            else:
                # Reduce learning rate
                lr *= 0.95
                if lr < 1e-6:
                    break

    if best_points is None:
        # Fallback to Fibonacci sphere if optimization failed
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
