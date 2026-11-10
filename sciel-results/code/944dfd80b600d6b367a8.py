# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    def static_seed_14():
        # 12 Icosahedron vertices + 2 Poles
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1], [0, 0, -1]])
        return np.vstack([ico, poles])

    def gradient_descent_sphere(points, iterations=2000, lr_init=0.01, lr_decay=0.999, time_limit=10):
        """
        Minimize repulsive energy E = sum d_ij^-12 on the unit sphere.
        This tends to maximize the minimum distance.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        
        # Precompute pairs
        pairs = []
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                pairs.append((i, j))
        pairs = np.array(pairs)
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        
        lr = lr_init
        for it in range(iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Compute distances
            diff = points[i_idx] - points[j_idx]
            dist_sq = np.sum(diff**2, axis=1)
            dist = np.sqrt(dist_sq)
            
            # Avoid division by zero
            dist = np.maximum(dist, 1e-12)
            
            # Energy gradient: dE/dx_i = sum_j -12 * d_ij^-13 * (x_i - x_j)
            # Force magnitude
            force_mag = 12.0 * (dist ** -13)
            
            # Gradient accumulation
            grad = np.zeros_like(points)
            # np.add.at is slow, use scatter or loops for small N
            # N=14 is small, so we can just loop or use advanced indexing carefully
            # Actually, for N=14, 91 pairs, vectorized update is fine if we handle indices
            
            # grad[i] += force_mag * diff
            # grad[j] -= force_mag * diff
            # Using np.add.at
            np.add.at(grad, i_idx, force_mag[:, np.newaxis] * diff)
            np.add.at(grad, j_idx, -force_mag[:, np.newaxis] * diff)
            
            # Project gradient to tangent space
            # g_tangent = g - (g . x) x
            dot = np.sum(grad * points, axis=1, keepdims=True)
            grad_tangent = grad - dot * points
            
            # Update
            points = points - lr * grad_tangent
            points = project_to_sphere(points)
            
            # Decay learning rate
            lr *= lr_decay
            
        return points

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []
    
    # 1. Static high-quality seed
    inits.append(static_seed_14())
    
    # 2. Fibonacci sphere
    inits.append(fibonacci_sphere(n))
    
    # 3. Random initializations
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))
        
    # 4. Perturbed static seed
    for pert in [0.1, 0.2, 0.3]:
        np.random.seed(999 + int(pert * 100))
        inits.append(static_seed_14() + np.random.randn(n, 3) * pert)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 10:
            break

        pts = project_to_sphere(init.copy())
        
        # Optimize
        remaining = timeout - 10 - (time.time() - start_time)
        if remaining < 5:
            break
            
        # Allocate time per init
        time_per_init = min(remaining / max(1, len(inits) - init_idx), 20)
        
        optimized = gradient_descent_sphere(pts, iterations=5000, time_limit=time_per_init)
        
        s = calculate_score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized.copy()

    if best_points is None:
        best_points = static_seed_14()

    return best_points

# EVOLVE-BLOCK-END
