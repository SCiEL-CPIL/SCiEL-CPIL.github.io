# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

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

    def min_dist_from_flat(x):
        """Compute minimum pairwise distance from flat array of n*3 coords."""
        pts = x.reshape(n, 3)
        dists = pdist(pts)
        return np.min(dists)

    def neg_min_dist(x):
        """Negative of minimum distance for minimization."""
        return -min_dist_from_flat(x)

    def neg_min_dist_grad(x):
        """Gradient of negative min distance (approximate via finite differences)."""
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (neg_min_dist(x_plus) - neg_min_dist(x_minus)) / (2 * eps)
        return grad

    def optimize_with_lbfgs(points, max_iter=500, tol=1e-8):
        """Use L-BFGS-B to maximize minimum distance on sphere."""
        pts = project_to_sphere(points.copy())
        x0 = pts.flatten()
        
        # Use SLSQP with sphere constraints for better performance
        def sphere_constraint(x):
            pts = x.reshape(n, 3)
            norms_sq = np.sum(pts * pts, axis=1)
            return norms_sq - 1.0
        
        def sphere_constraint_jac(x):
            pts = x.reshape(n, 3)
            jac = np.zeros((n, n * 3))
            for i in range(n):
                jac[i, i*3:(i+1)*3] = 2 * pts[i]
            return jac
        
        result = minimize(
            neg_min_dist,
            x0,
            method='SLSQP',
            jac=neg_min_dist_grad,
            constraints=[{'type': 'eq', 'fun': sphere_constraint, 'jac': sphere_constraint_jac}],
            options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
        )
        
        if result.success:
            return project_to_sphere(result.x.reshape(n, 3))
        else:
            return project_to_sphere(result.x.reshape(n, 3))

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

    def icosahedron_based_init():
        """Create initial points based on icosahedron vertices + perturbations."""
        t = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ], dtype=float)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add 2 more points to make 14
        np.random.seed(42)
        extra1 = np.random.randn(3)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(3)
        extra2 = extra2 / np.linalg.norm(extra2)
        
        return np.vstack([vertices, extra1.reshape(1,3), extra2.reshape(1,3)])

    def random_init(seed):
        np.random.seed(seed)
        pts = np.random.randn(n, 3)
        return project_to_sphere(pts)

    best_score = 0.0
    best_points = None

    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(fibonacci_sphere(n))
    
    # 2. Icosahedron-based
    inits.append(icosahedron_based_init())
    
    # 3. Multiple random inits
    for seed in [1, 2, 3, 7, 13, 21, 34, 55, 89, 144]:
        inits.append(random_init(seed))
    
    # 4. Perturbed fibonacci
    np.random.seed(123)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.15)
    
    np.random.seed(456)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.3)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break
        
        pts = project_to_sphere(init.copy())
        
        # Optimize using SLSQP
        if time.time() - start_time < timeout - 60:
            optimized = optimize_with_lbfgs(pts, max_iter=300, tol=1e-10)
            s = calculate_score(optimized)
            if s > best_score:
                best_score = s
                best_points = optimized.copy()
        
        # Also try with different max_iter for some
        if init_idx < 5 and time.time() - start_time < timeout - 90:
            optimized2 = optimize_with_lbfgs(pts, max_iter=800, tol=1e-12)
            s2 = calculate_score(optimized2)
            if s2 > best_score:
                best_score = s2
                best_points = optimized2.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
