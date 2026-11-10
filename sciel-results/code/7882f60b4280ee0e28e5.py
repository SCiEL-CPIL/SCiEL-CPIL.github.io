import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
        n = 14
        d = 3
        start_time = time.time()
        deadline = start_time + 600  # 600s limit, keep margin

        # Initial guess: Fibonacci sphere
        def fibonacci_sphere(n):
            points = []
            phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
            theta = np.pi * (1 + 5**0.5) * np.arange(n)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            return np.vstack([x, y, z]).T

        # Objective function: Minimize negative log sum of squared distances
        # We add a small epsilon to avoid log(0)
        def objective(x):
            # x is flattened (n*d,)
            X = x.reshape(n, d)
            # Normalize to sphere? No, let the optimizer handle it via constraints,
            # or we can project. Projection is cheaper.
            # Let's stick to Cartesian with constraints in SLSQP.
            
            dists_sq = pdist(X, metric='sqeuclidean')
            # Avoid log(0)
            dists_sq = np.maximum(dists_sq, 1e-10)
            return -np.sum(np.log(dists_sq))

        # Gradient of objective
        def objective_gradient(x):
            X = x.reshape(n, d)
            # Compute gradient of sum log(d_ij^2) w.r.t X
            # d/dX_i sum_{j!=i} log(||x_i - x_j||^2)
            # = sum_{j!=i} 2 * (x_i - x_j) / ||x_i - x_j||^2
            grad = np.zeros_like(X)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = X[i] - X[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-10:
                        continue
                    grad[i] += 2 * diff / dist_sq
                    grad[j] -= 2 * diff / dist_sq
            return -grad.flatten() # Minimize negative, so flip sign

        # Constraints: ||x_i||^2 = 1 for each i
        constraints = []
        for i in range(n):
            def make_constraint(i):
                def cons(x):
                    return np.dot(x[i*d:(i+1)*d], x[i*d:(i+1)*d]) - 1.0
                return cons
            constraints.append({'type': 'eq', 'fun': make_constraint(i)})

        # Try multiple random restarts if time permits
        best_x = None
        best_val = np.inf

        for restart in range(10):
            if time.time() > deadline - 60:
                break
                
            if restart == 0:
                x0 = fibonacci_sphere(n).flatten()
            else:
                # Perturb fibonacci sphere
                x0 = fibonacci_sphere(n).flatten()
                x0 += np.random.randn(n * d) * 0.1
                # Normalize initial guess roughly to sphere
                for i in range(n):
                    norm = np.linalg.norm(x0[i*d:(i+1)*d])
                    if norm > 1e-6:
                        x0[i*d:(i+1)*d] /= norm

            try:
                res = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    jac=objective_gradient,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )
                
                if res.success:
                    val = objective(res.x)
                    if val < best_val:
                        best_val = val
                        best_x = res.x
            except Exception:
                pass

        if best_x is None:
            # Fallback
            return fibonacci_sphere(n)
            
        points = best_x.reshape(n, d)
        
        # Final polish: Local search on the sphere to maximize min distance
        # The log-sum objective is a good proxy, but we want to maximize min distance.
        # We can do a few steps of gradient ascent on the min distance (using subgradient)
        # or just return the result of the log-sum optimization.
        
        # Let's verify the metric
        return points
