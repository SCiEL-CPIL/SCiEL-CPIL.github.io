import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    def energy_and_grad(points):
        """
        Compute logarithmic energy and its Riemannian gradient.
        E = sum_{i<j} log(1/||x_i - x_j||)
        """
        # points: (N, 3)
        # Compute pairwise differences
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        # Compute squared distances
        dists_sq = np.sum(diff ** 2, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists_sq, 1.0)
        
        # Distances
        dists = np.sqrt(dists_sq)
        
        # Energy
        # E = -0.5 * sum_{i!=j} log(d_ij)
        log_dists = np.log(dists)
        np.fill_diagonal(log_dists, 0.0)
        E = -0.5 * np.sum(log_dists)
        
        # Gradient in Euclidean space
        # G_k = sum_{j != k} (x_k - x_j) / dist_kj^2
        # inv_dists_sq: (N, N)
        inv_dists_sq = 1.0 / dists_sq
        
        # Weighted differences
        # diff[k, j] is x_k - x_j
        # We want sum_j diff[k, j] * inv_dists_sq[k, j]
        weighted_diff = diff * inv_dists_sq[:, :, np.newaxis]
        G = np.sum(weighted_diff, axis=1)
        
        # Project to tangent space
        # g_k = G_k - (G_k . x_k) x_k
        dot_products = np.sum(G * points, axis=1)
        g = G - dot_products[:, np.newaxis] * points
        
        return E, g

    def retraction(points, direction, alpha):
        """
        Retraction on the sphere: normalize the updated point.
        """
        new_points = points + alpha * direction
        norms = np.linalg.norm(new_points, axis=1)
        # Handle potential zero norms (unlikely with good alpha)
        norms[norms < 1e-12] = 1.0
        return new_points / norms[:, np.newaxis]

    def line_search(points, direction, grad, E, max_iter=30):
        """
        Armijo line search.
        """
        alpha = 1.0
        c1 = 1e-4
        # Slope at current point
        slope = np.sum(grad * direction)
        
        for _ in range(max_iter):
            new_points = retraction(points, direction, alpha)
            new_E, new_grad = energy_and_grad(new_points)
            
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad(x)^T d
            if new_E <= E + c1 * alpha * slope:
                return alpha, new_points, new_E, new_grad
            
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        # If no improvement found, return original
        return alpha, points, E, grad

    def halton_init(N):
        """
        Fibonacci spiral initialization.
        """
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_rotation(points):
        """
        Apply a random rotation to the points.
        """
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        d = np.sign(np.linalg.det(Q))
        Q[:, 0] *= d
        return Q @ points.T

    best_E = np.inf
    best_points = None

    # Run multiple restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        # Initialize points
        if restart == 0:
            points = halton_init(N)
        else:
            # Perturb best solution or use new random rotation
            if best_points is not None and restart % 2 == 0:
                points = random_rotation(best_points)
                # Add small noise to escape local minima
                noise = np.random.randn(N, 3) * 0.05
                points = points + noise
                points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
            else:
                points = halton_init(N)
                points = random_rotation(points)
        
        E, g = energy_and_grad(points)
        
        # Riemannian Conjugate Gradient
        p = -g
        # Project p to tangent space
        p = p - np.sum(p * points, axis=1)[:, np.newaxis] * points
        
        g_old = g.copy()
        
        for it in range(500):
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # Line search
            alpha, points, E, g = line_search(points, p, g, E)
            
            # Update best
            if E < best_E:
                best_E = E
                best_points = points.copy()
                
            # Check convergence
            grad_norm = np.linalg.norm(g)
            if grad_norm < 1e-9:
                break
                
            # Update CG direction (Polak-Ribiere)
            # beta = (g_new . (g_new - g_old)) / (g_old . g_old)
            g_old_dot_g_new = np.sum(g_old * g)
            g_old_dot_g_old = np.sum(g_old * g_old)
            
            if g_old_dot_g_old > 1e-12:
                beta = (g_old_dot_g_new - np.sum(g_old * g_old)) / g_old_dot_g_old
                # Actually PR beta is (g_new - g_old).g_new / g_old.g_old
                # Let's use standard PR: beta = (g_new . (g_new - g_old)) / (g_old . g_old)
                beta = (np.sum(g * (g - g_old))) / g_old_dot_g_old
            else:
                beta = 0.0
            
            # Restart if beta is negative
            if beta < 0:
                beta = 0.0
                
            p = -g + beta * p
            # Project p to tangent space
            p = p - np.sum(p * points, axis=1)[:, np.newaxis] * points
            
            g_old = g.copy()

    return best_points
