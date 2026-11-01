import numpy as np
import time
import math

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    def calc_energy(pts):
        """Compute logarithmic energy sum_{i<j} log(1/d_ij)"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-18)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy = np.sum(np.log(1.0 / dist[mask]))
        return energy

    def gradient(pts):
        """Compute gradient of energy on the sphere (tangent space)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-18)
        # Gradient of sum log(1/d) = -sum log(d) = sum log(d)
        # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
        # So gradient of -sum log(d) = -sum_j (x_i - x_j)/d_ij^2
        # But we want to minimize sum log(1/d) = -sum log(d)
        # d/dx_i [-log(d_ij)] = -(x_i - x_j)/d_ij^2
        # Total gradient at point i: -sum_j (x_i - x_j)/d_ij^2 = sum_j (x_j - x_i)/d_ij^2
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        grad = forces - dot_products[:, np.newaxis] * pts
        return grad

    def optimize(pts_init, max_time, lr0=1e-3):
        """Adam-based gradient descent on the sphere."""
        pts = pts_init.copy()
        energy = calc_energy(pts)
        best_energy = energy
        best_pts = pts.copy()
        
        # Adam parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        t = 0
        stagnation = 0
        best_lr = lr0

        while time.time() < max_time:
            # Compute gradient
            grad = gradient(pts)
            
            # Adam update
            t += 1
            # Decay learning rate
            lr = best_lr * (0.9995 ** t)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            update = m_hat / (np.sqrt(v_hat) + eps)
            
            new_pts = pts - lr * update
            norms = np.sqrt(np.sum(new_pts**2, axis=1))
            new_pts = new_pts / norms[:, np.newaxis]
            
            new_energy = calc_energy(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                pts = new_pts
                stagnation = 0
            else:
                stagnation += 1
                # If energy gets significantly worse or stagnates, revert
                if new_energy > best_energy + 1.0 or stagnation > 300:
                    pts = best_pts.copy()
                    m = np.zeros_like(pts)
                    v = np.zeros_like(pts)
                    t = 0
                    stagnation = 0
                    best_lr *= 0.8
                    if best_lr < 1e-10:
                        break

        return best_pts, best_energy

    def polish_scipy(pts_init, max_time):
        """Final polish using scipy's L-BFGS-B on flattened coordinates."""
        from scipy.optimize import minimize
        
        flat = pts_init.flatten()
        
        def objective(flat_pts):
            pts = flat_pts.reshape(N, 3)
            # Project to sphere
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            return calc_energy(pts)
        
        def jac(flat_pts):
            pts = flat_pts.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            grad = gradient(pts)
            return grad.flatten()
        
        try:
            result = minimize(
                objective, flat, jac=jac, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            best_pts = result.x.reshape(N, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            best_pts = best_pts / norms
            return best_pts
        except Exception:
            return pts_init

    # Strategy: Multiple restarts with different initializations
    best_overall_energy = float('inf')
    best_overall_pts = None

    # 1. Fibonacci Spiral (Good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    t_start = time.time()
    limit1 = 50.0
    pts1, e1 = optimize(fib_pts, t_start + limit1, lr0=1e-3)
    if e1 < best_overall_energy:
        best_overall_energy = e1
        best_overall_pts = pts1

    # 2. Random Start
    if time.time() < deadline - 25:
        np.random.seed(42)
        rand_pts = np.random.randn(N, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        t_start = time.time()
        limit2 = 20.0
        pts2, e2 = optimize(rand_pts, t_start + limit2, lr0=1e-2)
        if e2 < best_overall_energy:
            best_overall_energy = e2
            best_overall_pts = pts2

    # 3. Another Random Start with different seed
    if time.time() < deadline - 15:
        np.random.seed(123)
        rand_pts3 = np.random.randn(N, 3)
        rand_pts3 /= np.linalg.norm(rand_pts3, axis=1, keepdims=True)
        t_start = time.time()
        limit3 = 10.0
        pts3, e3 = optimize(rand_pts3, t_start + limit3, lr0=1e-2)
        if e3 < best_overall_energy:
            best_overall_energy = e3
            best_overall_pts = pts3

    # 4. Halton sequence for better initial distribution
    if time.time() < deadline - 10:
        def halton_sphere(n, seed=1):
            """Generate points using Halton sequence mapped to sphere."""
            pts = []
            for i in range(n):
                # Use 3D Halton sequence
                x = _halton(i + 1, 2)
                y = _halton(i + 1, 3)
                z = _halton(i + 1, 5)
                # Map to sphere using spherical coordinates
                theta = 2 * np.pi * x
                phi = np.arccos(2 * y - 1)
                pt = np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                pts.append(pt)
            return np.array(pts)
        
        halton_pts = halton_sphere(N)
        t_start = time.time()
        limit4 = 8.0
        pts4, e4 = optimize(halton_pts, t_start + limit4, lr0=1e-3)
        if e4 < best_overall_energy:
            best_overall_energy = e4
            best_overall_pts = pts4

    # Final polish with scipy
    if best_overall_pts is not None and time.time() < deadline - 3:
        polished = polish_scipy(best_overall_pts, deadline - 1)
        polished_energy = calc_energy(polished)
        if polished_energy < best_overall_energy:
            best_overall_energy = polished_energy
            best_overall_pts = polished

    if best_overall_pts is None:
        # Fallback to Fibonacci if something went wrong
        return fib_pts

    # Ensure all points are on unit sphere
    norms = np.linalg.norm(best_overall_pts, axis=1, keepdims=True)
    best_overall_pts = best_overall_pts / norms
    
    return best_overall_pts

def _halton(index, base):
    """Generate Halton sequence value."""
    f = 1.0
    result = 0.0
    while index > 0:
        f /= base
        result += f * (index % base)
        index //= base
    return result
