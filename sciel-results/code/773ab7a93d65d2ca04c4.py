"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # --- Energy computation (vectorized) ---
    def energy(pts):
        # pts: (N,3) unit vectors
        # Compute all pairwise chordal distances
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
        dists = np.linalg.norm(diff, axis=2)  # (N,N)
        # Mask diagonal
        np.fill_diagonal(dists, 1.0)
        # Energy = sum_{i<j} log(1/d) = -sum_{i<j} log(d)
        # Use upper triangle
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        # Avoid log(0) - shouldn't happen but safety
        d = np.maximum(d, 1e-15)
        return -np.sum(np.log(d))

    # --- Gradient of energy w.r.t. points on sphere ---
    def gradient(pts):
        # E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
        # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # This is the gradient in R^3; the constrained gradient (tangent to sphere)
        # is grad - (grad . x_i) * x_i
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
        dists_sq = np.sum(diff * diff, axis=2)  # (N,N)
        np.fill_diagonal(dists_sq, 1.0)
        # grad[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
        # = sum_j diff[i,j] / dists_sq[i,j]
        inv_d2 = 1.0 / dists_sq  # (N,N)
        np.fill_diagonal(inv_d2, 0.0)
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N,3)
        # Project to tangent space
        proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        return proj

    # --- Fibonacci spiral initial point ---
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_pts = pts.copy()
    best_e = energy(pts)

    # --- Conjugate gradient / L-BFGS style optimization on sphere ---
    # We'll use a simple projected gradient descent with line search,
    # then a higher-precision polish.
    
    # Use scipy's minimize if available for better convergence
    from scipy.optimize import minimize

    def obj_and_grad(x_flat):
        pts_ = x_flat.reshape(N, 3)
        e = energy(pts_)
        g = gradient(pts_).flatten()
        return e, g

    # Run multiple restarts with different initial conditions
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = pts.copy()
        else:
            # Perturbed Fibonacci
            x0 = pts.copy()
            noise = rng.normal(0, 0.05, size=(N, 3))
            x0 = x0 + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        
        try:
            res = minimize(
                lambda x: energy(x.reshape(N, 3)),
                x0.flatten(),
                jac=lambda x: gradient(x.reshape(N, 3)).flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts_ = res.x.reshape(N, 3)
                # Re-normalize
                norms = np.linalg.norm(pts_, axis=1)
                pts_ = pts_ / norms[:, np.newaxis]
                e = energy(pts_)
                if e < best_e:
                    best_e = e
                    best_pts = pts_.copy()
        except Exception:
            pass

    # --- High-precision polish: Newton-like / steepest descent with very tight tolerance ---
    if time.time() < deadline - 5:
        pts = best_pts.copy()
        lr = 1e-4
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            g = gradient(pts)
            gnorm = np.linalg.norm(g)
            if gnorm < 1e-14:
                break
            # Backtracking line search
            step = lr
            for _ in range(30):
                new_pts = pts + step * g
                # Project back to sphere
                new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_e = energy(new_pts)
                if new_e < energy(pts):
                    pts = new_pts
                    break
                step *= 0.5
            else:
                break
            # Adapt learning rate
            if gnorm < 1e-10:
                lr *= 0.5
        e = energy(pts)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()

    # Final normalization and validation
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate: no points closer than 1e-9
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
# EVOLVE-BLOCK-END
