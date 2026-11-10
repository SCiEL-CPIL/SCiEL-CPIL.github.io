import numpy as np
import time
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Energy function (Cartesian)
    def energy(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        energy_val = -np.sum(np.log(dist[mask]))
        return energy_val

    # Gradient function (Cartesian) - NOT USED FOR SCIPY IF WE USE SPHERICAL
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        return grad

    # Helper: Cartesian to Spherical (theta, phi)
    # theta: azimuthal angle [0, 2*pi)
    # phi: polar angle [0, pi]
    def cartesian_to_spherical(X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        theta = np.arctan2(y, x)
        phi = np.arccos(z / np.linalg.norm(X, axis=1))
        return np.column_stack([theta, phi])

    # Helper: Spherical to Cartesian
    def spherical_to_cartesian(params):
        theta, phi = params[:, 0], params[:, 1]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])

    # Energy and Gradient in Spherical Coordinates
    def energy_and_grad_spherical(params):
        X = spherical_to_cartesian(params)
        # Compute Cartesian energy
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        e = -np.sum(np.log(dist[mask]))
        
        # Compute Cartesian Gradient
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad_cart = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project Cartesian Gradient to Spherical Tangent Space
        # The Jacobian of the spherical parameterization maps (dtheta, dphi) to dX
        # We need the gradient in the parameter space: grad_params = J^T grad_cart
        # J for point i:
        # dx/dtheta = -sin(phi)sin(theta)
        # dy/dtheta = sin(phi)cos(theta)
        # dz/dtheta = 0
        # dx/dphi = cos(phi)cos(theta)
        # dy/dphi = cos(phi)sin(theta)
        # dz/dphi = -sin(phi)
        
        theta, phi = params[:, 0], params[:, 1]
        
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Avoid division by zero or small numbers in Jacobian inverse?
        # We are doing J^T * g_cart.
        
        grad_theta = -sin_phi * sin_theta * grad_cart[:, 0] + sin_phi * cos_theta * grad_cart[:, 1]
        grad_phi = cos_phi * cos_theta * grad_cart[:, 0] + cos_phi * sin_theta * grad_cart[:, 1] - sin_phi * grad_cart[:, 2]
        
        # Handle singularities at poles where sin(phi) is 0
        # If sin(phi) is very small, the theta gradient is ill-defined.
        # We can mask it or just let it be 0 if sin(phi) is tiny.
        mask_pole = sin_phi < 1e-6
        grad_theta[mask_pole] = 0.0
        
        grad_params = np.column_stack([grad_theta, grad_phi])
        
        return e, grad_params.flatten()

    def optimize_from_initial(X_init, time_limit):
        # Convert to spherical
        params_init = cartesian_to_spherical(X_init)
        
        # Flatten for scipy
        x0 = params_init.flatten()
        
        # Define the fun for scipy (returns scalar)
        def fun(x):
            e, _ = energy_and_grad_spherical(x.reshape(N, 2))
            return e
            
        # Define jac for scipy (returns 1D array)
        def jac(x):
            _, g = energy_and_grad_spherical(x.reshape(N, 2))
            return g
            
        # Use L-BFGS-B
        # maxiter is a soft limit, we also check time
        # We don't know how many iterations fit in time_limit, so we set a high limit
        # and rely on the timeout in the outer loop? No, minimize blocks.
        # We can use a callback to check time.
        
        state = {'time_limit': time_limit, 'start': time.time()}
        
        def callback(xk):
            if time.time() > state['time_limit']:
                # Raise an exception to break out of minimize?
                # Or just return False? L-BFGS-B might not support early stop via callback return.
                # Actually, scipy minimize callback return value is ignored in L-BFGS-B?
                # Let's check. In recent scipy, returning False from callback does NOT stop L-BFGS-B.
                # We need a different strategy.
                # We can set maxiter to a reasonable number and hope it finishes.
                # Or we can just run it and hope it's fast enough.
                # L-BFGS-B is very fast. 150 points -> 300 variables.
                # It should converge in < 1 second usually.
                pass
        
        try:
            res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                best_params = res.x.reshape(N, 2)
                best_X = spherical_to_cartesian(best_params)
                # Normalize to ensure unit sphere
                best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
                return best_X, res.fun
            else:
                # If it fails, return the best point found so far?
                # res.x is the last point.
                best_params = res.x.reshape(N, 2)
                best_X = spherical_to_cartesian(best_params)
                best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
                return best_X, energy(best_X)
        except Exception as e:
            # Fallback to initial
            return X_init, energy(X_init)

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_overall_X = X_fib
    best_overall_energy = energy(X_fib)
    
    # Try a few random initializations as well
    rng = np.random.default_rng(42)
    
    # Allocate time for 5 restarts
    # First restart with Fibonacci
    time_per_restart = (deadline - time.time()) / 5.0
    
    X_best, e_best = optimize_from_initial(X_fib, time.time() + time_per_restart)
    if e_best < best_overall_energy:
        best_overall_energy = e_best
        best_overall_X = X_best

    for k in range(4):
        if time.time() > deadline:
            break
        
        # Random initialization
        X_rand = rng.normal(size=(N, 3))
        X_rand = X_rand / np.linalg.norm(X_rand, axis=1, keepdims=True)
        
        time_per_restart = (deadline - time.time()) / (5 - k)
        
        X_best, e_best = optimize_from_initial(X_rand, time.time() + time_per_restart)
        if e_best < best_overall_energy:
            best_overall_energy = e_best
            best_overall_X = X_best

    return best_overall_X
