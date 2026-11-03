import numpy as np
import time
import math

N = 100

def compute_energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    X is an (N, 3) array.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||)
    Grad w.r.t x_i: - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    N = X.shape[0]
    # Compute pairwise distances
    # X[:, np.newaxis, :] - X[np.newaxis, :, :] gives (N, N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # dist_sq is (N, N)
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero (though points shouldn't be identical)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    
    # Energy
    # Sum over i < j
    # Upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    energies = -np.log(dist[i_idx, j_idx])
    energy = np.sum(energies)
    
    # Gradient
    # grad[i] = - sum_{j!=i} (x_i - x_j) / dist_sq[i, j]
    # term[i, j, k] = (X[i, k] - X[j, k]) / dist_sq[i, j]
    # We want to sum over j for each i.
    # Note: diff[i, j, :] is x_i - x_j.
    # We need to divide each vector by dist_sq[i, j].
    # dist_sq is (N, N). We need to reshape it to (N, N, 1) to divide the vectors.
    inv_dist_sq = 1.0 / dist_sq
    inv_dist_sq[range(N), range(N)] = 0.0  # Zero out diagonal
    
    # grad_vec[i, k] = - sum_j (X[i, k] - X[j, k]) * inv_dist_sq[i, j]
    # This is equivalent to:
    # grad = - (diff * inv_dist_sq[:, :, np.newaxis]).sum(axis=1)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X_init
    best_energy = compute_energy_and_grad(X_init)[0]
    
    # Define constraints for trust-constr
    # Each point must satisfy x^2 + y^2 + z^2 = 1
    # scipy.optimize.minimize with method='trust-constr' takes a list of constraints
    # Constraint type: 'eq'
    
    from scipy.optimize import minimize, NonlinearConstraint
    
    # We can define a single constraint function that returns an array of N values
    def constraint_func(X_flat):
        X = X_flat.reshape(N, 3)
        return np.sum(X * X, axis=1) - 1.0
        
    def constraint_jac(X_flat):
        X = X_flat.reshape(N, 3)
        # Jacobian of c_i = x_i^2 + y_i^2 + z_i^2 - 1 w.r.t all variables
        # It's a block diagonal matrix with 2*X_i on the diagonal blocks
        # Shape: (N, 3*N)
        J = np.zeros((N, 3 * N))
        for i in range(N):
            J[i, 3*i:3*i+3] = 2 * X[i]
        return J

    constraint = NonlinearConstraint(constraint_func, np.zeros(N), np.zeros(N), jac=constraint_jac)
    
    # Objective function wrapper
    def obj(X_flat):
        X = X_flat.reshape(N, 3)
        e, _ = compute_energy_and_grad(X)
        return e
        
    def obj_grad(X_flat):
        X = X_flat.reshape(N, 3)
        _, g = compute_energy_and_grad(X)
        return g.flatten()

    # Run optimization
    # We will try a few restarts if time permits
    attempts = 1
    # Check time for first attempt
    if time.time() - start_time < time_limit * 0.5:
        attempts = 3 # Try a couple of random perturbations
    
    for k in range(attempts):
        if time.time() - start_time > time_limit:
            break
            
        if k == 0:
            X_start = X_init
        else:
            # Perturb the best solution found so far or the initial one
            # Adding small noise
            noise = np.random.normal(0, 0.1, size=(N, 3))
            X_start = best_X + noise
            # Project back to sphere
            norms = np.linalg.norm(X_start, axis=1, keepdims=True)
            X_start = X_start / norms
            
        X_flat = X_start.flatten()
        
        try:
            res = minimize(
                obj, 
                X_flat, 
                jac=obj_grad, 
                method='trust-constr', 
                constraints=[constraint],
                options={'gtol': 1e-8, 'xtol': 1e-8, 'maxiter': 500}
            )
            
            if res.success:
                X_res = res.x.reshape(N, 3)
                # Normalize just in case
                X_res = X_res / np.linalg.norm(X_res, axis=1, keepdims=True)
                e_res, _ = compute_energy_and_grad(X_res)
                
                if e_res < best_energy:
                    best_energy = e_res
                    best_X = X_res
                    
        except Exception as e:
            pass
            
    return best_X
