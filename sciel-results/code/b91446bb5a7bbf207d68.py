import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    # Convert to spherical coordinates
    # z = cos(theta) => theta = arccos(z)
    theta = np.arccos(z)
    # phi is already the azimuthal angle
    # Ensure phi is in [0, 2*pi)
    phi = phi % (2 * np.pi)
    
    # Flatten parameters
    x0 = np.concatenate([theta, phi])
    
    # Bounds
    bounds_theta = [(0.0, np.pi) for _ in range(N)]
    bounds_phi = [(0.0, 2 * np.pi) for _ in range(N)]
    bounds = bounds_theta + bounds_phi
    
    def objective_and_grad(params):
        # Check time
        if time.time() > deadline:
            return np.inf, np.zeros_like(params)
            
        theta = params[:N]
        phi = params[N:]
        
        # Compute Cartesian coordinates
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta
        
        X = np.column_stack([x, y, z])
        
        # Compute pairwise differences
        # D[i, j] = X[i] - X[j]
        D = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        
        # Compute squared distances
        # dist2[i, j] = ||X[i] - X[j]||^2
        dist2 = np.sum(D * D, axis=2)
        
        # Avoid division by zero for i==j
        # dist2[i, i] = 0
        np.fill_diagonal(dist2, 1.0) # Temporary placeholder
        
        # Energy = sum_{i<j} log(1 / dist_ij) = - sum_{i<j} log(dist_ij)
        # dist_ij = sqrt(dist2_ij)
        # log(dist_ij) = 0.5 * log(dist2_ij)
        # So term is -0.5 * log(dist2_ij)
        
        # We sum over i<j. 
        # Matrix dist2 is symmetric.
        # We can sum over all i != j and divide by 2.
        
        # Mask for i != j
        mask = ~np.eye(N, dtype=bool)
        
        # Calculate log terms
        # dist2 is 0 on diagonal, we masked it.
        # However, if points are very close, dist2 is small.
        
        # To be safe with log, we can use the fact that log(1/d) is the energy term.
        # E = sum_{i<j} -log(sqrt(dist2_ij)) = -0.5 * sum_{i<j} log(dist2_ij)
        
        # Let's compute the full sum over i != j
        # log_dist2 = log(dist2)
        # We need to be careful with dist2 == 0. 
        # The problem states points closer than 1e-9 are invalid.
        # The optimizer should avoid this.
        
        # For gradient calculation stability, we might add a tiny epsilon
        # but for energy, we want the true value.
        
        # Let's compute gradient first, which requires 1/dist2
        # G_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
        
        # inv_dist2 = 1 / dist2
        # Set diagonal to 0
        inv_dist2 = 1.0 / dist2
        np.fill_diagonal(inv_dist2, 0.0)
        
        # G = D @ inv_dist2  (Matrix multiplication: sum over j)
        # D is (N, N, 3). inv_dist2 is (N, N).
        # We want for each i: sum_j D[i,j] * inv_dist2[i,j]
        # This is equivalent to (D * inv_dist2[:, :, np.newaxis]).sum(axis=1)
        
        G = np.sum(D * inv_dist2[:, :, np.newaxis], axis=1)
        
        # Calculate Energy
        # E = -0.5 * sum_{i != j} log(dist2_ij)
        # Note: sum_{i != j} is 2 * sum_{i < j}
        # So E = -0.5 * (2 * sum_{i < j} log(dist2_ij)) = - sum_{i < j} log(dist2_ij)
        # Wait.
        # Energy definition: sum_{i<j} log(1/d_ij) = sum_{i<j} -log(d_ij)
        # d_ij = sqrt(dist2_ij)
        # -log(d_ij) = -0.5 log(dist2_ij)
        # So E = sum_{i<j} -0.5 log(dist2_ij)
        
        # Using the full matrix sum:
        # Sum_{i != j} -0.5 log(dist2_ij) = -0.5 * 2 * Sum_{i < j} log(dist2_ij) = - Sum_{i < j} log(dist2_ij)
        # So if we compute S = Sum_{i != j} -0.5 log(dist2_ij), that is exactly 2 * E?
        # No.
        # Let T_{ij} = -0.5 log(dist2_ij).
        # E = Sum_{i < j} T_{ij}.
        # Sum_{i != j} T_{ij} = Sum_{i < j} T_{ij} + Sum_{j < i} T_{ij} = 2 * E.
        # So E = 0.5 * Sum_{i != j} T_{ij}.
        
        # Let's compute T for all i != j
        # log_dist2 = log(dist2)
        # We must handle dist2=0. 
        # If dist2 is very small, log is large negative.
        
        # To avoid log(0), we can clip dist2 to a small positive number for the log calculation,
        # but this distorts the energy if points are actually close.
        # However, the optimizer should keep them apart.
        
        # Let's just compute it.
        # dist2 has 0 on diagonal.
        # We masked diagonal in inv_dist2.
        # For log, we need to ignore diagonal.
        
        log_dist2 = np.log(dist2)
        # Set diagonal to 0 to ignore it in sum
        np.fill_diagonal(log_dist2, 0.0)
        
        # Sum over all i != j
        # T_{ij} = -0.5 * log_dist2[i,j]
        # Sum_T = sum(T_{ij})
        # E = 0.5 * Sum_T
        
        Sum_T = np.sum(-0.5 * log_dist2)
        energy = 0.5 * Sum_T
        
        # Calculate Gradient with respect to theta, phi
        # dE/dtheta_i = G_i . dx_i/dtheta_i
        # dE/dphi_i = G_i . dx_i/dphi_i
        
        # dx/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        # dx/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
        
        dtheta = np.column_stack([
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta
        ])
        
        dphi = np.column_stack([
            -sin_theta * sin_phi,
            sin_theta * cos_phi,
            np.zeros(N)
        ])
        
        grad_theta = np.sum(G * dtheta, axis=1)
        grad_phi = np.sum(G * dphi, axis=1)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        return energy, grad

    # Optimize
    # Use L-BFGS-B
    # maxiter should be sufficient
    # maxfun should be sufficient
    
    # We want to minimize energy.
    # The function returns (energy, grad).
    # scipy.optimize.minimize expects fun to return energy, and jac to return gradient.
    # Or fun returns (energy, grad) if we pass jac='2-point' or similar?
    # Better to define separate functions or use a wrapper.
    
    def fun(params):
        e, _ = objective_and_grad(params)
        return e
        
    def jac(params):
        _, g = objective_and_grad(params)
        return g
        
    # Run optimization
    # We can run multiple restarts if time permits
    best_x = x0
    best_energy = np.inf
    
    # Initial energy
    e0, _ = objective_and_grad(x0)
    best_energy = e0
    
    # Optimization
    # L-BFGS-B is good for bounded problems
    res = scipy.optimize.minimize(
        fun, 
        x0, 
        method='L-BFGS-B', 
        jac=jac, 
        bounds=bounds, 
        options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() < deadline:
        if res.success and res.fun < best_energy:
            best_energy = res.fun
            best_x = res.x
            
    # Convert best_x back to Cartesian
    theta_best = best_x[:N]
    phi_best = best_x[N:]
    
    sin_theta = np.sin(theta_best)
    cos_theta = np.cos(theta_best)
    cos_phi = np.cos(phi_best)
    sin_phi = np.sin(phi_best)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    points = np.column_stack([x, y, z])
    
    # Normalize just in case of numerical drift
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points
