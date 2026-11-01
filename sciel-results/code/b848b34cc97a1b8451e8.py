import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

def construct():
    N = 50
    # 1. Initialize points on a unit sphere using Fibonacci lattice
    # This provides a good initial distribution
    i = np.arange(N)
    phi = (1 + 5**0.5) / 2
    theta = np.arccos(1 - 2 * (i + 0.5) / N)
    psi = 2 * np.pi * i / phi
    
    # Convert spherical to Cartesian (unit vectors)
    u = np.column_stack([
        np.sin(theta) * np.cos(psi),
        np.sin(theta) * np.sin(psi),
        np.cos(theta)
    ])
    
    # We will optimize the angles (theta, psi)
    # Flatten to 1D array for optimizer
    x0 = np.concatenate([theta, psi])
    
    # Objective function to minimize (negative radius)
    def objective(x):
        theta = x[:N]
        psi = x[N:]
        
        # Reconstruct unit vectors
        u = np.column_stack([
            np.sin(theta) * np.cos(psi),
            np.sin(theta) * np.sin(psi),
            np.cos(theta)
        ])
        
        # Compute M_u: max absolute coordinate
        # M_u = max(|u_ik|)
        M_u = np.max(np.abs(u))
        
        # Compute m_u: min pairwise distance on unit sphere
        # diff[i, j, k] = u_i - u_j
        diff = u[:, np.newaxis, :] - u[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Exclude diagonal
        np.fill_diagonal(dists, np.inf)
        m_u = np.min(dists)
        
        # Calculate optimal rho and radius
        # r = min(0.5 - rho * M_u, rho * m_u / 2)
        # Optimal rho balances them: 0.5 - rho * M_u = rho * m_u / 2
        # rho = 0.5 / (M_u + m_u / 2)
        # r = 0.5 * M_u / (M_u + m_u / 2)
        
        denom = M_u + 0.5 * m_u
        if denom < 1e-9:
            return 1.0 # Large positive value (bad)
            
        r = 0.5 * M_u / denom
        return -r

    # Use Powell method as it doesn't require gradients and handles non-smooth functions reasonably
    # Nelder-Mead is another option, but Powell is often more robust for this dimension
    # We need to be careful with time limit. 100 variables.
    
    result = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-6, 'disp': False})
    
    # Get optimal angles
    x_opt = result.x
    theta_opt = x_opt[:N]
    psi_opt = x_opt[N:]
    
    # Reconstruct unit vectors
    u_opt = np.column_stack([
        np.sin(theta_opt) * np.cos(psi_opt),
        np.sin(theta_opt) * np.sin(psi_opt),
        np.cos(theta_opt)
    ])
    
    # Compute M_u and m_u for optimal configuration
    M_u = np.max(np.abs(u_opt))
    diff = u_opt[:, np.newaxis, :] - u_opt[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    m_u = np.min(dists)
    
    # Calculate optimal rho
    rho = 0.5 / (M_u + 0.5 * m_u)
    
    # Final centers
    centers = rho * u_opt
    
    # Ensure centers are within the cube (they should be, but clip just in case due to numerical errors)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
