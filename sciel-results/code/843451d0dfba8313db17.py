import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s margin for 150s limit

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Create a dense spherical mesh to approximate the sphere
    n_grid = 150
    theta = np.linspace(0, np.pi, n_grid)
    phi_grid = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(theta, phi_grid)
    
    mesh_x = np.sin(THETA) * np.cos(PHI)
    mesh_y = np.sin(THETA) * np.sin(PHI)
    mesh_z = np.cos(THETA)
    mesh_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel(), mesh_z.ravel()])
    mesh_points = mesh_points / np.linalg.norm(mesh_points, axis=1, keepdims=True)
    
    def compute_covering_radius(pts):
        dots = mesh_points @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)
    
    current_radius = compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Repulsion-based optimization with simulated annealing
    # Energy = -sum of log(pairwise distances) + repulsion for close points
    # We minimize a combined objective that correlates with covering radius
    
    def repulsion_energy(pts):
        """Compute repulsion energy: sum of 1/d^2 for close pairs + -log(d) for all pairs"""
        # Pairwise dot products
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)  # Avoid self-interaction
        dots = np.clip(dots, -1.0, 1.0)
        # Angular distances
        angles = np.arccos(dots)
        # For pairs that are too close, add strong repulsion
        # Energy = sum over i<j of max(0, (theta_min - angle_ij)^2) - alpha * sum log(angle_ij)
        # This encourages spreading while penalizing clustering
        lower_tri = np.tril_indices_from(angles, k=-1)
        pair_angles = angles[lower_tri]
        # Strong penalty for very close points
        close_penalty = np.sum(np.maximum(0, 0.05 - pair_angles) ** 2)
        # Mild repulsion to spread points
        spread_energy = -0.1 * np.sum(np.log(np.maximum(pair_angles, 1e-10)))
        return close_penalty + spread_energy
    
    def repulsion_gradient(pts):
        """Compute gradient of repulsion energy w.r.t. points on sphere"""
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Gradient of close_penalty: d/dp_i [max(0, 0.05 - angle_ij)^2]
        # = -2*max(0, 0.05-angle_ij) * d(angle_ij)/d(p_i)
        # d(angle_ij)/d(p_i) = -(p_j - (p_i·p_j)p_i) / sin(angle_ij) for i≠j
        
        grad = np.zeros_like(pts)
        lower_tri = np.tril_indices_from(angles, k=-1)
        i_idx, j_idx = lower_tri
        pair_angles = angles[lower_tri]
        
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            ang = pair_angles[k]
            if ang < 1e-10:
                continue
            
            # Close penalty gradient
            if ang < 0.05:
                coeff = -2.0 * (0.05 - ang)
            else:
                coeff = 0.0
            
            # Spread energy gradient: -0.1 * d(-log(angle))/d(p_i) = -0.1 * (1/angle) * d(angle)/d(p_i)
            spread_coeff = -0.1 / ang
            
            total_coeff = coeff + spread_coeff
            
            if abs(total_coeff) < 1e-15:
                continue
            
            # d(angle_ij)/d(p_i) = -(p_j - (p_i·p_j)p_i) / sin(angle_ij)
            sin_ang = np.sin(ang)
            if sin_ang < 1e-10:
                continue
            
            pi = pts[i]
            pj = pts[j]
            dot_ij = pi @ pj
            
            # Tangent component of pj at pi
            tangent = pj - dot_ij * pi
            tangent /= sin_ang
            
            # Gradient contribution to point i
            grad[i] += total_coeff * (-tangent)
            # Gradient contribution to point j (by symmetry, opposite sign)
            # d(angle_ij)/d(p_j) = -(p_i - (p_j·p_i)p_j) / sin(angle_ij)
            tangent_j = pi - dot_ij * pj
            tangent_j /= sin_ang
            grad[j] += total_coeff * (-tangent_j)
        
        # Project gradient onto tangent plane (remove radial component)
        for k in range(N):
            p = pts[k]
            grad[k] -= (grad[k] @ p) * p
        
        return grad
    
    # Simulated annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.02
    
    current_energy = repulsion_energy(points)
    
    while time.time() < deadline:
        # Compute gradient
        grad = repulsion_gradient(points)
        
        # Add random perturbation (annealing)
        if T > T_min:
            noise = np.random.randn(N, 3) * np.sqrt(2 * T)
            # Project noise onto tangent planes
            for k in range(N):
                p = points[k]
                noise[k] -= (noise[k] @ p) * p
        
        # Update: gradient descent + noise
        new_points = points - step_size * grad + noise
        
        # Normalize
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Compute new energy
        new_energy = repulsion_energy(new_points)
        
        # Metropolis acceptance
        delta_E = new_energy - current_energy
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            points = new_points
            current_energy = new_energy
            
            # Check covering radius
            current_radius = compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive step size
        if T < T_min:
            step_size *= 0.98
    
    return best_points
