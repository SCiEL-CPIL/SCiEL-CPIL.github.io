"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Voronoi
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in the unit square maximizing sum of radii.
    
    Strategy:
    1. Initialize with a good grid-like starting point.
    2. Use L-BFGS-B to optimize positions and radii jointly.
    3. The objective is -sum(radii) + penalty for overlaps and boundary violations.
    4. Use a smooth penalty (soft-min) for the minimum distance constraint.
    5. Final polish with SLSQP for tight constraints.
    """
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Initial grid layout
    side = int(np.ceil(np.sqrt(N)))
    # Use a hexagonal-like offset for better packing
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add slight offset for odd rows to improve packing
            if i % 2 == 1:
                x += 0.25 / side
            positions.append([x, y])
    
    positions = np.array(positions[:N])
    # Initial radii: slightly less than half the cell size
    initial_radius = 0.5 / side - 0.01
    radii = np.full(N, initial_radius)
    
    x0 = np.concatenate([positions.flatten(), radii])
    
    def objective_and_grad(x):
        """
        Objective: maximize sum(radii) - penalty for violations
        Returns (value, gradient)
        """
        pos = x[:2*N].reshape(N, 2)
        rad = x[2*N:]
        
        # Sum of radii
        sum_r = np.sum(rad)
        
        # Penalty for overlaps: for each pair, distance - (r_i + r_j) should be >= 0
        # Use soft-min penalty: -log(sum(exp(-beta * min_dist))) approximates -min_dist
        # Actually, let's use a smooth approximation to the minimum constraint violation
        # violation_ij = max(0, (r_i + r_j) - dist_ij)
        # We want to minimize sum of violations
        
        # Compute pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Required separation
        req_sep = rad[:, np.newaxis] + rad[np.newaxis, :]
        
        # Violation: positive if circles overlap
        viol = req_sep - dist
        viol = np.maximum(viol, 0)
        
        # Smooth penalty using log-sum-exp for the max violation
        # beta controls smoothness
        beta = 50.0
        # For numerical stability, subtract max
        viol_flat = viol.flatten()
        viol_flat = viol_flat[viol_flat > -100]  # ignore very negative
        if len(viol_flat) > 0:
            max_viol = np.max(viol_flat)
            if max_viol > 1e-10:
                # log-sum-exp approximation of max
                lse = max_viol + np.log(np.sum(np.exp((viol_flat - max_viol) * beta)) / beta)
                overlap_penalty = lse
            else:
                overlap_penalty = 0.0
        else:
            overlap_penalty = 0.0
        
        # Boundary violations: circle must be inside [0,1]^2
        # x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
        boundary_viol = np.concatenate([
            np.maximum(0, rad - pos[:, 0]),       # left
            np.maximum(0, rad - (1 - pos[:, 0])),  # right
            np.maximum(0, rad - pos[:, 1]),       # bottom
            np.maximum(0, rad - (1 - pos[:, 1]))   # top
        ])
        
        if len(boundary_viol) > 0:
            max_bv = np.max(boundary_viol)
            if max_bv > 1e-10:
                boundary_penalty = max_bv + np.log(np.sum(np.exp((boundary_viol - max_bv) * beta)) / beta)
            else:
                boundary_penalty = 0.0
        else:
            boundary_penalty = 0.0
        
        # Total objective: minimize -sum_r + penalty
        value = -sum_r + 10.0 * (overlap_penalty + boundary_penalty)
        
        # Gradient computation
        grad = np.zeros_like(x)
        
        # Gradient of -sum_r
        grad[2*N:] = -1.0
        
        # Gradient of overlap penalty
        if overlap_penalty > 1e-10:
            # d/dx of log-sum-exp
            exp_viol = np.exp((viol - max_viol) * beta)
            weights = exp_viol / np.sum(exp_viol)
            # weights shape (N, N)
            
            # d(viol_ij)/d(pos_i) = (pos_i - pos_j) / dist_ij
            # d(viol_ij)/d(rad_i) = 1
            
            # For each pair (i,j), contribution to gradient
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    w = weights[i, j]
                    if w < 1e-15:
                        continue
                    d_ij = diff[i, j]
                    dist_ij = dist[i, j]
                    if dist_ij < 1e-10:
                        continue
                    # d(viol_ij)/d(pos_i) = d_ij / dist_ij
                    grad[2*i] += 10.0 * w * d_ij[0] / dist_ij
                    grad[2*i + 1] += 10.0 * w * d_ij[1] / dist_ij
                    # d(viol_ij)/d(rad_i) = 1
                    grad[2*N + i] += 10.0 * w
                    # d(viol_ij)/d(rad_j) = 1 (from req_sep)
                    grad[2*N + j] += 10.0 * w
        
        # Gradient of boundary penalty
        if boundary_penalty > 1e-10:
            exp_bv = np.exp((boundary_viol - max_bv) * beta)
            bweights = exp_bv / np.sum(exp_bv)
            
            # Left: rad - pos_x
            for i in range(N):
                w = bweights[i]
                if w > 1e-15:
                    grad[2*N + i] += 10.0 * w
                    grad[2*i] -= 10.0 * w
            # Right: rad - (1 - pos_x)
            for i in range(N):
                w = bweights[N + i]
                if w > 1e-15:
                    grad[2*N + i] += 10.0 * w
                    grad[2*i] += 10.0 * w
            # Bottom: rad - pos_y
            for i in range(N):
                w = bweights[2*N + i]
                if w > 1e-15:
                    grad[2*N + i] += 10.0 * w
                    grad[2*i + 1] -= 10.0 * w
            # Top: rad - (1 - pos_y)
            for i in range(N):
                w = bweights[3*N + i]
                if w > 1e-15:
                    grad[2*N + i] += 10.0 * w
                    grad[2*i + 1] += 10.0 * w
        
        return value, grad
    
    # Run L-BFGS-B
    best_x = x0.copy()
    best_val = objective_and_grad(x0)[0]
    
    try:
        res = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        pos = best_x[:2*N].reshape(N, 2)
        rad = best_x[2*N:]
        result = np.column_stack([pos, rad])
        return result
    
    # Polish with SLSQP: fix positions, optimize radii
    pos = best_x[:2*N].reshape(N, 2)
    
    def radius_objective(rad):
        """Minimize -sum(rad) subject to constraints"""
        return -np.sum(rad)
    
    # Constraints: for each pair, r_i + r_j <= dist_ij
    # And boundary: r_i <= min(pos_i, 1-pos_i)
    
    constraints = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            constraints.append({
                'type': 'ineq',
                'fun': lambda r, i=i, j=j, d=d: d - r[i] - r[j],
                'jac': lambda r, i=i, j=j: np.concatenate([
                    np.zeros(i), [-1], np.zeros(j-i-1), [-1], np.zeros(N-j-1)
                ])
            })
    
    # Boundary constraints
    for i in range(N):
        max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        constraints.append({
            'type': 'ineq',
            'fun': lambda r, i=i, max_r=max_r: max_r - r[i],
            'jac': lambda r, i=i: np.concatenate([
                np.zeros(i), [-1], np.zeros(N-i-1)
            ])
        })
    
    bounds = [(0, 1)] * N
    
    try:
        res2 = minimize(
            radius_objective,
            best_x[2*N:],
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if res2.success:
            best_x[2*N:] = res2.x
    except Exception:
        pass
    
    # Final result
    pos = best_x[:2*N].reshape(N, 2)
    rad = best_x[2*N:]
    
    # Ensure feasibility: clip radii to be non-negative and within bounds
    rad = np.maximum(rad, 0)
    for i in range(N):
        max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        rad[i] = min(rad[i], max_r)
    
    result = np.column_stack([pos, rad])
    return result
# EVOLVE-BLOCK-END
