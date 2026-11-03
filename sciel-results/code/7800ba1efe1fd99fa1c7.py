import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # Area of unit circle = pi. Area per circle = pi/300.
    # Hex lattice area per circle = (sqrt(3)/2) * s^2 where s is spacing.
    # (sqrt(3)/2) s^2 = pi/300 => s^2 = 2*pi / (300*sqrt(3))
    s_est = np.sqrt(2 * np.pi / (300 * np.sqrt(3)))
    
    # Generate a grid of hex lattice points
    # We need enough points to cover the unit disk
    # Max radius is 1.
    # Number of rows/cols needed ~ 1/s_est
    k = int(1.0 / s_est) + 5
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s_est
            y = j * s_est * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    
    points = np.array(rows)
    # Filter points inside unit circle (with some margin)
    # We want points that can fit a circle of radius r = s_est/2
    # So |p| <= 1 - r
    # But we don't know final r. Just take points inside unit circle.
    dists = np.hypot(points[:, 0], points[:, 1])
    # Sort by distance to origin to pick central ones
    points = points[dists <= 1.0]
    order = np.argsort(dists[dists <= 1.0])
    # We need N points. If we have more, take the N closest to origin.
    if len(points) > N:
        points = points[order[:N]]
    else:
        # If not enough, decrease spacing and retry
        s_est *= 0.95
        while len(points) < N:
             k = int(1.0 / s_est) + 5
             rows = []
             for j in range(-k, k + 1):
                 for i in range(-k, k + 1):
                     x = (i + 0.5 * (j % 2)) * s_est
                     y = j * s_est * np.sqrt(3.0) / 2.0
                     rows.append((x, y))
             points = np.array(rows)
             dists = np.hypot(points[:, 0], points[:, 1])
             points = points[dists <= 1.0]
             if len(points) > N:
                 order = np.argsort(dists[dists <= 1.0])
                 points = points[order[:N]]
             s_est *= 0.95
             
    # Initial radius
    r_init = s_est / 2.0
    
    # State vector: [x0, y0, ..., xN-1, yN-1, r]
    x0 = np.zeros(2 * N + 1)
    x0[:2 * N:2] = points[:, 0]
    x0[1:2 * N:2] = points[:, 1]
    x0[-1] = r_init

    def objective_and_grad(z):
        # Extract positions and r
        pos = z[:2 * N].reshape(N, 2)
        r = z[-1]
        
        # Compute pairwise distances
        # diff[i, j] = pos[i] - pos[j]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Mask diagonal to avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Overlap Penalty: sum( max(0, 2r - dist)^2 )
        # Only consider i < j to avoid double counting
        iu = np.triu_indices(N, k=1)
        d_ij = dists[iu]
        v_ij = 2 * r - d_ij
        # Penalty is only for violations
        mask_ij = v_ij > 0
        p_ij = np.zeros_like(v_ij)
        p_ij[mask_ij] = v_ij[mask_ij] ** 2
        
        # Gradient of overlap penalty
        # d/dr sum(v^2) = sum(2*v * 2) = 4 * sum(v) for masked
        grad_r_overlap = 4 * np.sum(v_ij[mask_ij])
        
        # d/dpos_i sum(v^2)
        # v = 2r - d
        # dv/dpos_i = - (pos_i - pos_j) / d
        # d(v^2)/dpos_i = 2*v * (- (pos_i - pos_j) / d)
        # We need to accumulate this for each i.
        # This is tricky to vectorize directly into a (N, 2) array.
        # Let's compute the gradient for all pairs and scatter.
        # grad_pos = np.zeros((N, 2))
        # For each pair (i, j) in mask:
        #   grad_pos[i] += -2 * v_ij * (pos[i] - pos[j]) / d_ij
        #   grad_pos[j] += -2 * v_ij * (pos[j] - pos[i]) / d_ij
        
        # Vectorized approach:
        # Create a matrix of gradients for all pairs
        # diff[i, j] is pos[i] - pos[j]
        # grad_vec = -2 * v_ij * (diff[i, j] / d_ij)
        # We only care about upper triangle.
        
        # Let's compute the full gradient matrix G where G[i, j] is the contribution to pos[i] from pair (i, j)?
        # No, let's just loop over the active pairs. N=300, active pairs might be ~1000-5000.
        # A Python loop over 5000 items is fast enough (~1ms).
        
        grad_pos = np.zeros((N, 2))
        active_pairs = np.where(mask_ij)[0]
        for idx in active_pairs:
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            v = v_ij[idx]
            # d(v^2)/dpos_i = 2*v * (- (pos_i - pos_j) / d)
            # d(v^2)/dpos_j = 2*v * (- (pos_j - pos_i) / d) = 2*v * ((pos_i - pos_j) / d)
            vec = (pos[i] - pos[j]) / d
            grad_pos[i] -= 2 * v * vec
            grad_pos[j] += 2 * v * vec
            
        # Boundary Penalty: sum( max(0, |c_i| - (1-r))^2 )
        radii = np.linalg.norm(pos, axis=1)
        v_bound = radii - (1 - r)
        mask_bound = v_bound > 0
        p_bound = np.zeros_like(v_bound)
        p_bound[mask_bound] = v_bound[mask_bound] ** 2
        
        # Gradient of boundary penalty
        # d/dr sum(v^2) = sum(2*v * 1) = 2 * sum(v)
        grad_r_bound = 2 * np.sum(v_bound[mask_bound])
        
        # d/dpos_i
        # v = |c_i| - (1-r)
        # dv/dpos_i = c_i / |c_i|
        # d(v^2)/dpos_i = 2*v * (c_i / |c_i|)
        grad_pos_bound = np.zeros((N, 2))
        active_bound = np.where(mask_bound)[0]
        for i in active_bound:
            d = radii[i]
            v = v_bound[i]
            if d > 1e-8:
                grad_pos_bound[i] = 2 * v * (pos[i] / d)
                
        # Total Penalty
        penalty = np.sum(p_ij) + np.sum(p_bound)
        
        # Objective: -r + lambda * penalty
        # We need to pass lambda. We can use a closure or global.
        # Let's use a global variable for lambda to allow tuning.
        global LAMBDA
        obj = -r + LAMBDA * penalty
        
        # Gradient
        grad = np.zeros_like(z)
        grad[:2 * N] = grad_pos.flatten() + grad_pos_bound.flatten()
        grad[-1] = grad_r_overlap + grad_r_bound
        
        # Note: The gradient of -r is -1.
        # So grad_r = -1 + LAMBDA * (grad_r_overlap + grad_r_bound)
        grad[-1] = -1.0 + LAMBDA * (grad_r_overlap + grad_r_bound)
        
        return obj, grad

    # We need to tune LAMBDA.
    # Start with a high value to enforce constraints.
    LAMBDA = 1000.0
    
    best_x = x0
    best_val = objective_and_grad(x0)[0]
    
    # Optimization loop
    # We might need multiple runs or increasing lambda.
    # Let's try one run first.
    
    options = {
        'maxiter': 200,
        'ftol': 1e-10,
        'gtol': 1e-8,
        'disp': False
    }
    
    def callback(xk):
        nonlocal best_x, best_val
        # Check time
        if time.time() - start_time > timeout:
            raise RuntimeError("Timeout")
        # Check if better
        val = objective_and_grad(xk)[0]
        # We want to minimize objective.
        # But we also need to ensure feasibility for the "best" tracking?
        # The objective includes penalty, so low objective implies feasibility.
        if val < best_val:
            best_val = val
            best_x = xk.copy()
            
    try:
        res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', options=options, callback=callback)
        if res.success or res.fun < best_val:
             # Check if res.fun is better
             if res.fun < best_val:
                 best_val = res.fun
                 best_x = res.x
    except RuntimeError as e:
        if "Timeout" in str(e):
            pass
        else:
            raise

    # Extract result
    pos = best_x[:2 * N].reshape(N, 2)
    r = best_x[-1]
    
    # Final check and cleanup
    # Ensure r is positive
    if r < 0:
        r = 0
        
    # Return centers
    return pos

# Global variable for LAMBDA
LAMBDA = 1000.0
