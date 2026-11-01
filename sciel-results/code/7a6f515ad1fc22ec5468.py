"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Use log-sum-exp style soft-min for smoother optimization
    # Objective: minimize -soft_min(pairwise distances) - soft_min(boundary distances)
    # We maximize the minimum, so we minimize the negative of a smooth approximation.
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft-min of distances using log-sum-exp trick
        # soft_min(d) ≈ -1/β * log(sum(exp(-β * d)))
        # We want to maximize min(d), so minimize -soft_min(d)
        # Let's use a moderate β to balance smoothness and accuracy
        beta = 50.0
        
        # For numerical stability, work with -dists_sq
        # soft_min(d) = -1/β * log(sum(exp(-β*d)))
        # We can rewrite: let m = min(dists), then sum(exp(-β*d)) = exp(-β*m) * sum(exp(-β*(d-m)))
        # log(sum(exp(-β*d))) = -β*m + log(sum(exp(-β*(d-m))))
        # So soft_min(d) = m - 1/β * log(sum(exp(-β*(d-m))))
        
        # But we need gradients. Let's use the direct approach with safe exponentials.
        # To avoid overflow, subtract the max (which is -min of -d = min of d... wait)
        # Actually, for soft-min, we subtract the min to keep exponents ≤ 0.
        
        # Pairwise part
        # We want f_pair = -soft_min(dists) = 1/β * log(sum(exp(-β * dists)))
        # Gradient: d f_pair / d x_i = (1/β) * (1/S) * sum_j (-β * exp(-β*d_ij) * d(d_ij)/d x_i)
        # where S = sum_j exp(-β*d_ij)
        # d(d_ij)/d x_i = (x_i - x_j) / d_ij
        
        # Let's compute this carefully
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # Safe version
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        # Compute weights w_ij = exp(-β * d_ij)
        # For stability, subtract min per row (which is the closest point)
        # Actually, we want sum over j for each i. Let's do it row by row conceptually.
        
        # exp(-β * dists_safe) can underflow for large distances, which is fine.
        # But for the min distance, exp(-β * d_min) could be very small if β is large.
        # Let's normalize by subtracting the min per row.
        
        # For each i, let m_i = min_j d_ij
        # Then sum_j exp(-β*d_ij) = exp(-β*m_i) * sum_j exp(-β*(d_ij - m_i))
        # log(sum_j exp(-β*d_ij)) = -β*m_i + log(sum_j exp(-β*(d_ij - m_i)))
        
        # f_pair = (1/β) * sum_i log(sum_j exp(-β*d_ij))
        #        = (1/β) * sum_i [-β*m_i + log(sum_j exp(-β*(d_ij - m_i)))]
        #        = -sum_i m_i + (1/β) * sum_i log(sum_j exp(-β*(d_ij - m_i)))
        
        # This is still complex. Let's use a simpler approach:
        # Use a power-law approximation but with better gradient handling.
        # Or use the fact that for large P, sum(d^-P) ≈ N * d_min^-P
        
        # Let's stick with the power law but fix the gradient issues and use multiple restarts.
        P = 30.0
        
        # Pairwise objective: sum_i sum_j d_ij^(-P)
        # This is dominated by the minimum distance.
        
        inv_dists_P = dists_safe ** (-P)
        np.fill_diagonal(inv_dists_P, 0.0)
        
        obj_pair = np.sum(inv_dists_P)
        
        # Gradient of obj_pair w.r.t. x_i:
        # d/dx_i sum_j d_ij^(-P) = sum_j (-P) * d_ij^(-P-1) * d(d_ij)/d x_i
        # d(d_ij)/d x_i = (x_i - x_j) / d_ij
        # So d/dx_i = sum_j (-P) * d_ij^(-P-2) * (x_i - x_j)
        
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        
        row_sums = np.sum(W, axis=1)
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - (W @ pts))
        
        # Boundary part
        b_dists = 0.5 - np.abs(pts)
        b_dists_safe = np.where(b_dists < 1e-10, 1e-10, b_dists)
        
        # For each point, the boundary distance is the min over the 6 faces
        # We can approximate by summing over all 6 faces with a soft-min
        # Or just use the min and handle the gradient for the active face
        
        # Let's use the min approach with subgradient
        min_b = np.min(b_dists, axis=1)
        min_b_safe = np.where(min_b < 1e-10, 1e-10, min_b)
        
        obj_bound = np.sum(min_b_safe ** (-P))
        
        # Gradient
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            # Find which axis and sign is the minimum
            # b_dists[i] has 3 elements: 0.5-|x|, 0.5-|y|, 0.5-|z|
            # The minimum could be from +x, -x, +y, -y, +z, or -z
            # b_dists[i,0] = 0.5 - |x_i|
            # If x_i > 0, this is 0.5 - x_i, gradient -1
            # If x_i < 0, this is 0.5 + x_i, gradient +1
            # If x_i = 0, subgradient [-1, 1]
            
            # Find the minimum coordinate-wise distance
            min_coord = np.argmin(b_dists[i])
            # Determine the sign
            coord_val = pts[i, min_coord]
            if coord_val > 0:
                sign = 1.0
            elif coord_val < 0:
                sign = -1.0
            else:
                sign = 0.0  # Subgradient, pick 0 for now
            
            # d(min_b)/dx_i = -sign * e_min_coord
            # d(min_b^-P)/dx_i = -P * min_b^(-P-1) * d(min_b)/dx_i
            #                 = -P * min_b^(-P-1) * (-sign * e_min_coord)
            #                 = P * sign * min_b^(-P-1) * e_min_coord
            
            if sign != 0:
                grad_bound[i, min_coord] = P * sign * min_b_safe[i] ** (-P - 1.0)
        
        total_obj = obj_pair + obj_bound
        total_grad = grad_pair + grad_bound
        
        return total_obj, total_grad.flatten()

    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r

    def init_grid():
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].flatten()

    def init_random():
        return np.random.uniform(-0.4, 0.4, size=(N, 3)).flatten()

    def init_fcc():
        # Face-centered cubic lattice, scaled to fit
        # FCC has points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in a unit cell
        # Let's create a 4x4x4 grid of FCC points
        pts = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    pts.append([i*0.25, j*0.25, k*0.25])
                    pts.append([i*0.25+0.125, j*0.25+0.125, k*0.25])
                    pts.append([i*0.25+0.125, j*0.25, k*0.25+0.125])
                    pts.append([i*0.25, j*0.25+0.125, k*0.25+0.125])
        pts = np.array(pts)
        # Scale to fit in [-0.5, 0.5]
        pts = pts * (1.0 / 1.0) - 0.5  # This doesn't quite work, let me fix
        # Actually, let's just use a simple body-centered cubic
        pts = []
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    pts.append([i*0.2-0.5, j*0.2-0.5, k*0.2-0.5])
                    if i < 4 and j < 4 and k < 4:
                        pts.append([i*0.2-0.5+0.1, j*0.2-0.5+0.1, k*0.2-0.5+0.1])
        pts = np.array(pts[:N])
        return pts.flatten()

    starts = [
        init_grid(),
        init_grid() + np.random.normal(0, 0.015, size=(N*3,)),
        init_grid() + np.random.normal(0, 0.03, size=(N*3,)),
        init_random(),
        init_random() + np.random.normal(0, 0.05, size=(N*3,)),
        init_fcc()
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use multiple optimization strategies
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            # First try L-BFGS-B
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                           jac=True,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < 1e10:
                pts = res.x.reshape(N, 3)
                pts = np.clip(pts, -0.5, 0.5)
                score = calculate_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            continue

    # If we have time, do local refinement on the best solution
    if best_points is not None and time.time() < deadline - 20:
        try:
            x0 = best_points.flatten()
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                           jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 1e10:
                pts = res.x.reshape(N, 3)
                pts = np.clip(pts, -0.5, 0.5)
                score = calculate_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = init_grid().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
