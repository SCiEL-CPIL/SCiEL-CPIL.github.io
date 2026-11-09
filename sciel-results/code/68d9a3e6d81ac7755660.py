"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    best_score = 0.0
    best_points = None
    
    # Helper to calculate the score (minimum radius)
    def get_score(pts):
        # Pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
        # Boundary distance
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    # Initial configuration: Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_points = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98

    # SQP Solver for NCP:
    # Maximize r subject to:
    # 1. ||x_i - x_j||^2 >= (2r)^2  => ||x_i - x_j||^2 - 4r^2 >= 0
    # 2. ||x_i||^2 <= (1-r)^2       => ||x_i||^2 - (1-r)^2 <= 0
    
    # Variables: X (N x 2) and r (scalar). Total M = 2N + 1 = 101 variables.
    # We will use a penalty method to handle the constraints, transforming it into a smooth minimization problem.
    # Objective: Minimize -r + mu * (sum of penalty terms for violated constraints)
    
    def flatten(x, r):
        return np.concatenate([x.flatten(), [r]])
    
    def unflatten(v):
        x = v[:2*N].reshape(N, 2)
        r = v[2*N]
        return x, r

    def compute_penalty_and_grad(v, mu):
        x, r = unflatten(v)
        # Constraints:
        # C1_ij: ||x_i - x_j||^2 - 4r^2 >= 0  (Violation if < 0)
        # C2_i: ||x_i||^2 - (1-r)^2 <= 0      (Violation if > 0)
        
        # Pairwise distances squared
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        
        # C1 violations: max(0, 4r^2 - d_sq)
        c1_vals = 4*r*r - d_sq
        c1_viol = np.maximum(0, c1_vals)
        
        # C2 violations: max(0, ||x_i||^2 - (1-r)^2)
        r_sq = np.sum(x*x, axis=1)
        c2_vals = r_sq - (1-r)**2
        c2_viol = np.maximum(0, c2_vals)
        
        # Penalty term
        pen = np.sum(c1_viol**2) + np.sum(c2_viol**2)
        obj = -r + mu * pen
        
        # Gradients
        grad_x = np.zeros_like(x)
        grad_r = -1.0
        
        # Gradient of C1
        # d(C1)/dx_i = 2(x_i - x_j) for each j
        # d(C1)/dr = 8r
        if np.any(c1_viol > 0):
            # c1_viol is (N, N) matrix
            # grad_x_i += sum_j 2 * c1_viol_ij * (x_i - x_j)
            # Note: c1_viol is symmetric.
            # Let's compute efficiently.
            # grad_x = 2 * (c1_viol * diff).sum(axis=1) ? 
            # Wait, diff[i,j] = x_i - x_j.
            # d/dx_i (4r^2 - ||x_i - x_j||^2) = -2(x_i - x_j)
            # So d/dx_i (max(0, 4r^2 - ||x_i - x_j||^2)^2) 
            # = 2 * (4r^2 - d_ij^2) * (-2(x_i - x_j)) if violated
            # = -4 * c1_viol_ij * (x_i - x_j)
            
            # Sum over j
            # grad_x_i = sum_j -4 * c1_viol_ij * diff_ij
            # Since c1_viol is 0 where not violated, we can just sum.
            # diff is (N, N, 2)
            # c1_viol is (N, N)
            # We need to sum over j (axis 1)
            # grad_x = -4 * np.sum(c1_viol[:, :, np.newaxis] * diff, axis=1)
            
            # Also grad_r from C1:
            # d/dr (4r^2 - d^2) = 8r
            # d/dr (max(0, 4r^2 - d^2)^2) = 2 * (4r^2 - d^2) * 8r = 16r * c1_viol
            # Sum over pairs. Since c1_viol is symmetric and diagonal is inf (but 0 violation), 
            # sum over all i,j where i!=j.
            # Sum_{i,j} c1_viol_ij = 2 * Sum_{i<j} c1_viol_ij
            # So grad_r += mu * 16r * sum(c1_viol)
            
            grad_x += -4.0 * np.sum(c1_viol[:, :, np.newaxis] * diff, axis=1)
            grad_r += 16.0 * r * np.sum(c1_viol)

        # Gradient of C2
        # C2_i = ||x_i||^2 - (1-r)^2
        # d(C2_i)/dx_i = 2x_i
        # d(C2_i)/dr = -2(1-r)(-1) = 2(1-r)
        # Penalty: max(0, C2_i)^2
        # d/dx_i = 2 * C2_viol_i * 2x_i = 4 * c2_viol_i * x_i
        # d/dr = 2 * C2_viol_i * 2(1-r) = 4 * (1-r) * c2_viol_i
        
        if np.any(c2_viol > 0):
            grad_x += 4.0 * c2_viol[:, np.newaxis] * x
            grad_r += 4.0 * (1.0 - r) * np.sum(c2_viol)

        # Total gradient
        grad_x *= 2 * mu # Wait, penalty is sum(viol^2). d/dx (mu * sum(viol^2)) = mu * sum(2*viol*grad_viol)
        # My derivation above:
        # d/dx (mu * c1_viol^2) = mu * 2 * c1_viol * d(c1_viol)/dx
        # d(c1_viol)/dx = -2(x_i - x_j) (if violated)
        # So term is mu * 2 * c1_viol * (-2 diff) = -4 mu c1_viol diff.
        # I calculated grad_x contribution as -4 * sum(c1_viol * diff).
        # So I need to multiply by mu.
        
        # Let's restart gradient calculation carefully.
        grad_x = np.zeros_like(x)
        grad_r = -1.0
        
        if np.any(c1_viol > 0):
            # Term: mu * sum_{i,j} (c1_viol_ij)^2
            # d/dx_i = mu * sum_j 2 * c1_viol_ij * d(c1_viol_ij)/dx_i
            # d(c1_viol_ij)/dx_i = -2(x_i - x_j)
            # = mu * sum_j 2 * c1_viol_ij * (-2(x_i - x_j))
            # = -4 mu sum_j c1_viol_ij (x_i - x_j)
            grad_x += -4.0 * mu * np.sum(c1_viol[:, :, np.newaxis] * diff, axis=1)
            
            # d/dr = mu * sum_{i,j} 2 * c1_viol_ij * d(c1_viol_ij)/dr
            # d(c1_viol_ij)/dr = 8r
            # = mu * sum_{i,j} 16r c1_viol_ij
            grad_r += 16.0 * mu * r * np.sum(c1_viol)

        if np.any(c2_viol > 0):
            # Term: mu * sum_i (c2_viol_i)^2
            # d/dx_i = mu * 2 * c2_viol_i * d(c2_viol_i)/dx_i
            # d(c2_viol_i)/dx_i = 2x_i
            # = mu * 4 c2_viol_i x_i
            grad_x += 4.0 * mu * c2_viol[:, np.newaxis] * x
            
            # d/dr = mu * sum_i 2 * c2_viol_i * d(c2_viol_i)/dr
            # d(c2_viol_i)/dr = 2(1-r)
            # = mu * sum_i 4 (1-r) c2_viol_i
            grad_r += 4.0 * mu * (1.0 - r) * np.sum(c2_viol)

        grad = np.concatenate([grad_x.flatten(), [grad_r]])
        return obj, grad, pen

    # Main SQP Loop
    # We will run multiple restarts.
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            x = init_points.copy()
            r = get_score(x)
        else:
            # Perturb best solution
            x = best_points + np.random.randn(N, 2) * 0.05
            r = get_score(x)
        
        v = flatten(x, r)
        
        # Adaptive mu
        mu = 1.0
        max_mu = 1e6
        
        # SQP iterations
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
                
            obj, grad, pen = compute_penalty_and_grad(v, mu)
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-6 and pen < 1e-8:
                break
                
            # Newton step? Or simple gradient descent with line search?
            # For N=50, Hessian is 101x101. We can approximate Hessian or use BFGS.
            # Given time constraints, let's use a simple gradient descent with adaptive step size.
            # Or even better, use scipy.optimize.minimize with BFGS on the penalized objective.
            # But we want to control mu.
            # Let's do a few steps of gradient descent with line search.
            
            # Step size
            step = 0.01
            max_step = 0.1
            min_step = 1e-6
            
            improved = False
            for ls in range(10):
                v_new = v - step * grad
                # Ensure r is positive? Not strictly necessary if penalty handles it, but good for stability.
                # v_new[2*N] = max(0.01, v_new[2*N]) 
                
                obj_new, grad_new, pen_new = compute_penalty_and_grad(v_new, mu)
                
                if obj_new < obj:
                    v = v_new
                    improved = True
                    break
                else:
                    step *= 0.5
                    
            if not improved:
                # Increase penalty if not improving
                mu *= 10.0
                if mu > max_mu:
                    break
            else:
                # If penalty is small, we can decrease mu to focus on objective
                if pen < 1e-4 and mu > 1.0:
                    mu *= 0.5

        # Update best
        x_final, r_final = unflatten(v)
        # Ensure validity
        score = get_score(x_final)
        if score > best_score:
            best_score = score
            best_points = x_final.copy()
            
        # If we have time, do a local polish on the best point found so far
        # This is a structural change: using the SQP result as a starting point for a high-precision local search.

    # Final Polish
    if best_points is not None and time.time() < deadline - 2.0:
        # Use scipy minimize on the penalized objective with high mu
        def final_obj(v):
            x, r = unflatten(v)
            _, _, pen = compute_penalty_and_grad(v, 1e5)
            return -r + 1e5 * pen
        
        def final_grad(v):
            _, grad, _ = compute_penalty_and_grad(v, 1e5)
            return grad

        from scipy.optimize import minimize
        x0 = best_points.flatten()
        r0 = best_score
        v0 = flatten(best_points, r0)
        
        try:
            res = minimize(final_obj, v0, jac=final_grad, method='BFGS', options={'gtol': 1e-8, 'maxiter': 100})
            if res.success:
                x_polished, r_polished = unflatten(res.x)
                score_polished = get_score(x_polished)
                if score_polished > best_score:
                    best_score = score_polished
                    best_points = x_polished.copy()
        except Exception:
            pass

    if best_points is None:
        return init_points
    return best_points
# EVOLVE-BLOCK-END
