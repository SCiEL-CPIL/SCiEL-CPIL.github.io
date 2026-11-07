"""Rewritten program using Levenberg-Marquardt optimization for robust root finding."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses the Levenberg-Marquardt (LM) algorithm.
    """
    x = np.array(x0, dtype=float)
    eps = 1e-6  # Step for finite differences
    
    def compute_jacobian(x):
        """Compute Jacobian using central finite differences."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            xp = x + e
            xm = x - e
            Fp = F(xp)
            Fm = F(xm)
            J[:, j] = (Fp - Fm) / (2 * eps)
        return J

    # Initial evaluation
    Fx = F(x)
    fx_norm_sq = np.dot(Fx, Fx)
    if fx_norm_sq <= 1e-16: # (1e-8)^2
        return x

    # LM parameters
    lambda_lm = 1e-3  # Initial damping factor
    lambda_min = 1e-12
    lambda_max = 1e12
    max_iter = 200
    max_inner_iter = 20
    
    I = np.eye(2)
    
    for _ in range(max_iter):
        # Check convergence
        if fx_norm_sq <= 1e-16:
            return x
            
        # Compute Jacobian
        J = compute_jacobian(x)
        
        # Compute J^T J and J^T F
        JtJ = J.T @ J
        JtF = J.T @ Fx
        
        # Inner loop to find a good lambda
        accepted = False
        current_lambda = lambda_lm
        
        for _ in range(max_inner_iter):
            try:
                A = JtJ + current_lambda * I
                # Solve for step delta
                delta = np.linalg.solve(A, -JtF)
            except np.linalg.LinAlgError:
                # If singular, increase lambda
                current_lambda *= 10.0
                if current_lambda > lambda_max:
                    # Fallback to gradient descent
                    grad = -JtF
                    norm_grad = np.linalg.norm(grad)
                    if norm_grad > 1e-10:
                        delta = -grad / norm_grad * 0.1
                    else:
                        return x
                    # Evaluate fallback
                    x_new = x + delta
                    F_new = F(x_new)
                    fx_norm_sq_new = np.dot(F_new, F_new)
                    if fx_norm_sq_new < fx_norm_sq:
                        x = x_new
                        Fx = F_new
                        fx_norm_sq = fx_norm_sq_new
                        lambda_lm = max(lambda_lm / 10.0, lambda_min)
                    else:
                        lambda_lm = min(lambda_lm * 10.0, lambda_max)
                    accepted = True
                    break
                continue

            # Evaluate new point
            x_new = x + delta
            F_new = F(x_new)
            fx_norm_sq_new = np.dot(F_new, F_new)
            
            if fx_norm_sq_new < fx_norm_sq:
                # Accept step
                x = x_new
                Fx = F_new
                fx_norm_sq = fx_norm_sq_new
                # Decrease lambda (move towards Newton)
                lambda_lm = max(current_lambda / 10.0, lambda_min)
                accepted = True
                break
            else:
                # Reject step, increase lambda (move towards Gradient Descent)
                current_lambda = min(current_lambda * 10.0, lambda_max)
                if current_lambda >= lambda_max:
                     # If we've hit max lambda and still failing, update lambda and break
                     lambda_lm = min(lambda_lm * 10.0, lambda_max)
                     accepted = True
                     break

    return x
# EVOLVE-BLOCK-END
