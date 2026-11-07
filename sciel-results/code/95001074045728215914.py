"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Surrogate-assisted optimizer for the Bent Cigar (f12) problem.
    
    Strategy:
    1. Use a local quadratic surrogate model (paraboloid) fitted to recent points.
    2. The surrogate predicts the next step toward the minimum.
    3. Fall back to random search or simple gradient-free steps if the surrogate
       is ill-conditioned or fails to improve.
    
    The Bent Cigar function is highly anisotropic: f(x) = 100*(x1-0.5)^2 + sum_{i=2}^n (x_i - 1/i)^2
    (in its standard form, though here it's shifted/rotated). The key feature is that one direction
    is much steeper than the others. A quadratic model can capture this curvature locally and
    take large steps in the flat directions while being careful in the steep direction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initialize with a few random points to build an initial model
        # Use a small number of initial random evaluations to get a sense of the landscape
        X = []
        F = []
        
        # Start with a random point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        X.append(x)
        F.append(fx)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Keep a buffer of recent points for the surrogate model
        # We'll use up to 2*dim+1 points for a quadratic model (dim*(dim+1)/2 + dim + 1 coefficients)
        # Actually, a full quadratic in dim dims has dim*(dim+1)/2 + dim + 1 = (dim^2+3dim+2)/2 coefficients
        # For dim=5: (25+15+2)/2 = 21 coefficients, so we need at least 21 points
        # But we can use a simpler approach: fit a quadratic to the last K points where K is manageable
        
        # Let's use a simpler strategy: maintain a set of recent points and fit a local quadratic
        # using least squares when we have enough points
        
        # For the bent cigar, a good approach is:
        # 1. Estimate the gradient direction using finite differences or the surrogate
        # 2. Take a step in that direction with adaptive step size
        
        # Let's implement a surrogate-based approach:
        # - Keep the last 30 points
        # - Fit a quadratic model: f(x) ≈ c0 + g^T x + x^T H x
        # - Find the minimizer of the surrogate within the box
        # - Take a step toward that minimizer
        
        # For efficiency, we'll use a simplified approach:
        # Fit a quadratic to recent points and predict the next point
        
        # Number of points needed for a full quadratic in dim=5: 21
        # Let's start with random search until we have enough points, then switch to surrogate
        
        min_points_for_model = dim * (dim + 1) // 2 + dim + 1  # 21 for dim=5
        
        # Phase 1: Random search to gather points
        while len(X) < min(min_points_for_model, budget) and calls < budget:
            x_new = rng.uniform(-5, 5, dim)
            fx_new = func(x_new)
            X.append(x_new)
            F.append(fx_new)
            calls += 1
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
        
        # Phase 2: Surrogate-guided search
        # Keep only the last K points for the model
        K = min(len(X), 40)  # Use last 40 points for the model
        
        # Current best point
        x = best_x.copy()
        fx = best_f
        
        # Adaptive step size
        sigma = 1.0
        
        while calls < budget:
            # If we have enough points, fit a quadratic model
            if len(X) >= min_points_for_model:
                # Use the last K points for fitting
                X_model = np.array(X[-K:])
                F_model = np.array(F[-K:])
                
                # Fit a quadratic model: f(x) ≈ c0 + g^T x + x^T H x
                # Using least squares
                # Design matrix: [1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]
                d = dim
                n_params = 1 + d + d * (d + 1) // 2
                
                # Build design matrix
                A = np.ones((len(X_model), n_params))
                for j in range(d):
                    A[:, 1 + j] = X_model[:, j]
                
                # Quadratic terms
                idx = 1 + d
                for i in range(d):
                    for j in range(i, d):
                        A[:, idx] = X_model[:, i] * X_model[:, j]
                        idx += 1
                
                # Solve least squares
                try:
                    coeffs, residuals, rank, sv = np.linalg.lstsq(A, F_model, rcond=None)
                    
                    # Check if the fit is reasonable
                    if rank < n_params:
                        # Ill-conditioned, fall back to random step
                        raise np.linalg.LinAlgError("Rank deficient")
                    
                    # Extract parameters
                    c0 = coeffs[0]
                    g = coeffs[1:1+d]
                    H_flat = coeffs[1+d:]
                    
                    # Reconstruct H (symmetric)
                    H = np.zeros((d, d))
                    idx = 0
                    for i in range(d):
                        for j in range(i, d):
                            H[i, j] = H_flat[idx]
                            if i != j:
                                H[j, i] = H_flat[idx]
                            idx += 1
                    
                    # The quadratic model is: f(x) = c0 + g^T x + x^T H x
                    # To find the minimizer, we need to solve: (2H) x + g = 0
                    # i.e., x_min = -0.5 * inv(2H) * g = -0.5 * inv(H) * g
                    # But we need to be careful: the model is f(x) = c0 + g^T x + x^T H x
                    # The gradient is: g + 2Hx = 0 => x = -0.5 * H^{-1} g
                    
                    # Check if H is positive definite (for a minimum)
                    eigvals = np.linalg.eigvalsh(H)
                    if np.all(eigvals > -1e-6):
                        # H is positive semi-definite, we can find the minimizer
                        try:
                            # Solve H * x_min = -0.5 * g
                            x_min = np.linalg.solve(H, -0.5 * g)
                        except np.linalg.LinAlgError:
                            # Use pseudo-inverse
                            x_min = np.linalg.lstsq(H, -0.5 * g, rcond=None)[0]
                        
                        # Clip to the box
                        x_min = np.clip(x_min, -5, 5)
                        
                        # Take a step toward x_min from current best
                        # Use a damped step to avoid overshooting
                        step = x_min - x
                        step_norm = np.linalg.norm(step)
                        if step_norm > 1e-10:
                            # Limit the step size
                            max_step = 2.0
                            if step_norm > max_step:
                                step = step / step_norm * max_step
                            x_candidate = np.clip(x + step, -5, 5)
                        else:
                            x_candidate = x_min
                    else:
                        # H is not positive definite, use gradient direction instead
                        # Take a step in the negative gradient direction
                        grad = g + 2 * H @ x
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm > 1e-10:
                            x_candidate = np.clip(x - sigma * grad / grad_norm, -5, 5)
                        else:
                            x_candidate = rng.uniform(-5, 5, dim)
                
                except (np.linalg.LinAlgError, Exception):
                    # Fallback: random step
                    x_candidate = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            else:
                # Not enough points for a model, use random search
                x_candidate = rng.uniform(-5, 5, dim)
            
            # Evaluate the candidate
            fx_candidate = func(x_candidate)
            calls += 1
            
            # Update best
            if fx_candidate < best_f:
                best_f = fx_candidate
                best_x = x_candidate.copy()
                x = x_candidate.copy()
                fx = fx_candidate
                # Increase step size slightly on success
                sigma = min(sigma * 1.2, 5.0)
            else:
                # Decrease step size on failure
                sigma = max(sigma * 0.8, 0.1)
            
            # Add to history
            X.append(x_candidate)
            F.append(fx_candidate)
            
            # Keep history bounded
            if len(X) > 100:
                X = X[-100:]
                F = F[-100:]
        
        return best_x, best_f
# EVOLVE-BLOCK-END
