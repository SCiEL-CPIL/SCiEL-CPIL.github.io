import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB f12 (Bent Cigar).
    
    Strategy:
    1. Quadratic Surrogate Model: Fits a local quadratic model to collected points 
       to estimate the valley direction and take large Newton-like steps.
    2. Hooke-Jeeves Refinement: When the surrogate model stagnates or the step 
       becomes small, switches to coordinate-wise search for precise local minimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')
        
        # Store history for quadratic fitting
        # We need at least d*(d+1)/2 + 1 points for a full quadratic, but we can use ridge regression with fewer
        points = []
        
        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            
            x = np.clip(x, -5, 5)
            # Ensure x is a 1D array of floats
            x = np.asarray(x, dtype=float).flatten()
            
            try:
                f = float(func(x))
            except Exception:
                return float('inf')
                
            if not np.isfinite(f):
                return float('inf')

            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
            # Add to history for surrogate model
            # We keep a limited history to prevent memory issues and stale data
            if len(points) < 100:
                points.append((x, f))
            else:
                # Keep the best 50 and most recent 50? 
                # Simpler: just append and let the fitter use the most relevant ones.
                # For efficiency, we'll just keep the last 100.
                points.pop(0)
                points.append((x, f))
                
            return f

        def hooke_jees(start_x, initial_f=None):
            """
            Standard Hooke-Jeeves pattern search.
            Used for local refinement when surrogate model fails.
            """
            nonlocal eval_count
            if eval_count >= budget - 5:
                return

            x = np.clip(start_x, -5, 5).copy()
            if initial_f is not None:
                f = initial_f
            else:
                f = eval_func(x)
                if eval_count >= budget:
                    return

            step_sizes = np.ones(d) * 0.5
            min_step = 1e-5
            max_iter = 100

            for _ in range(max_iter):
                if eval_count >= budget or time.time() > deadline:
                    break

                moved = False
                for i in range(d):
                    if eval_count >= budget or time.time() > deadline:
                        break

                    # Try positive step
                    x_plus = x.copy()
                    x_plus[i] += step_sizes[i]
                    f_plus = eval_func(x_plus)
                    if eval_count >= budget:
                        break

                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        step_sizes[i] *= 2.0
                        moved = True
                        # Continue exploring this direction if it improved
                        # In standard HJ, we might loop back to i, but for efficiency 
                        # and to avoid getting stuck in oscillation, we proceed to next i 
                        # unless we want aggressive exploitation. 
                        # Given the narrow valley, aggressive exploitation in the 
                        # aligned direction is good, but HJ is coordinate-wise.
                        # We stick to the standard coordinate scan for robustness.
                    else:
                        step_sizes[i] *= 0.5

                    # Try negative step
                    x_minus = x.copy()
                    x_minus[i] -= step_sizes[i]
                    f_minus = eval_func(x_minus)
                    if eval_count >= budget:
                        break

                    if f_minus < f:
                        x = x_minus
                        f = f_minus
                        step_sizes[i] *= 2.0
                        moved = True
                    else:
                        step_sizes[i] *= 0.5

                if np.all(step_sizes < min_step):
                    break
                if not moved:
                    # If no move in a full cycle, we are likely converged or stuck.
                    # Break to save budget.
                    break

        # --- Phase 1: Global Search with Quadratic Surrogate ---
        
        # Initialize with a few random points to get a sense of the landscape
        rng = np.random.default_rng(42)
        
        # Start at origin
        eval_func(np.zeros(d))
        
        # Add a few random points to build the initial model
        for _ in range(10):
            if eval_count >= budget:
                break
            x_rand = rng.uniform(-5, 5, d)
            eval_func(x_rand)

        # Main loop: Alternate between surrogate steps and local refinement
        surrogate_fail_count = 0
        min_surrogate_step = 1e-3

        while eval_count < budget - 10:
            if time.time() > deadline:
                break
            
            # Check if we have enough points to fit a quadratic
            # d=5, quadratic has 1 + 5 + 10 = 16 params. 
            # We need at least 16 points for a unique fit, but ridge regression works with fewer.
            # Let's require at least d + 1 points to start, and use ridge for stability.
            
            if len(points) < d + 1:
                # Not enough points, do a random step
                x_new = best_x + rng.normal(0, 1.0, d)
                f_new = eval_func(x_new)
                if f_new < best_f - 1e-8:
                    surrogate_fail_count = 0
                else:
                    surrogate_fail_count += 1
                continue

            # Build design matrix for quadratic fit
            # f(x) ~ c0 + c1^T x + x^T C x
            # We will use a ridge regression approach.
            
            X = []
            y = []
            
            # Use the most recent points, but ensure we have enough
            # For d=5, 16 params. Let's use up to 30 points for stability.
            fit_points = points[-30:] if len(points) > 30 else points
            
            for x, f in fit_points:
                row = [1.0]
                row.extend(x)
                # Add quadratic terms x_i * x_j for i <= j
                for i in range(d):
                    for j in range(i, d):
                        row.append(x[i] * x[j])
                X.append(row)
                y.append(f)
                
            X = np.array(X)
            y = np.array(y)
            
            # Add regularization (Ridge) to handle ill-conditioning
            lambda_reg = 1e-6
            try:
                # Solve (X^T X + lambda I) beta = X^T y
                # Note: The intercept term (1.0) should not be regularized as strongly, 
                # but for simplicity, we regularize all or just the higher order terms.
                # Let's regularize the matrix part.
                A = X.T @ X
                b = X.T @ y
                
                # Regularize the quadratic and linear terms (indices 1 onwards)
                # Index 0 is the constant term.
                reg_matrix = np.eye(A.shape[0]) * lambda_reg
                reg_matrix[0, 0] = 0.0 # No regularization on constant
                
                beta = np.linalg.solve(A + reg_matrix, b)
                
                # Extract coefficients
                c0 = beta[0]
                c1 = beta[1:1+d]
                
                # Reconstruct C from beta[1+d:]
                # The order in X construction was: i=0..d-1, j=i..d-1
                C = np.zeros((d, d))
                idx = 1 + d
                for i in range(d):
                    for j in range(i, d):
                        val = beta[idx]
                        C[i, j] = val
                        C[j, i] = val
                        idx += 1
                        
                # Find minimum of the quadratic model
                # grad f = c1 + 2 C x = 0 => x* = -0.5 C^{-1} c1
                # Check if C is positive definite
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) > 1e-6:
                        # C is PD, so x* is a minimum
                        x_opt = -0.5 * np.linalg.solve(C, c1)
                    else:
                        # Not a minimum (saddle or max), move in direction of negative gradient
                        # grad = c1 + 2 C x
                        # We are at best_x. Let's just move a bit in -grad direction
                        grad = c1 + 2 * C @ best_x
                        norm_grad = np.linalg.norm(grad)
                        if norm_grad > 1e-8:
                            x_opt = best_x - 0.5 * grad / norm_grad
                        else:
                            # Stagnant
                            x_opt = best_x
                except np.linalg.LinAlgError:
                    x_opt = best_x

            except np.linalg.LinAlgError:
                # Fitting failed
                x_opt = best_x

            # Clip x_opt to bounds
            x_opt = np.clip(x_opt, -5, 5)
            
            # Check step size
            step_norm = np.linalg.norm(x_opt - best_x)
            
            if step_norm < min_surrogate_step:
                # Surrogate model suggests we are converged or step is too small
                # Switch to local refinement
                hooke_jees(best_x, best_f)
                break

            # Take the step
            f_new = eval_func(x_opt)
            
            if f_new < best_f - 1e-8:
                surrogate_fail_count = 0
            else:
                surrogate_fail_count += 1
                # If surrogate fails multiple times, it's likely inaccurate
                # Switch to local refinement to stabilize
                if surrogate_fail_count >= 3:
                    hooke_jees(best_x, best_f)
                    break

        # --- Phase 2: Final Polish ---
        # If we still have budget, do a final HJ polish on the best point found
        if eval_count < budget - 5:
            hooke_jees(best_x, best_f)

        return best_x, best_f
